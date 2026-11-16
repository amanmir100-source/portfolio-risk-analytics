"""SQLite persistence + SQL analytics layer.

All analytical SQL lives in standalone .sql files under sql/ (window
functions, CTEs, ranking). This module owns the plumbing: schema creation,
bulk loading, and running query files into pandas DataFrames.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# repo_root/sql  (this file lives at repo_root/src/portfolio_risk/database.py)
DEFAULT_SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

ANALYTICS_QUERIES = [
    "daily_returns",
    "rolling_volatility",
    "monthly_performance",
    "drawdown_events",
    "asset_summary",
    "data_quality",
]

# Rolling volatility is a long chain of float sums, so different SQLite
# builds disagree in the last digit or two. Rounding in Python (SQLite's own
# ROUND() isn't consistent across versions either) keeps the CSV identical on
# every machine.
ROUND_DECIMALS = {"rolling_volatility": 10}


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, always close.

    sqlite3's own context manager only commits; it leaves the connection
    open, which leaks file handles and keeps the .db locked on Windows.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def content_hash(prices: pd.DataFrame) -> str:
    """SHA-256 of the price rows in a fixed order, to fingerprint a load."""
    ordered = prices.sort_values(["ticker", "date"])[["date", "ticker", "close", "volume"]]
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def read_sql_file(name: str, sql_dir: str | Path | None = None) -> str:
    sql_dir = Path(sql_dir) if sql_dir else DEFAULT_SQL_DIR
    path = sql_dir / f"{name}.sql"
    return path.read_text(encoding="utf-8")


def initialize_database(
    db_path: str | Path,
    prices: pd.DataFrame,
    assets: pd.DataFrame,
    weights: pd.DataFrame,
    sql_dir: str | Path | None = None,
    source: str = "unspecified",
) -> None:
    """Create the schema and reload reference + price data in one transaction.

    This is a full refresh on purpose. The live feed serves ADJUSTED closes,
    which the vendor restates backwards after every dividend, so appending
    only new dates would splice together prices adjusted on different days
    and create a false jump at the join. Reloading everything keeps the
    series consistent; the transaction means a failed load leaves the
    previous data in place; and each load is recorded in load_log.

    Rows go in with plain executemany rather than pandas' to_sql: to_sql
    commits after every table, which would break the single transaction, and
    its if_exists="replace" would drop the PRIMARY KEY / FOREIGN KEY / CHECK
    constraints.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(read_sql_file("schema", sql_dir))
    with get_connection(db_path) as conn:
        for table in ("prices", "portfolio_weights", "assets"):
            conn.execute(f"DELETE FROM {table};")
        _insert(conn, "assets", assets)
        _insert(conn, "portfolio_weights", weights)
        _insert(conn, "prices", prices)
        conn.execute(read_sql_file("record_load", sql_dir), (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source,
            len(prices),
            str(prices["date"].min()),
            str(prices["date"].max()),
            content_hash(prices),
        ))


def _insert(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    """Bulk-insert a DataFrame into an existing table, inside the open transaction."""
    columns = ", ".join(frame.columns)
    marks = ", ".join("?" for _ in frame.columns)
    rows = frame.astype(object).where(frame.notna(), None).itertuples(index=False, name=None)
    conn.executemany(f"INSERT INTO {table} ({columns}) VALUES ({marks})", rows)


def run_query(db_path: str | Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def run_query_file(
    db_path: str | Path, name: str, sql_dir: str | Path | None = None
) -> pd.DataFrame:
    """Execute sql/<name>.sql and return the result as a DataFrame."""
    return run_query(db_path, read_sql_file(name, sql_dir))


def run_all_analytics(
    db_path: str | Path,
    out_dir: str | Path | None = None,
    sql_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every analytical query; optionally dump each result to CSV."""
    results: dict[str, pd.DataFrame] = {}
    for name in ANALYTICS_QUERIES:
        results[name] = run_query_file(db_path, name, sql_dir)
        if name in ROUND_DECIMALS:
            results[name] = results[name].round(ROUND_DECIMALS[name])
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            results[name].to_csv(out / f"{name}.csv", index=False)
    return results


def table_row_counts(db_path: str | Path) -> dict[str, int]:
    """Small sanity helper used by the CLI and tests."""
    counts = {}
    with get_connection(db_path) as conn:
        for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts
