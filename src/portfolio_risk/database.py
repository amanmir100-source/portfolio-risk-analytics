"""SQLite persistence + SQL analytics layer.

All analytical SQL lives in standalone .sql files under sql/ (window
functions, CTEs, ranking). This module owns the plumbing: schema creation,
bulk loading, and running query files into pandas DataFrames.
"""
from __future__ import annotations

import sqlite3
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
]


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


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
) -> None:
    """Create the schema and (re)load reference + price data.

    Uses DELETE + append rather than pandas' if_exists="replace" so the
    tables keep their PRIMARY KEY / FOREIGN KEY / CHECK constraints.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(read_sql_file("schema", sql_dir))
        for table in ("prices", "portfolio_weights", "assets"):
            conn.execute(f"DELETE FROM {table};")
        assets.to_sql("assets", conn, if_exists="append", index=False)
        weights.to_sql("portfolio_weights", conn, if_exists="append", index=False)
        prices.to_sql("prices", conn, if_exists="append", index=False)
        conn.commit()


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
