"""Dated snapshots of a live data pull, so real-data results can be re-run.

Yahoo restates adjusted closes after every dividend, so the same command run
a month later gives slightly different history. A snapshot freezes one pull:
the analysis-window prices, the long history the stress tests need, and the
risk-free rate used. `run_analysis.py --snapshot DIR` then reproduces that
run offline, number for number.

    data/snapshots/<as_of>/
        prices.csv     long format: date, ticker, close, volume
        history.csv    same columns, back to stress.HISTORY_START
        meta.json      as-of date, source, risk-free rate, row counts, hashes
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .database import content_hash

PRICE_COLUMNS = ["date", "ticker", "close", "volume"]


@dataclass
class Snapshot:
    as_of: str
    source: str
    risk_free_rate: float
    prices: pd.DataFrame
    history: pd.DataFrame


def save_snapshot(
    root: str | Path,
    as_of: str,
    prices: pd.DataFrame,
    history: pd.DataFrame,
    risk_free_rate: float,
    source: str,
) -> Path:
    """Write one snapshot under root/<as_of>/ and return that directory."""
    out = Path(root) / as_of
    out.mkdir(parents=True, exist_ok=True)
    prices[PRICE_COLUMNS].to_csv(out / "prices.csv", index=False)
    history[PRICE_COLUMNS].to_csv(out / "history.csv", index=False)
    meta = {
        "as_of": as_of,
        "source": source,
        "risk_free_rate": risk_free_rate,
        "tickers": sorted(prices["ticker"].unique()),
        "prices": _describe(prices),
        "history": _describe(history),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def load_snapshot(directory: str | Path) -> Snapshot:
    """Read a snapshot back, refusing files that no longer match their hash."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    frames = {}
    for name in ("prices", "history"):
        frame = pd.read_csv(directory / f"{name}.csv")
        if content_hash(frame) != meta[name]["content_sha256"]:
            raise ValueError(
                f"{directory / (name + '.csv')} was changed after the snapshot was taken")
        frames[name] = frame
    return Snapshot(
        as_of=meta["as_of"],
        source=meta["source"],
        risk_free_rate=float(meta["risk_free_rate"]),
        prices=frames["prices"],
        history=frames["history"],
    )


def _describe(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "content_sha256": content_hash(frame),
    }
