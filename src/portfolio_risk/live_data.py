"""Optional live market data loader (yfinance).

The pipeline runs on the bundled synthetic dataset by default; pass --live to
run_analysis.py to pull real adjusted closes for the same tickers instead.
The output schema matches the generator exactly, so everything downstream
(SQL, metrics, charts) is agnostic to where the data came from.
"""
from __future__ import annotations

import pandas as pd

from . import config


def fetch_live_prices(
    tickers: list[str] | None = None, years: int = 5, start: str | None = None
) -> pd.DataFrame:
    """Download daily adjusted closes + volume; return long-format DataFrame.

    Pulls the last `years` years, or everything from `start` (YYYY-MM-DD) if given.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "yfinance is required for live data: pip install yfinance "
            "(or run without --live to use the bundled demo dataset)"
        ) from exc

    tickers = tickers or config.TICKERS
    raw = yf.download(
        tickers=" ".join(tickers),
        **({"start": start} if start else {"period": f"{years}y"}),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data - check network access")

    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    volumes = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    long = (
        closes.round(4)
        .stack()
        .rename("close")
        .reset_index()
        .rename(columns={"Date": "date", "level_1": "ticker", "Ticker": "ticker"})
    )
    vol_long = (
        volumes.stack()
        .rename("volume")
        .reset_index()
        .rename(columns={"Date": "date", "level_1": "ticker", "Ticker": "ticker"})
    )
    long = long.merge(vol_long, on=["date", "ticker"], how="left")
    long["volume"] = long["volume"].fillna(0).astype("int64")
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long = long.dropna(subset=["close"])
    return long[["date", "ticker", "close", "volume"]].sort_values(["ticker", "date"])
