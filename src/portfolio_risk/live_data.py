"""Optional live market data loader (yfinance).

The pipeline runs on the bundled synthetic dataset by default; pass --live to
run_analysis.py to pull real adjusted closes for the same tickers instead.
The output schema matches the generator exactly, so everything downstream
(SQL, metrics, charts) is agnostic to where the data came from.
"""
from __future__ import annotations

import pandas as pd

from . import config

RISK_FREE_TICKER = "^IRX"  # CBOE 13-week Treasury bill yield, quoted in percent


def _yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "yfinance is required for live data: pip install yfinance "
            "(or run without --live to use the bundled demo dataset)"
        ) from exc
    return yf


def fetch_risk_free_rate(start: str, end: str, ticker: str = RISK_FREE_TICKER) -> float:
    """Average annualised 3-month T-bill yield between two dates, as a fraction.

    Sharpe's numerator is the mean return in excess of cash. The mean of
    (return - cash rate) equals mean return minus the AVERAGE cash rate, so
    one window-average number is the right hurdle for a whole-period Sharpe.
    """
    yf = _yfinance()
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers=ticker, start=start, end=end_exclusive, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(
            f"no {ticker} data between {start} and {end}; pass --risk-free RATE to set it by hand"
        )
    closes = pd.DataFrame(raw["Close"]).iloc[:, 0].dropna()
    rate = float(closes.mean()) / 100.0
    if not 0.0 <= rate < 0.25:
        raise RuntimeError(f"implausible risk-free rate {rate:.4f} from {ticker}")
    return rate


def fetch_live_prices(
    tickers: list[str] | None = None, years: int = 5, start: str | None = None
) -> pd.DataFrame:
    """Download daily adjusted closes + volume; return long-format DataFrame.

    Pulls the last `years` years, or everything from `start` (YYYY-MM-DD) if given.
    """
    yf = _yfinance()
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
