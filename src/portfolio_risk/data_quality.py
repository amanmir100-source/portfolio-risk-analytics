"""Data-quality checks run on prices as they load.

Structural problems (duplicate rows, zero/negative/missing closes) stop the
run, because every number downstream would be wrong. Everything else is a
warning the run carries on past:

* missing closes - dates where some assets have a price and others don't
  (sql/data_quality.sql; missing_closes() here is its pandas twin)
* extreme moves  - one-day moves beyond MAX_DAILY_MOVE, more likely a bad
  tick than a real move for these ETFs
* stale prices   - the same close STALE_DAYS or more days in a row, which
  usually means the feed stopped updating
"""
from __future__ import annotations

import pandas as pd

MAX_DAILY_MOVE = 0.25
STALE_DAYS = 5


class DataQualityError(ValueError):
    """Raised when the price data is too broken to analyse."""


def check_structure(prices: pd.DataFrame) -> None:
    """Raise DataQualityError on duplicate rows or unusable closes."""
    problems = []
    dupes = prices[prices.duplicated(["date", "ticker"], keep=False)]
    if len(dupes):
        sample = dupes[["date", "ticker"]].drop_duplicates().head(5)
        problems.append(f"{len(dupes)} duplicate date/ticker rows, e.g. "
                        + ", ".join(f"{d} {t}" for d, t in sample.itertuples(index=False)))
    bad = prices[~(prices["close"] > 0)]  # catches <= 0 and NaN
    if len(bad):
        sample = bad[["date", "ticker", "close"]].head(5)
        problems.append(f"{len(bad)} zero, negative or missing closes, e.g. "
                        + ", ".join(f"{d} {t} ({c})" for d, t, c in sample.itertuples(index=False)))
    if problems:
        raise DataQualityError("price data failed checks: " + "; ".join(problems))


def missing_closes(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Dates where some of `tickers` have no close (pandas twin of the SQL)."""
    have = prices.groupby("date")["ticker"].agg(set)
    rows = []
    for date, present in have.sort_index().items():
        gone = sorted(set(tickers) - present)
        if gone:
            rows.append({"date": date, "missing_count": len(gone),
                         "missing_tickers": ", ".join(gone)})
    return pd.DataFrame(rows, columns=["date", "missing_count", "missing_tickers"])


def extreme_moves(prices: pd.DataFrame, threshold: float = MAX_DAILY_MOVE) -> pd.DataFrame:
    """One-day moves (each ticker against its own previous close) beyond threshold."""
    p = prices.sort_values(["ticker", "date"])
    move = p.groupby("ticker")["close"].transform(lambda s: s / s.shift(1) - 1.0)
    out = p.assign(daily_return=move)[move.abs() > threshold]
    return out[["date", "ticker", "daily_return"]].reset_index(drop=True)


def stale_prices(prices: pd.DataFrame, days: int = STALE_DAYS) -> pd.DataFrame:
    """Runs of `days` or more identical closes in a row for one ticker."""
    rows = []
    for ticker, s in prices.sort_values("date").groupby("ticker"):
        run_id = (s["close"] != s["close"].shift(1)).cumsum()
        for _, run in s.groupby(run_id):
            if len(run) >= days:
                rows.append({"ticker": ticker, "start": run["date"].iloc[0],
                             "end": run["date"].iloc[-1], "days": len(run),
                             "close": run["close"].iloc[0]})
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "days", "close"])


def warnings_report(prices: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    """All warnings in one table: check, date, ticker, detail."""
    rows = [{"check": "missing_close", "date": r.date, "ticker": r.missing_tickers,
             "detail": f"{r.missing_count} asset(s) have no close"}
            for r in missing.itertuples(index=False)]
    rows += [{"check": "extreme_move", "date": r.date, "ticker": r.ticker,
              "detail": f"{r.daily_return:+.1%} in one day"}
             for r in extreme_moves(prices).itertuples(index=False)]
    rows += [{"check": "stale_price", "date": f"{r.start} to {r.end}", "ticker": r.ticker,
              "detail": f"close {r.close} unchanged for {r.days} days"}
             for r in stale_prices(prices).itertuples(index=False)]
    return pd.DataFrame(rows, columns=["check", "date", "ticker", "detail"])
