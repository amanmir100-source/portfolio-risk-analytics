"""Risk & performance metrics built on NumPy/pandas.

Conventions
-----------
* Returns are daily simple returns.
* Annualisation uses 252 trading days; annualised return is geometric.
* VaR / CVaR are reported as POSITIVE numbers representing a loss at the
  given confidence level (e.g. var = 0.018 -> "we expect to lose more than
  1.8% on the worst 5% of days").
* Parametric VaR uses the normal quantile from the standard library
  (statistics.NormalDist) - no SciPy dependency required.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_TICKER,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)

_ANN = TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def to_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long [date, ticker, close] rows into a date x ticker price matrix."""
    wide = prices_long.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def daily_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns between consecutive dates where every asset has a close.

    A date missing a close for any asset is dropped *before* differencing, so
    every return spans the same interval for every asset. Uses an explicit
    shift rather than pct_change(), whose gap-filling default differs across
    pandas versions and would make results depend on the installed pandas.
    """
    prices = wide_prices.sort_index().dropna(how="any")
    return (prices / prices.shift(1) - 1.0).iloc[1:]


def portfolio_returns(
    returns: pd.DataFrame, weights: dict[str, float] | pd.Series
) -> pd.Series:
    """Fixed-weight (daily-rebalanced) portfolio return series."""
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise KeyError(f"weights reference tickers absent from returns: {missing}")
    return returns[list(w.index)].mul(w, axis=1).sum(axis=1).rename("PORTFOLIO")


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
def annualized_return(r: pd.Series) -> float:
    """Geometric annualised return: (prod(1+r))^(252/n) - 1."""
    r = pd.Series(r).dropna()
    total_growth = float((1.0 + r).prod())
    return total_growth ** (_ANN / len(r)) - 1.0


def annualized_volatility(r: pd.Series) -> float:
    """Sample stdev of daily returns scaled by sqrt(252)."""
    return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(_ANN))


def sharpe_ratio(r: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """(annualised return - rf) / annualised volatility."""
    vol = annualized_volatility(r)
    return (annualized_return(r) - risk_free) / vol if vol > 0 else np.nan


def sortino_ratio(r: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """Like Sharpe, but penalises only downside deviation below the target."""
    r = pd.Series(r).dropna()
    target_daily = (1.0 + risk_free) ** (1.0 / _ANN) - 1.0
    downside = np.minimum(r - target_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(_ANN))
    return (annualized_return(r) - risk_free) / downside_dev if downside_dev > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    """Drawdown from the running peak of cumulative wealth (<= 0)."""
    wealth = (1.0 + pd.Series(r).dropna()).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    """Deepest peak-to-trough loss (negative number)."""
    return float(drawdown_series(r).min())


def calmar_ratio(r: pd.Series) -> float:
    """Annualised return / |max drawdown|."""
    mdd = abs(max_drawdown(r))
    return annualized_return(r) / mdd if mdd > 0 else np.nan


# --------------------------------------------------------------------------
# Tail risk
# --------------------------------------------------------------------------
def historical_var(r: pd.Series, confidence: float = 0.95) -> float:
    """Empirical daily VaR: the (1-confidence) quantile of returns, negated."""
    r = pd.Series(r).dropna()
    return float(-np.percentile(r, (1.0 - confidence) * 100.0))


def historical_cvar(r: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall: mean loss on days beyond the VaR threshold."""
    r = pd.Series(r).dropna()
    cutoff = np.percentile(r, (1.0 - confidence) * 100.0)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


def parametric_var(r: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance) daily VaR: -(mu + sigma * z_alpha)."""
    r = pd.Series(r).dropna()
    z = NormalDist().inv_cdf(1.0 - confidence)  # negative quantile
    return float(-(r.mean() + r.std(ddof=1) * z))


def beta(r_asset: pd.Series, r_benchmark: pd.Series) -> float:
    """CAPM beta: cov(asset, benchmark) / var(benchmark)."""
    aligned = pd.concat([r_asset, r_benchmark], axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def rolling_sharpe(
    r: pd.Series, window: int = 63, risk_free: float = RISK_FREE_RATE
) -> pd.Series:
    """Rolling annualised Sharpe (arithmetic approximation inside the window)."""
    mean = r.rolling(window).mean() * _ANN
    vol = r.rolling(window).std(ddof=1) * np.sqrt(_ANN)
    return ((mean - risk_free) / vol).rename(f"rolling_sharpe_{window}d")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summary_table(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    risk_free: float = RISK_FREE_RATE,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Full risk/performance summary, one row per asset plus the portfolio."""
    port = portfolio_returns(returns, weights)
    bench = returns[benchmark]
    w = pd.Series(weights, dtype=float)

    rows = {}
    for name in list(returns.columns) + ["PORTFOLIO"]:
        series = port if name == "PORTFOLIO" else returns[name]
        rows[name] = {
            "weight": 1.0 if name == "PORTFOLIO" else float(w.get(name, 0.0)),
            "ann_return": annualized_return(series),
            "ann_volatility": annualized_volatility(series),
            "sharpe": sharpe_ratio(series, risk_free),
            "sortino": sortino_ratio(series, risk_free),
            "max_drawdown": max_drawdown(series),
            "calmar": calmar_ratio(series),
            "var_95_daily": historical_var(series, 0.95),
            "cvar_95_daily": historical_cvar(series, 0.95),
            "var_99_daily": historical_var(series, 0.99),
            f"beta_vs_{benchmark}": beta(series, bench),
        }
    return pd.DataFrame(rows).T
