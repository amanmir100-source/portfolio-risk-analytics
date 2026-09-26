"""Which positions drive the portfolio's risk?

Two ways to split total risk into per-asset pieces that add up exactly:

* Volatility (Euler): asset i contributes w_i * (Cov w)_i / sigma_p. The
  pieces sum to the portfolio's volatility. It's the everyday view and
  assumes risk is captured by the covariance matrix.
* Tail (CVaR): on the portfolio's worst (1 - confidence) share of days, the
  average loss each asset caused, w_i * r_i. The pieces sum to the
  portfolio's historical CVaR, and they use the real fat-tailed history.

Where the two disagree is the interesting part: an asset can look calm day
to day but hurt most on the worst days, or the other way round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def euler_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-asset volatility contributions; they sum to sqrt(w' cov w)."""
    sigma = np.sqrt(weights @ cov @ weights)
    return weights * (cov @ weights) / sigma


def risk_contributions(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """One row per asset plus TOTAL: weight, volatility and CVaR contributions."""
    w = pd.Series(weights, dtype=float)
    w = (w / w.sum()).reindex(returns.columns).fillna(0.0)
    cov = returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    vol = euler_contributions(w.to_numpy(), cov)

    port = returns @ w
    cutoff = np.percentile(port, (1.0 - confidence) * 100.0)  # same as metrics.historical_cvar
    tail = port <= cutoff
    cvar = -(returns[tail] * w).mean()

    out = pd.DataFrame({
        "weight": w,
        "vol_contribution": vol,
        "vol_share": vol / vol.sum(),
        "cvar_contribution": cvar,
        "cvar_share": cvar / cvar.sum(),
    })
    out.loc["TOTAL"] = out.sum()
    return out
