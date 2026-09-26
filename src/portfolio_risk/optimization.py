"""Mean-variance portfolio optimisation (Markowitz) in closed form.

Two complementary views, both pure NumPy (no optimiser dependency):

1. Analytic minimum-variance frontier (short sales allowed) from the classic
   two-fund separation algebra:
       A = 1' S^-1 1,  B = 1' S^-1 mu,  C = mu' S^-1 mu,  D = A*C - B^2
       sigma^2(r*) = (A*r*^2 - 2*B*r* + C) / D
   The global minimum-variance portfolio sits at r* = B/A with var = 1/A.

2. Exact long-only max-Sharpe portfolio. On any fixed set of assets the best
   Sharpe portfolio is the tangency portfolio, w ~ S^-1 (mu - rf). The
   long-only optimum is the tangency portfolio of some subset whose weights
   all come out positive (KKT conditions), so trying every subset and keeping
   the best valid one is exact. 8 assets means 255 subsets; for hundreds of
   assets you would use a QP solver instead.

3. A 20,000-portfolio long-only Monte Carlo cloud (Dirichlet-sampled weights)
   coloured by Sharpe ratio, for context on the chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

_ANN = TRADING_DAYS_PER_YEAR


@dataclass
class FrontierResult:
    cloud: pd.DataFrame            # columns: ret, vol, sharpe (long-only random)
    frontier: pd.DataFrame         # columns: ret, vol (analytic curve)
    min_var_point: tuple[float, float]        # (vol, ret)
    max_sharpe_point: tuple[float, float]     # (vol, ret), exact long-only optimum
    max_sharpe_weights: pd.Series
    max_sharpe: float
    current_point: tuple[float, float]        # (vol, ret) of the model portfolio
    current_sharpe: float
    sampled_max_sharpe_point: tuple[float, float] = (np.nan, np.nan)  # best of the cloud
    sampled_max_sharpe: float = np.nan


def _annualized_inputs(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = returns.mean().to_numpy() * _ANN
    cov = returns.cov().to_numpy() * _ANN
    return mu, cov


def portfolio_point(
    returns: pd.DataFrame, weights: dict[str, float] | pd.Series
) -> tuple[float, float]:
    """(annualised vol, annualised return) for a fixed-weight portfolio."""
    mu, cov = _annualized_inputs(returns)
    w = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0).to_numpy()
    return float(np.sqrt(w @ cov @ w)), float(w @ mu)


def analytic_frontier(returns: pd.DataFrame, n_points: int = 200) -> pd.DataFrame:
    """Closed-form minimum-variance frontier (shorting allowed)."""
    mu, cov = _annualized_inputs(returns)
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a = ones @ inv @ ones
    b = ones @ inv @ mu
    c = mu @ inv @ mu
    d = a * c - b**2

    r_min_var = b / a
    targets = np.linspace(r_min_var, mu.max() * 1.10, n_points)
    variances = (a * targets**2 - 2.0 * b * targets + c) / d
    return pd.DataFrame({"ret": targets, "vol": np.sqrt(variances)})


def tangency_weights(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> np.ndarray:
    """Unconstrained max-Sharpe weights (shorting allowed), summing to 1."""
    w = np.linalg.solve(cov, mu - risk_free)
    return w / w.sum()


def max_sharpe_long_only(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> np.ndarray:
    """Exact long-only max-Sharpe weights by checking every subset of assets."""
    n = len(mu)
    excess = mu - risk_free
    best_sharpe, best_w = -np.inf, None
    for mask in range(1, 2**n):
        idx = [i for i in range(n) if mask >> i & 1]
        sub = np.linalg.solve(cov[np.ix_(idx, idx)], excess[idx])
        if (sub <= 0).any():  # not a valid long-only candidate
            continue
        w = np.zeros(n)
        w[idx] = sub / sub.sum()
        sharpe = (w @ excess) / np.sqrt(w @ cov @ w)
        if sharpe > best_sharpe:
            best_sharpe, best_w = sharpe, w
    if best_w is None:
        raise ValueError("no long-only portfolio beats the risk-free rate")
    return best_w


def random_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 20_000,
    seed: int = 11,
    risk_free: float = RISK_FREE_RATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Long-only random portfolios; returns (stats DataFrame, weight matrix)."""
    mu, cov = _annualized_inputs(returns)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(mu)), size=n_portfolios)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    cloud = pd.DataFrame(
        {"ret": rets, "vol": vols, "sharpe": (rets - risk_free) / vols}
    )
    return cloud, w


def efficient_frontier_analysis(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    n_portfolios: int = 20_000,
    seed: int = 11,
    risk_free: float = RISK_FREE_RATE,
) -> FrontierResult:
    """Everything the frontier chart and the CLI report need, in one call."""
    mu, cov = _annualized_inputs(returns)
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a = ones @ inv @ ones
    b = ones @ inv @ mu

    cloud, _ = random_portfolios(returns, n_portfolios, seed, risk_free)
    best = int(cloud["sharpe"].idxmax())
    w_exact = max_sharpe_long_only(mu, cov, risk_free)
    exact_vol, exact_ret = float(np.sqrt(w_exact @ cov @ w_exact)), float(w_exact @ mu)
    max_sharpe_weights = pd.Series(w_exact, index=returns.columns).round(4)

    cur_vol, cur_ret = portfolio_point(returns, weights)
    return FrontierResult(
        cloud=cloud,
        frontier=analytic_frontier(returns),
        min_var_point=(float(np.sqrt(1.0 / a)), float(b / a)),
        max_sharpe_point=(exact_vol, exact_ret),
        max_sharpe_weights=max_sharpe_weights,
        max_sharpe=(exact_ret - risk_free) / exact_vol,
        current_point=(cur_vol, cur_ret),
        current_sharpe=float((cur_ret - risk_free) / cur_vol),
        sampled_max_sharpe_point=(
            float(cloud.loc[best, "vol"]),
            float(cloud.loc[best, "ret"]),
        ),
        sampled_max_sharpe=float(cloud.loc[best, "sharpe"]),
    )
