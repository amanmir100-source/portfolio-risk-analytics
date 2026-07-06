"""Monte Carlo simulation of the portfolio's 1-year forward distribution.

Approach: estimate the daily mean vector and covariance matrix from history,
then simulate correlated multivariate-normal daily returns (Cholesky) for all
assets, aggregate to fixed-weight portfolio returns, and compound into wealth
paths. Simulation runs in chunks to keep memory flat at 10k+ paths.

Known simplification (worth stating in an interview): normal shocks understate
fat tails, so simulated VaR is a floor rather than a ceiling. Historical and
parametric VaR in metrics.py give complementary views.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics

PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class MonteCarloResult:
    initial_value: float
    horizon_days: int
    n_sims: int
    percentile_bands: pd.DataFrame          # day x {p5..p95} wealth percentiles
    sample_paths: np.ndarray                # subset of paths for plotting
    terminal_values: np.ndarray             # (n_sims,)
    var_amount: float                       # 1-horizon 95% VaR in currency
    cvar_amount: float
    prob_loss: float
    summary: dict = field(default_factory=dict)


def simulate_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    n_sims: int = 10_000,
    horizon_days: int = 252,
    initial_value: float = 100_000.0,
    seed: int = 7,
    n_sample_paths: int = 100,
    chunk_size: int = 2_500,
) -> MonteCarloResult:
    """Simulate n_sims wealth paths over horizon_days trading days."""
    w = pd.Series(weights, dtype=float)
    w = (w / w.sum()).reindex(returns.columns).fillna(0.0)

    mu = returns.mean().to_numpy()          # daily arithmetic means
    cov = returns.cov().to_numpy()          # daily covariance
    chol = np.linalg.cholesky(cov)
    w_vec = w.to_numpy()

    rng = np.random.default_rng(seed)
    wealth_paths = np.empty((n_sims, horizon_days))
    done = 0
    while done < n_sims:
        size = min(chunk_size, n_sims - done)
        z = rng.standard_normal((size, horizon_days, len(mu)))
        asset_returns = z @ chol.T + mu          # correlated daily returns
        port_returns = asset_returns @ w_vec     # fixed-weight aggregation
        wealth_paths[done : done + size] = initial_value * np.cumprod(
            1.0 + port_returns, axis=1
        )
        done += size

    terminal = wealth_paths[:, -1]
    bands = pd.DataFrame(
        {f"p{p}": np.percentile(wealth_paths, p, axis=0) for p in PERCENTILES},
        index=pd.RangeIndex(1, horizon_days + 1, name="day"),
    )
    # prepend day 0 at the initial value so plots start from a single point
    bands.loc[0] = initial_value
    bands = bands.sort_index()

    p5 = np.percentile(terminal, 5)
    var_amount = initial_value - p5
    cvar_amount = initial_value - terminal[terminal <= p5].mean()
    prob_loss = float(np.mean(terminal < initial_value))

    result = MonteCarloResult(
        initial_value=initial_value,
        horizon_days=horizon_days,
        n_sims=n_sims,
        percentile_bands=bands,
        sample_paths=wealth_paths[
            rng.choice(n_sims, size=min(n_sample_paths, n_sims), replace=False)
        ],
        terminal_values=terminal,
        var_amount=float(var_amount),
        cvar_amount=float(cvar_amount),
        prob_loss=prob_loss,
    )
    result.summary = {
        "median_terminal": float(np.median(terminal)),
        "mean_terminal": float(terminal.mean()),
        "p5_terminal": float(p5),
        "p95_terminal": float(np.percentile(terminal, 95)),
        "var_95_pct": float(var_amount / initial_value),
        "cvar_95_pct": float(cvar_amount / initial_value),
        "prob_loss": prob_loss,
        "hist_daily_var_95": metrics.historical_var(
            metrics.portfolio_returns(returns, w.to_dict())
        ),
    }
    return result
