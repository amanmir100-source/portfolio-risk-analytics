"""Monte Carlo simulation of the portfolio's 1-year forward distribution.

Three ways to draw each simulated day:

* "normal": correlated multivariate-normal returns (Cholesky) from the
  historical means and covariance. Understates fat tails.
* "student_t": the same means and covariance, but with multivariate
  Student-t shocks. One shared chi-squared draw per day scales every asset,
  so bad days hit all of them together. Degrees of freedom come from the
  portfolio's excess kurtosis (nu = 4 + 6 / kurtosis), and the shocks are
  rescaled so volatility is unchanged.
* "bootstrap": real historical days resampled in blocks of consecutive days,
  which keeps fat tails, volatility clustering and crisis correlations as
  they happened.

Daily returns are aggregated to fixed-weight portfolio returns and
compounded into wealth paths, in chunks to keep memory flat at 10k+ paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics

PERCENTILES = (5, 25, 50, 75, 95)
METHODS = ("normal", "student_t", "bootstrap")
DEFAULT_BLOCK = 21  # about one trading month


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
    method: str = "normal"


def student_t_dof(port_returns: pd.Series) -> float:
    """Degrees of freedom whose excess kurtosis (6 / (nu - 4)) matches the data."""
    kurt = float(pd.Series(port_returns).dropna().kurt())
    if kurt <= 0.06:  # thin or normal tails: effectively normal
        return 104.0
    return 4.0 + 6.0 / kurt


def simulate_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    n_sims: int = 10_000,
    horizon_days: int = 252,
    initial_value: float = 100_000.0,
    seed: int = 7,
    n_sample_paths: int = 100,
    chunk_size: int = 2_500,
    method: str = "normal",
    block_size: int = DEFAULT_BLOCK,
) -> MonteCarloResult:
    """Simulate n_sims wealth paths over horizon_days trading days."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    w = pd.Series(weights, dtype=float)
    w = (w / w.sum()).reindex(returns.columns).fillna(0.0)

    mu = returns.mean().to_numpy()          # daily arithmetic means
    cov = returns.cov().to_numpy()          # daily covariance
    chol = np.linalg.cholesky(cov)
    w_vec = w.to_numpy()
    history = returns.to_numpy()
    dof = student_t_dof(returns @ w_vec) if method == "student_t" else None

    rng = np.random.default_rng(seed)
    # separate stream for the Student-t scaling draws, so chunk size never
    # changes which random numbers a path gets
    rng_scale = np.random.default_rng([seed, 1])
    wealth_paths = np.empty((n_sims, horizon_days))
    done = 0
    while done < n_sims:
        size = min(chunk_size, n_sims - done)
        if method == "bootstrap":
            asset_returns = bootstrap_days(history, size, horizon_days, block_size, rng)
        else:
            z = rng.standard_normal((size, horizon_days, len(mu)))
            if method == "student_t":
                z = z * t_scale(rng_scale, dof, (size, horizon_days, 1))
            asset_returns = z @ chol.T + mu      # correlated daily returns
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
        method=method,
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
    if dof is not None:
        result.summary["student_t_dof"] = dof
    return result


def t_scale(rng: np.random.Generator, dof: float, shape: tuple) -> np.ndarray:
    """Multiplier turning standard normals into unit-variance Student-t draws."""
    chi2 = rng.chisquare(dof, shape)
    return np.sqrt((dof - 2.0) / chi2)  # = 1/sqrt(chi2/dof) * sqrt((dof-2)/dof)


def bootstrap_days(
    history: np.ndarray, size: int, horizon_days: int, block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(size, horizon_days, n_assets) returns built from blocks of real days."""
    n_hist = len(history)
    block_size = min(block_size, n_hist)
    n_blocks = math.ceil(horizon_days / block_size)
    starts = rng.integers(0, n_hist - block_size + 1, size=(size, n_blocks))
    idx = (starts[..., None] + np.arange(block_size)).reshape(size, -1)[:, :horizon_days]
    return history[idx]
