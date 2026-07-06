"""Synthetic market data generator (NumPy).

Produces 5 years of realistic daily ETF prices using a regime-switching,
correlated geometric Brownian motion:

* A two-state Markov chain flips the market between "calm" and "stress".
* Each regime has its own drift, volatility and correlation matrix
  (in stress, risk-asset correlations tighten and treasuries rally --
  the classic flight-to-quality pattern).
* Correlated shocks are produced with a Cholesky factorisation of the
  regime's correlation matrix.
* Daily close: S_t = S_{t-1} * exp((mu - 0.5*sigma^2) * dt + sigma * sqrt(dt) * z_t)

Everything is seeded, so the demo dataset is fully reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def nearest_correlation_matrix(corr: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the set of valid correlation matrices.

    Hand-specified correlation matrices are occasionally not positive
    semi-definite. This clips negative eigenvalues and rescales the diagonal
    back to exactly 1 so that the Cholesky factorisation is always defined.
    """
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return (fixed + fixed.T) / 2.0


def simulate_regimes(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the calm(0)/stress(1) Markov chain."""
    regimes = np.zeros(n_days, dtype=int)
    state = 0
    for t in range(1, n_days):
        u = rng.random()
        if state == 0 and u < config.P_CALM_TO_STRESS:
            state = 1
        elif state == 1 and u < config.P_STRESS_TO_CALM:
            state = 0
        regimes[t] = state
    return regimes


DEFAULT_SEED = 40  # produces a realistic 5y sample: growth + one -25% correction


def generate_market_data(
    n_days: int = 1260,
    seed: int = DEFAULT_SEED,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate the synthetic dataset.

    Returns
    -------
    prices : long-format DataFrame with columns [date, ticker, close, volume]
    regimes : Series of 0 (calm) / 1 (stress) indexed by date, useful for
        validating that the generator behaves as designed.
    """
    rng = np.random.default_rng(seed)
    tickers = config.TICKERS
    n_assets = len(tickers)
    dt = 1.0 / config.TRADING_DAYS_PER_YEAR

    if end_date is None:
        end_date = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(end=end_date, periods=n_days)

    specs = [config.ASSETS[t] for t in tickers]
    drift = np.array([s.annual_drift for s in specs])
    vol = np.array([s.annual_vol for s in specs])
    crisis_beta = np.array([s.crisis_beta for s in specs])

    # Regime-dependent parameters
    drift_by_regime = np.vstack([drift, drift - config.CRISIS_DRIFT_SHOCK * crisis_beta])
    vol_by_regime = np.vstack([vol, vol * (1.0 + config.CRISIS_VOL_FACTOR * np.abs(crisis_beta))])
    chol_by_regime = [
        np.linalg.cholesky(nearest_correlation_matrix(np.array(config.CORR_CALM))),
        np.linalg.cholesky(nearest_correlation_matrix(np.array(config.CORR_STRESS))),
    ]

    regimes = simulate_regimes(n_days, rng)

    # Correlated daily log-returns
    z = rng.standard_normal((n_days, n_assets))
    log_returns = np.empty((n_days, n_assets))
    for regime in (0, 1):
        mask = regimes == regime
        shocks = z[mask] @ chol_by_regime[regime].T
        mu, sigma = drift_by_regime[regime], vol_by_regime[regime]
        log_returns[mask] = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    start_prices = np.array([s.start_price for s in specs])
    closes = start_prices * np.exp(np.cumsum(log_returns, axis=0))

    # Volume: lognormal noise around each asset's base, elevated in stress
    base_volume = np.array([s.base_volume for s in specs])
    volume_noise = np.exp(rng.normal(0.0, 0.35, size=(n_days, n_assets)))
    stress_boost = np.where(regimes == 1, 1.8, 1.0)[:, None]
    volumes = (base_volume * volume_noise * stress_boost).astype(np.int64)

    prices = pd.DataFrame(closes, index=dates, columns=tickers).round(2)
    prices.index.name = "date"
    long = prices.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    long["volume"] = pd.DataFrame(volumes, index=dates, columns=tickers).stack().to_numpy()
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")

    regime_series = pd.Series(regimes, index=dates, name="regime")
    return long, regime_series


def assets_frame() -> pd.DataFrame:
    """Asset reference data for the `assets` SQL table."""
    return pd.DataFrame(
        [(t, s.name, s.asset_class) for t, s in config.ASSETS.items()],
        columns=["ticker", "name", "asset_class"],
    )


def weights_frame() -> pd.DataFrame:
    """Portfolio weights for the `portfolio_weights` SQL table."""
    return pd.DataFrame(
        list(config.PORTFOLIO_WEIGHTS.items()), columns=["ticker", "weight"]
    )


def generate_and_save(
    data_dir: str | Path, n_days: int = 1260, seed: int = DEFAULT_SEED
) -> Path:
    """Generate the demo dataset and write it to data/demo_prices.csv."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    prices, _ = generate_market_data(n_days=n_days, seed=seed)
    out = data_dir / "demo_prices.csv"
    prices.to_csv(out, index=False)
    return out
