"""Central configuration: asset universe, portfolio weights, market constants.

The universe is a realistic 8-asset multi-asset-class portfolio (equities,
duration, credit, gold, real estate) so that diversification, flight-to-quality
and crisis-correlation effects all show up in the analytics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSpec:
    """Static description of one asset in the universe.

    annual_drift / annual_vol parameterise the calm-regime price process used
    by the synthetic data generator. crisis_beta controls how hard the asset
    is hit in the stress regime (negative = tends to rally in a crisis).
    """

    name: str
    asset_class: str
    annual_drift: float
    annual_vol: float
    crisis_beta: float
    start_price: float
    base_volume: float  # average daily shares traded


# Order matters: correlation matrices below use this exact ordering.
ASSETS: dict[str, AssetSpec] = {
    "SPY": AssetSpec("SPDR S&P 500 ETF", "US Large-Cap Equity", 0.090, 0.16, 1.00, 420.0, 80e6),
    "QQQ": AssetSpec("Invesco Nasdaq-100 ETF", "US Tech Equity", 0.120, 0.22, 1.20, 350.0, 50e6),
    "IWM": AssetSpec("iShares Russell 2000 ETF", "US Small-Cap Equity", 0.080, 0.20, 1.15, 180.0, 30e6),
    "EFA": AssetSpec("iShares MSCI EAFE ETF", "Intl Developed Equity", 0.065, 0.17, 1.00, 70.0, 18e6),
    "TLT": AssetSpec("iShares 20+ Yr Treasury Bond ETF", "Long-Duration Treasuries", 0.030, 0.15, -0.30, 100.0, 20e6),
    "LQD": AssetSpec("iShares IG Corporate Bond ETF", "Investment-Grade Credit", 0.045, 0.09, 0.35, 105.0, 15e6),
    "GLD": AssetSpec("SPDR Gold Shares", "Commodities (Gold)", 0.060, 0.15, -0.10, 180.0, 8e6),
    "VNQ": AssetSpec("Vanguard Real Estate ETF", "US REITs", 0.070, 0.19, 1.05, 85.0, 5e6),
}

TICKERS: list[str] = list(ASSETS.keys())

# Model "moderate growth" allocation analysed throughout the project.
PORTFOLIO_WEIGHTS: dict[str, float] = {
    "SPY": 0.25,
    "QQQ": 0.15,
    "IWM": 0.05,
    "EFA": 0.10,
    "TLT": 0.15,
    "LQD": 0.10,
    "GLD": 0.10,
    "VNQ": 0.10,
}
assert abs(sum(PORTFOLIO_WEIGHTS.values()) - 1.0) < 1e-12, "weights must sum to 1"

BENCHMARK_TICKER = "SPY"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02  # annualised, used for Sharpe / Sortino / frontier

# --- Regime-switching parameters for the synthetic generator ----------------
# Two-state Markov chain: calm <-> stress.
P_CALM_TO_STRESS = 0.012   # ~ stationary stress share of ~11%
P_STRESS_TO_CALM = 0.100   # ~ average stress spell of ~10 trading days

# In stress: drift shifted down by CRISIS_DRIFT_SHOCK * crisis_beta (annualised)
# and volatility scaled by (1 + CRISIS_VOL_FACTOR * |crisis_beta|).
CRISIS_DRIFT_SHOCK = 0.30
CRISIS_VOL_FACTOR = 1.10

# Calm-regime correlation structure (symmetric, unit diagonal).
# Rows/cols follow TICKERS order: SPY QQQ IWM EFA TLT LQD GLD VNQ
CORR_CALM = [
    [1.00, 0.85, 0.80, 0.75, -0.30, 0.30, 0.05, 0.60],
    [0.85, 1.00, 0.70, 0.65, -0.25, 0.25, 0.00, 0.50],
    [0.80, 0.70, 1.00, 0.60, -0.30, 0.25, 0.05, 0.60],
    [0.75, 0.65, 0.60, 1.00, -0.20, 0.30, 0.10, 0.50],
    [-0.30, -0.25, -0.30, -0.20, 1.00, 0.50, 0.30, -0.10],
    [0.30, 0.25, 0.25, 0.30, 0.50, 1.00, 0.15, 0.30],
    [0.05, 0.00, 0.05, 0.10, 0.30, 0.15, 1.00, 0.10],
    [0.60, 0.50, 0.60, 0.50, -0.10, 0.30, 0.10, 1.00],
]

# Stress-regime correlations: risk assets converge ("correlations go to one"),
# treasuries decouple further (flight to quality), gold turns mildly defensive.
CORR_STRESS = [
    [1.00, 0.92, 0.90, 0.88, -0.45, 0.40, -0.05, 0.75],
    [0.92, 1.00, 0.85, 0.80, -0.40, 0.35, -0.10, 0.65],
    [0.90, 0.85, 1.00, 0.78, -0.45, 0.35, -0.05, 0.75],
    [0.88, 0.80, 0.78, 1.00, -0.35, 0.40, 0.00, 0.65],
    [-0.45, -0.40, -0.45, -0.35, 1.00, 0.55, 0.35, -0.20],
    [0.40, 0.35, 0.35, 0.40, 0.55, 1.00, 0.10, 0.40],
    [-0.05, -0.10, -0.05, 0.00, 0.35, 0.10, 1.00, 0.00],
    [0.75, 0.65, 0.75, 0.65, -0.20, 0.40, 0.00, 1.00],
]
