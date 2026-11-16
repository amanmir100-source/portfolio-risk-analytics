"""Compare candidate portfolios side by side on the same history.

The risk memo ends with "re-run the analysis on two or three candidate
weightings before changing anything". This is that step: one row per
portfolio with return, volatility, Sharpe, tail risk, how concentrated the
risk is, and (when long history is available) each historical stress replay.
"""
from __future__ import annotations

import pandas as pd

from . import metrics, stress
from .config import ASSETS, RISK_FREE_RATE
from .risk_contribution import risk_contributions


def equity_tickers() -> list[str]:
    return [t for t, spec in ASSETS.items() if "Equity" in spec.asset_class]


def compare_portfolios(
    returns: pd.DataFrame,
    portfolios: dict[str, dict[str, float]],
    risk_free: float = RISK_FREE_RATE,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per named portfolio. `history` is wide prices for stress replays."""
    equities = equity_tickers()
    rows = {}
    for name, weights in portfolios.items():
        r = metrics.portfolio_returns(returns, weights)
        shares = risk_contributions(returns, weights).drop(index="TOTAL")
        top = shares["vol_share"].idxmax()
        row = {
            "ann_return": metrics.annualized_return(r),
            "ann_volatility": metrics.annualized_volatility(r),
            "sharpe": metrics.sharpe_ratio(r, risk_free),
            "max_drawdown": metrics.max_drawdown(r),
            "var_95_daily": metrics.historical_var(r, 0.95),
            "cvar_95_daily": metrics.historical_cvar(r, 0.95),
            "equity_weight": float(sum(weights.get(t, 0.0) for t in equities)),
            "equity_vol_share": float(shares.loc[shares.index.isin(equities), "vol_share"].sum()),
            "largest_risk_source": top,
            "largest_vol_share": float(shares.loc[top, "vol_share"]),
        }
        if history is not None:
            replay = stress.run_scenarios(history, weights)["portfolio_return"]
            row.update({f"stress: {scenario}": float(v) for scenario, v in replay.items()})
        rows[name] = row
    out = pd.DataFrame(rows).T
    out.index.name = "portfolio"
    numeric = [c for c in out.columns if c != "largest_risk_source"]
    out[numeric] = out[numeric].astype(float)
    return out
