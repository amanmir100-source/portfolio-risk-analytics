"""Historical stress scenarios: past crises replayed on today's portfolio.

Each scenario takes every ETF's actual return from the S&P 500's peak close
to its trough close in that crisis. Today's weights are held from the start
of the window with no rebalancing, so an asset's contribution is just
weight x return and the contributions add up to the portfolio's loss.

Needs price history back to 2007, so it only runs with --live.
"""
from __future__ import annotations

import pandas as pd

# S&P 500 peak close -> trough close
SCENARIOS: dict[str, tuple[str, str]] = {
    "2008 financial crisis": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}
HISTORY_START = "2007-01-01"


def run_scenarios(
    wide_prices: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    scenarios: dict[str, tuple[str, str]] = SCENARIOS,
) -> pd.DataFrame:
    """One row per scenario: dates used, portfolio return, each asset's return."""
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    prices = wide_prices.sort_index()[list(w.index)]
    rows = []
    for name, (start, end) in scenarios.items():
        # first/last dates inside the window where every asset has a close
        window = prices.loc[start:end].dropna(how="any")
        if len(window) < 2:
            raise ValueError(f"not enough price history for {name!r} ({start} to {end})")
        asset_returns = window.iloc[-1] / window.iloc[0] - 1.0
        rows.append({
            "scenario": name,
            "start": window.index[0].strftime("%Y-%m-%d"),
            "end": window.index[-1].strftime("%Y-%m-%d"),
            "portfolio_return": float((w * asset_returns).sum()),
            **asset_returns.to_dict(),
        })
    return pd.DataFrame(rows).set_index("scenario")
