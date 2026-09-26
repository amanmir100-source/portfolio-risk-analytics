import numpy as np
import pandas as pd
import pytest

from portfolio_risk import metrics, risk_contribution
from portfolio_risk.data_generator import generate_market_data

WEIGHTS = {"SPY": 0.4, "QQQ": 0.2, "TLT": 0.2, "GLD": 0.2}


@pytest.fixture(scope="module")
def returns():
    prices, _ = generate_market_data(n_days=500, seed=9)
    return metrics.daily_returns(metrics.to_wide(prices))


def test_euler_matches_hand_calculation():
    # independent assets: contribution_i = w_i^2 var_i / sigma
    w, cov = np.array([0.5, 0.5]), np.diag([0.04, 0.01])
    sigma = np.sqrt(0.25 * 0.04 + 0.25 * 0.01)
    assert risk_contribution.euler_contributions(w, cov) == pytest.approx(
        [0.25 * 0.04 / sigma, 0.25 * 0.01 / sigma])


def test_contributions_add_up_to_portfolio_vol_and_cvar(returns):
    rc = risk_contribution.risk_contributions(returns, WEIGHTS)
    port = metrics.portfolio_returns(returns, WEIGHTS)
    total = rc.loc["TOTAL"]
    assert total["vol_contribution"] == pytest.approx(metrics.annualized_volatility(port))
    assert total["cvar_contribution"] == pytest.approx(metrics.historical_cvar(port, 0.95))
    assert total["vol_share"] == pytest.approx(1.0)
    assert total["cvar_share"] == pytest.approx(1.0)


def test_unheld_assets_contribute_nothing(returns):
    rc = risk_contribution.risk_contributions(returns, WEIGHTS)
    unheld = [t for t in returns.columns if t not in WEIGHTS]
    assert unheld and (rc.loc[unheld, ["vol_contribution", "cvar_contribution"]] == 0).all().all()


def test_cvar_contribution_is_average_loss_on_worst_days():
    r = pd.DataFrame({"A": [-0.10, 0.01, 0.02, 0.03], "B": [0.02, 0.01, -0.01, 0.00]})
    rc = risk_contribution.risk_contributions(r, {"A": 0.5, "B": 0.5}, confidence=0.75)
    # portfolio returns: -0.04, 0.01, 0.005, 0.015 -> worst 25% is day 0 only
    assert rc.loc["A", "cvar_contribution"] == pytest.approx(0.05)
    assert rc.loc["B", "cvar_contribution"] == pytest.approx(-0.01)  # B helped that day
