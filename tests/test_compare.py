import pandas as pd
import pytest

from portfolio_risk import compare, config, metrics
from portfolio_risk.data_generator import generate_market_data


@pytest.fixture(scope="module")
def returns():
    prices, _ = generate_market_data(n_days=400, seed=9)
    return metrics.daily_returns(metrics.to_wide(prices))


def test_comparison_row_matches_the_single_portfolio_metrics(returns):
    table = compare.compare_portfolios(returns, {"current": config.PORTFOLIO_WEIGHTS}, 0.03)
    row = table.loc["current"]
    port = metrics.portfolio_returns(returns, config.PORTFOLIO_WEIGHTS)
    assert row["sharpe"] == pytest.approx(metrics.sharpe_ratio(port, 0.03))
    assert row["ann_volatility"] == pytest.approx(metrics.annualized_volatility(port))
    assert row["equity_weight"] == pytest.approx(0.55)  # SPY + QQQ + IWM + EFA
    assert 0 < row["equity_vol_share"] <= 1.5
    assert row["largest_risk_source"] in config.TICKERS


def test_all_bond_portfolio_has_no_equity_risk(returns):
    table = compare.compare_portfolios(
        returns, {"bonds": {"TLT": 0.5, "LQD": 0.5}, "stocks": {"SPY": 1.0}})
    assert table.loc["bonds", "equity_vol_share"] == pytest.approx(0.0)
    assert table.loc["stocks", "equity_vol_share"] == pytest.approx(1.0)
    assert table.loc["stocks", "largest_risk_source"] == "SPY"


def test_stress_columns_appear_when_history_is_given():
    history, _ = generate_market_data(n_days=5200, seed=4)
    wide = metrics.to_wide(history)
    returns = metrics.daily_returns(wide.iloc[-300:])
    table = compare.compare_portfolios(returns, {"current": config.PORTFOLIO_WEIGHTS},
                                       history=wide)
    stress_cols = [c for c in table.columns if c.startswith("stress: ")]
    assert len(stress_cols) == 3
    assert pd.api.types.is_float_dtype(table[stress_cols[0]])
