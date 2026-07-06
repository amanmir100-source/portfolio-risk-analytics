from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import metrics


def test_annualized_return_constant_daily():
    # 504 days (2 years) of exactly 0.1%/day compounds to 1.001^252 - 1 per year
    r = pd.Series([0.001] * 504)
    assert metrics.annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_annualized_volatility_matches_definition():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 500))
    assert metrics.annualized_volatility(r) == pytest.approx(
        r.std(ddof=1) * np.sqrt(252)
    )


def test_sharpe_is_excess_return_over_vol():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 750))
    expected = (metrics.annualized_return(r) - 0.02) / metrics.annualized_volatility(r)
    assert metrics.sharpe_ratio(r, risk_free=0.02) == pytest.approx(expected)


def test_max_drawdown_hand_computed():
    # wealth: 1.10 -> 0.99 -> 1.045  => trough/peak - 1 = 0.99/1.10 - 1 = -10%
    r = pd.Series([0.10, -0.10, 1.045 / 0.99 - 1])
    assert metrics.max_drawdown(r) == pytest.approx(-0.10)


def test_drawdown_series_never_positive():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.02, 400))
    dd = metrics.drawdown_series(r)
    assert (dd <= 1e-12).all()


def test_historical_var_and_cvar_hand_computed():
    # 20 observations; 5th percentile interpolates between the two worst
    r = pd.Series([-0.04, -0.03, -0.02, -0.01] + [0.01] * 16)
    # np.percentile(linear): -0.04 + 0.95 * 0.01 = -0.0305
    assert metrics.historical_var(r, 0.95) == pytest.approx(0.0305)
    # only -0.04 lies at/below the cutoff -> CVaR = 4%
    assert metrics.historical_cvar(r, 0.95) == pytest.approx(0.04)


def test_parametric_var_matches_normal_quantile():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0002, 0.012, 1000))
    z = NormalDist().inv_cdf(0.05)
    expected = -(r.mean() + r.std(ddof=1) * z)
    assert metrics.parametric_var(r, 0.95) == pytest.approx(expected)
    # for a near-symmetric sample, parametric and historical should be close
    assert metrics.parametric_var(r) == pytest.approx(
        metrics.historical_var(r), rel=0.15
    )


def test_beta_of_scaled_benchmark_is_the_scale():
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0, 0.01, 600))
    assert metrics.beta(2.0 * bench, bench) == pytest.approx(2.0)


def test_portfolio_returns_is_weighted_sum():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.00, 0.04, -0.01]})
    port = metrics.portfolio_returns(returns, {"A": 0.6, "B": 0.4})
    expected = 0.6 * returns["A"] + 0.4 * returns["B"]
    assert np.allclose(port.to_numpy(), expected.to_numpy())


def test_portfolio_returns_rejects_unknown_ticker():
    returns = pd.DataFrame({"A": [0.01, 0.02]})
    with pytest.raises(KeyError):
        metrics.portfolio_returns(returns, {"A": 0.5, "MISSING": 0.5})


def test_summary_table_contains_all_assets_and_portfolio():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(300, 3)), columns=["SPY", "TLT", "GLD"]
    )
    table = metrics.summary_table(returns, {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
    assert list(table.index) == ["SPY", "TLT", "GLD", "PORTFOLIO"]
    assert table.loc["PORTFOLIO", "weight"] == 1.0
    assert table.loc["SPY", "beta_vs_SPY"] == pytest.approx(1.0)
