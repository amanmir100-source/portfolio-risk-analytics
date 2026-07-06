import numpy as np
import pandas as pd
import pytest

from portfolio_risk import metrics
from portfolio_risk.data_generator import generate_market_data
from portfolio_risk.monte_carlo import simulate_portfolio
from portfolio_risk.optimization import (
    analytic_frontier,
    efficient_frontier_analysis,
    portfolio_point,
)

WEIGHTS = {"SPY": 0.4, "QQQ": 0.2, "TLT": 0.2, "GLD": 0.2}


@pytest.fixture(scope="module")
def returns():
    prices, _ = generate_market_data(n_days=500, seed=9)
    return metrics.daily_returns(metrics.to_wide(prices))


def test_monte_carlo_shapes_and_sanity(returns):
    mc = simulate_portfolio(returns, WEIGHTS, n_sims=800, horizon_days=60, seed=1)
    assert mc.terminal_values.shape == (800,)
    assert (mc.terminal_values > 0).all()
    assert 0.0 <= mc.prob_loss <= 1.0
    assert mc.var_amount < mc.cvar_amount  # expected shortfall exceeds VaR
    bands = mc.percentile_bands
    assert (bands["p5"] <= bands["p50"]).all() and (bands["p50"] <= bands["p95"]).all()


def test_monte_carlo_is_reproducible(returns):
    a = simulate_portfolio(returns, WEIGHTS, n_sims=300, horizon_days=40, seed=11)
    b = simulate_portfolio(returns, WEIGHTS, n_sims=300, horizon_days=40, seed=11)
    assert np.array_equal(a.terminal_values, b.terminal_values)


def test_monte_carlo_chunking_does_not_change_results(returns):
    a = simulate_portfolio(returns, WEIGHTS, n_sims=500, horizon_days=30, seed=2,
                           chunk_size=500)
    b = simulate_portfolio(returns, WEIGHTS, n_sims=500, horizon_days=30, seed=2,
                           chunk_size=137)
    assert np.allclose(a.terminal_values, b.terminal_values)


def test_frontier_dominates_random_cloud(returns):
    fr = efficient_frontier_analysis(returns, WEIGHTS, n_portfolios=3000, seed=4)
    # closed-form frontier: recompute sigma(target) and require it to lower-bound
    # every random long-only portfolio at the same return level
    mu = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a = ones @ inv @ ones
    b = ones @ inv @ mu
    c = mu @ inv @ mu
    d = a * c - b**2
    sigma_frontier = np.sqrt(
        (a * fr.cloud["ret"] ** 2 - 2 * b * fr.cloud["ret"] + c) / d
    )
    assert (fr.cloud["vol"].to_numpy() >= sigma_frontier.to_numpy() - 1e-9).all()
    # global minimum-variance portfolio has the lowest possible volatility
    assert fr.min_var_point[0] <= fr.cloud["vol"].min() + 1e-9


def test_frontier_min_var_matches_closed_form(returns):
    frontier = analytic_frontier(returns)
    mu = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    assert frontier["vol"].min() == pytest.approx(
        np.sqrt(1.0 / (ones @ inv @ ones)), rel=1e-9
    )


def test_max_sharpe_weights_are_long_only_and_sum_to_one(returns):
    fr = efficient_frontier_analysis(returns, WEIGHTS, n_portfolios=2000, seed=6)
    w = fr.max_sharpe_weights
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-3)


def test_portfolio_point_vol_matches_series_vol(returns):
    # w' cov w must equal the variance of the aggregated return series
    vol, _ = portfolio_point(returns, WEIGHTS)
    port_r = metrics.portfolio_returns(returns, WEIGHTS)
    assert vol == pytest.approx(port_r.std(ddof=1) * np.sqrt(252), rel=1e-9)
