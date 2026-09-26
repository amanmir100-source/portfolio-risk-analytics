import numpy as np
import pandas as pd
import pytest

from portfolio_risk import metrics, monte_carlo
from portfolio_risk.data_generator import generate_market_data
from portfolio_risk.monte_carlo import simulate_portfolio

WEIGHTS = {"SPY": 0.4, "QQQ": 0.2, "TLT": 0.2, "GLD": 0.2}


@pytest.fixture(scope="module")
def returns():
    prices, _ = generate_market_data(n_days=500, seed=9)
    return metrics.daily_returns(metrics.to_wide(prices))


def test_normal_is_still_the_default(returns):
    a = simulate_portfolio(returns, WEIGHTS, n_sims=300, horizon_days=40, seed=5)
    b = simulate_portfolio(returns, WEIGHTS, n_sims=300, horizon_days=40, seed=5,
                           method="normal")
    assert np.array_equal(a.terminal_values, b.terminal_values)


@pytest.mark.parametrize("method", ["student_t", "bootstrap"])
def test_new_methods_are_reproducible_and_chunk_safe(returns, method):
    a = simulate_portfolio(returns, WEIGHTS, n_sims=400, horizon_days=30, seed=3,
                           method=method, chunk_size=400)
    b = simulate_portfolio(returns, WEIGHTS, n_sims=400, horizon_days=30, seed=3,
                           method=method, chunk_size=97)
    assert np.allclose(a.terminal_values, b.terminal_values)
    assert a.method == method


def test_dof_matches_kurtosis():
    # excess kurtosis of a Student-t is 6 / (nu - 4), so kurtosis 6 -> nu = 5
    r = pd.Series(np.random.default_rng(0).standard_t(5, 400_000))
    assert monte_carlo.student_t_dof(r) == pytest.approx(4 + 6 / r.kurt())
    assert monte_carlo.student_t_dof(pd.Series(np.linspace(-1, 1, 1000))) == 104.0


def test_t_shocks_keep_unit_variance_but_fatter_tails():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(1_000_000) * monte_carlo.t_scale(
        np.random.default_rng(2), 5.0, (1_000_000,))
    assert z.std() == pytest.approx(1.0, rel=0.02)
    assert pd.Series(z).kurt() > 2  # a normal would be ~0


def test_bootstrap_uses_only_real_consecutive_days():
    history = np.arange(100, dtype=float).reshape(50, 2)  # row i = [2i, 2i+1]
    out = monte_carlo.bootstrap_days(history, size=20, horizon_days=30, block_size=7,
                                     rng=np.random.default_rng(4))
    assert out.shape == (20, 30, 2)
    rows = out[..., 0] / 2  # which historical day each simulated day came from
    assert np.isin(rows, np.arange(50)).all()
    # inside a block, days follow each other in order
    first_block = rows[:, :7]
    assert (np.diff(first_block, axis=1) == 1).all()


def test_unknown_method_raises(returns):
    with pytest.raises(ValueError, match="unknown method"):
        simulate_portfolio(returns, WEIGHTS, n_sims=10, horizon_days=5, method="cauchy")
