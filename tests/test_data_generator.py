import numpy as np
import pandas as pd

from portfolio_risk import config
from portfolio_risk.data_generator import (
    generate_market_data,
    nearest_correlation_matrix,
)


def test_generation_is_reproducible():
    a, _ = generate_market_data(n_days=300, seed=7)
    b, _ = generate_market_data(n_days=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_shapes_prices_and_volumes_are_valid():
    prices, _ = generate_market_data(n_days=300, seed=1)
    assert set(prices["ticker"]) == set(config.TICKERS)
    assert len(prices) == 300 * len(config.TICKERS)
    assert (prices["close"] > 0).all()
    assert (prices["volume"] >= 0).all()
    assert not prices[["date", "ticker"]].duplicated().any()


def test_correlation_matrices_are_positive_definite():
    for matrix in (config.CORR_CALM, config.CORR_STRESS):
        fixed = nearest_correlation_matrix(np.array(matrix))
        assert np.all(np.linalg.eigvalsh(fixed) > 0)
        assert np.allclose(np.diag(fixed), 1.0)
        assert np.allclose(fixed, fixed.T)


def test_stress_regime_is_more_volatile():
    prices, regimes = generate_market_data(n_days=1260, seed=3)
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = wide.pct_change().iloc[1:]
    regime_flags = regimes.to_numpy()[1:]
    assert regime_flags.sum() >= 30, "need enough stress days for a stable estimate"
    spy = returns["SPY"].to_numpy()
    assert spy[regime_flags == 1].std() > 1.5 * spy[regime_flags == 0].std()
