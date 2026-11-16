import math

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import backtest, metrics


def _returns(n: int = 600, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.standard_t(4, n) * 0.01, index=dates)


def test_kupiec_is_zero_when_breach_rate_matches_target():
    # 50 breaches in 1,000 days is exactly the 5% a 95% VaR promises
    lr, p_value = backtest.kupiec_pof(1000, 50, 0.95)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0)


def test_kupiec_matches_hand_computed_value():
    # 10 breaches in 250 days at 99%: expected 2.5.
    # LR = -2[240 ln .99 + 10 ln .01 - (240 ln .96 + 10 ln .04)] = 12.9555
    lr, p_value = backtest.kupiec_pof(250, 10, 0.99)
    assert lr == pytest.approx(12.9555, abs=1e-4)
    assert p_value < 0.001  # far too many breaches: the model is rejected


def test_kupiec_p_value_uses_chi_squared_one_dof():
    # 3.8415 is the 95th percentile of chi-squared(1), so its p-value is 0.05
    assert math.erfc(math.sqrt(3.841459 / 2)) == pytest.approx(0.05, abs=1e-6)
    # zero breaches is allowed and gives a finite statistic
    lr, _ = backtest.kupiec_pof(500, 0, 0.99)
    assert lr == pytest.approx(-2 * 500 * math.log(0.99))


@pytest.mark.parametrize("method, fn", [
    ("historical", metrics.historical_var),
    ("parametric", metrics.parametric_var),
])
def test_rolling_forecast_matches_metrics_on_trailing_window(method, fn):
    r = _returns()
    window = 250
    forecasts = backtest.rolling_var_forecasts(r, window, 0.99, method)
    for day in (0, 137, len(forecasts) - 1):
        position = window + day
        trailing = r.iloc[position - window:position]  # the days before `day`
        assert forecasts.iloc[day] == pytest.approx(fn(trailing, 0.99), rel=1e-9)


def test_forecasts_never_use_future_returns():
    r = _returns()
    base = backtest.rolling_var_forecasts(r, 250, 0.95, "historical")
    shocked = r.copy()
    shocked.iloc[400:] = -0.5  # a catastrophe after day 400
    after = backtest.rolling_var_forecasts(shocked, 250, 0.95, "historical")
    # the forecast for day 400 itself only uses days 150-399, so it is unchanged
    unchanged = base.index <= r.index[400]
    pd.testing.assert_series_equal(base[unchanged], after[unchanged])


def test_backtest_counts_breaches_against_the_forecast():
    r = _returns()
    result = backtest.backtest_var(r, window=250)
    assert len(result.table) == 4  # 2 methods x 2 confidence levels
    row = result.table.query("method == 'historical' and confidence == 0.99").iloc[0]
    var = result.forecasts["var_historical_99"]
    assert row["actual_breaches"] == int((result.returns < -var).sum())
    assert row["test_days"] == len(r) - 250
    assert set(result.table["result"]) <= {"pass", "reject"}


def _markov_lr(n00: int, n01: int, n10: int, n11: int) -> float:
    """Christoffersen's statistic written out longhand, as a check."""
    def ll(stay: int, move: int, p: float) -> float:
        return (stay * math.log(1 - p) if stay else 0.0) + (move * math.log(p) if move else 0.0)
    pooled = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = ll(n00 + n10, n01 + n11, pooled)
    free = ll(n00, n01, n01 / (n00 + n01)) + ll(n10, n11, n11 / (n10 + n11))
    return -2 * (restricted - free)


def test_christoffersen_matches_hand_counted_transitions():
    # transitions: 00 x4, 01 x2, 10 x2, 11 x1
    breaches = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    lr, p_value = backtest.christoffersen_independence(breaches)
    assert lr == pytest.approx(_markov_lr(4, 2, 2, 1))
    assert p_value == pytest.approx(math.erfc(math.sqrt(lr / 2)))


def test_christoffersen_rejects_clustered_breaches_with_the_right_count():
    # 50 breaches in 1,000 days is exactly 5%, so Kupiec is happy either way
    spread = np.zeros(1000, dtype=bool)
    spread[::20] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[300:350] = True
    assert spread.sum() == clustered.sum() == 50
    assert backtest.kupiec_pof(1000, 50, 0.95)[1] == pytest.approx(1.0)
    assert backtest.christoffersen_independence(clustered)[1] < 0.001
    assert backtest.christoffersen_independence(spread)[1] > 0.01


def test_christoffersen_has_nothing_to_say_without_breaches():
    assert backtest.christoffersen_independence(np.zeros(500, dtype=bool)) == (0.0, 1.0)
    assert backtest.christoffersen_independence([True]) == (0.0, 1.0)


def test_conditional_coverage_combines_both_tests():
    # chi-squared(2) survival function is exp(-x / 2); 5.9915 is its 95th percentile
    assert backtest.conditional_coverage_p_value(5.9915, 0.0) == pytest.approx(0.05, abs=1e-4)
    assert backtest.conditional_coverage_p_value(0.0, 0.0) == 1.0
    table = backtest.backtest_var(_returns(), window=250).table
    row = table.iloc[0]
    assert row["conditional_coverage_p"] == pytest.approx(
        math.exp(-(row["kupiec_lr"] + row["independence_lr"]) / 2))
    assert set(table["independence_result"]) <= {"pass", "reject"}
