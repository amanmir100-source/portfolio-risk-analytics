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
