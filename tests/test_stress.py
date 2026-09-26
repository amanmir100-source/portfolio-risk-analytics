import pandas as pd
import pytest

from portfolio_risk import stress


def _prices() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    return pd.DataFrame({"AAA": [100.0, 90.0, 80.0, 85.0],
                         "BBB": [50.0, 52.0, 55.0, 54.0]}, index=dates)


def test_scenario_return_is_weighted_sum_of_asset_returns():
    out = stress.run_scenarios(_prices(), {"AAA": 0.6, "BBB": 0.4},
                               {"test": ("2020-01-02", "2020-01-06")})
    row = out.loc["test"]
    assert row["AAA"] == pytest.approx(-0.20)  # 100 -> 80
    assert row["BBB"] == pytest.approx(0.10)   # 50 -> 55
    assert row["portfolio_return"] == pytest.approx(0.6 * -0.20 + 0.4 * 0.10)


def test_window_snaps_to_trading_days_inside_it():
    # 1 Jan and 5 Jan are not trading days: use 2 Jan and 3 Jan
    out = stress.run_scenarios(_prices(), {"AAA": 1.0},
                               {"test": ("2020-01-01", "2020-01-05")})
    assert (out.loc["test", "start"], out.loc["test", "end"]) == ("2020-01-02", "2020-01-03")
    assert out.loc["test", "portfolio_return"] == pytest.approx(-0.10)


def test_missing_history_raises():
    with pytest.raises(ValueError, match="not enough price history"):
        stress.run_scenarios(_prices(), {"AAA": 1.0}, {"old": ("2008-01-01", "2009-01-01")})
