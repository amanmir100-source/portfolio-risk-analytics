from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk import data_quality, database
from portfolio_risk.data_generator import assets_frame, generate_market_data, weights_frame

REPO = Path(__file__).resolve().parents[1]


def _prices(n_days: int = 60, seed: int = 3) -> pd.DataFrame:
    prices, _ = generate_market_data(n_days=n_days, seed=seed)
    return prices


def test_clean_data_passes_every_check():
    prices = _prices()
    data_quality.check_structure(prices)
    tickers = sorted(prices["ticker"].unique())
    assert data_quality.missing_closes(prices, tickers).empty
    assert data_quality.warnings_report(
        prices, data_quality.missing_closes(prices, tickers)).empty


def test_committed_demo_dataset_is_clean():
    prices = pd.read_csv(REPO / "data" / "demo_prices.csv")
    data_quality.check_structure(prices)
    tickers = sorted(prices["ticker"].unique())
    assert data_quality.warnings_report(
        prices, data_quality.missing_closes(prices, tickers)).empty


def test_duplicates_and_bad_closes_stop_the_run():
    prices = _prices()
    with pytest.raises(data_quality.DataQualityError, match="duplicate"):
        data_quality.check_structure(pd.concat([prices, prices.head(1)]))
    for bad in (0.0, -5.0, float("nan")):
        broken = prices.copy()
        broken.loc[broken.index[10], "close"] = bad
        with pytest.raises(data_quality.DataQualityError, match="negative or missing"):
            data_quality.check_structure(broken)


def test_missing_close_sql_matches_pandas(tmp_path):
    prices = _prices()
    dates = sorted(prices["date"].unique())
    holes = ((prices["date"] == dates[5]) & prices["ticker"].isin(["LQD", "EFA"])) | (
        (prices["date"] == dates[20]) & (prices["ticker"] == "GLD"))
    gappy = prices[~holes]
    db = tmp_path / "gappy.db"
    database.initialize_database(db, gappy, assets_frame(), weights_frame())
    sql = database.run_query_file(db, "data_quality")
    pandas_twin = data_quality.missing_closes(gappy, sorted(assets_frame()["ticker"]))
    pd.testing.assert_frame_equal(sql.reset_index(drop=True), pandas_twin,
                                  check_dtype=False)
    assert list(sql["missing_tickers"]) == ["EFA, LQD", "GLD"]


def test_extreme_moves_use_the_threshold():
    prices = _prices()
    spy = prices.index[prices["ticker"] == "SPY"]
    jumped = prices.copy()
    jumped.loc[spy[30:], "close"] *= 1.3  # a bad tick: SPY re-based 30% higher
    moves = data_quality.extreme_moves(jumped)
    assert list(moves["ticker"]) == ["SPY"]
    expected = 1.3 * prices.loc[spy[30], "close"] / prices.loc[spy[29], "close"] - 1
    assert moves["daily_return"].iloc[0] == pytest.approx(expected)
    assert data_quality.extreme_moves(jumped, threshold=0.5).empty


def test_stale_prices_need_five_identical_closes():
    prices = _prices()
    rows = prices.index[prices["ticker"] == "TLT"]
    four = prices.copy()
    four.loc[rows[10:14], "close"] = 99.0
    assert data_quality.stale_prices(four).empty
    five = prices.copy()
    five.loc[rows[10:15], "close"] = 99.0
    stale = data_quality.stale_prices(five)
    assert list(stale["ticker"]) == ["TLT"] and stale["days"].iloc[0] == 5
