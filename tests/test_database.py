"""The key idea here: every SQL result is cross-validated against an
independent pandas implementation of the same statistic."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import database, metrics
from portfolio_risk.data_generator import (
    assets_frame,
    generate_market_data,
    weights_frame,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    prices, _ = generate_market_data(n_days=280, seed=5)
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    database.initialize_database(db_path, prices, assets_frame(), weights_frame())
    return db_path, prices


def test_tables_loaded_with_expected_row_counts(db):
    db_path, prices = db
    counts = database.table_row_counts(db_path)
    assert counts["prices"] == len(prices)
    assert counts["assets"] == 8
    assert counts["portfolio_weights"] == 8


def test_sql_daily_returns_match_pandas(db):
    db_path, prices = db
    sql = database.run_query_file(db_path, "daily_returns")
    wide = metrics.to_wide(prices)
    pandas_r = wide["SPY"].pct_change().dropna().to_numpy()
    sql_r = (
        sql[sql["ticker"] == "SPY"].sort_values("date")["daily_return"]
        .dropna()
        .to_numpy()
    )
    assert np.allclose(sql_r, pandas_r, atol=1e-12)


def test_sql_rolling_volatility_matches_pandas(db):
    db_path, prices = db
    sql = database.run_query_file(db_path, "rolling_volatility")
    wide = metrics.to_wide(prices)
    pandas_vol = (
        wide["QQQ"].pct_change().rolling(21).std(ddof=1) * np.sqrt(252)
    ).dropna().to_numpy()
    sql_vol = (
        sql[sql["ticker"] == "QQQ"].sort_values("date")["ann_volatility_21d"]
        .to_numpy()
    )
    assert len(sql_vol) == len(pandas_vol)
    assert np.allclose(sql_vol, pandas_vol, rtol=1e-9)


def test_sql_monthly_returns_match_pandas(db):
    db_path, prices = db
    sql = database.run_query_file(db_path, "monthly_performance")
    wide = metrics.to_wide(prices)
    pandas_m = wide["TLT"].resample("ME").last().pct_change().dropna().to_numpy()
    sql_m = (
        sql[sql["ticker"] == "TLT"].sort_values("month")["monthly_return"]
        .dropna()
        .to_numpy()
    )
    assert np.allclose(sql_m, pandas_m, atol=1e-12)


def test_sql_worst_drawdown_matches_pandas(db):
    db_path, prices = db
    sql = database.run_query_file(db_path, "drawdown_events")
    wide = metrics.to_wide(prices)
    returns = wide["SPY"].pct_change().dropna()
    pandas_worst = metrics.drawdown_series(returns).min()
    sql_worst = sql[(sql["ticker"] == "SPY") & (sql["dd_rank"] == 1)][
        "drawdown_pct"
    ].iloc[0]
    assert sql_worst == pytest.approx(round(pandas_worst * 100, 2), abs=0.02)


def test_sql_drawdown_episodes_match_pandas(db):
    db_path, prices = db
    sql = database.run_query_file(db_path, "drawdown_events")
    wide = metrics.to_wide(prices)
    for ticker in wide.columns:
        ours = sql[sql["ticker"] == ticker].reset_index(drop=True)
        twin = metrics.drawdown_episodes(wide[ticker], top=5)
        assert list(ours["peak_date"]) == list(twin["peak_date"]), ticker
        assert list(ours["trough_date"]) == list(twin["trough_date"]), ticker
        assert list(ours["recovery_date"].fillna("none")) == list(
            twin["recovery_date"].fillna("none")), ticker
        assert ours["drawdown_pct"].to_numpy() == pytest.approx(
            (twin["drawdown"] * 100).round(2).to_numpy(), abs=0.011)


def test_drawdown_episodes_are_separate_falls_not_days():
    # two falls: 100 -> 80 -> 100 (recovered), then 110 -> 88 (not recovered)
    closes = [100, 90, 80, 95, 100, 110, 99, 88, 90]
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    ep = metrics.drawdown_episodes(pd.Series(closes, index=dates, dtype=float))
    assert len(ep) == 2  # two episodes, not the five worst days
    first, second = ep.sort_values("peak_date").to_dict("records")
    assert first["drawdown"] == pytest.approx(-0.20)
    assert first["recovery_date"] == dates[4].strftime("%Y-%m-%d")
    assert (first["days_to_trough"], first["days_to_recover"]) == (2, 4)
    assert second["drawdown"] == pytest.approx(-0.20)
    assert pd.isna(second["recovery_date"])  # None or NaN depending on pandas


def test_asset_summary_joins_weights_and_reference_data(db):
    db_path, _ = db
    summary = database.run_query_file(db_path, "asset_summary")
    assert len(summary) == 8
    assert summary["portfolio_weight"].sum() == pytest.approx(1.0)
    assert (summary["ann_volatility_pct"] > 0).all()
    spy = summary[summary["ticker"] == "SPY"].iloc[0]
    assert spy["asset_class"] == "US Large-Cap Equity"


def test_foreign_keys_reject_unknown_tickers(db):
    db_path, _ = db
    with database.get_connection(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO prices (date, ticker, close, volume) "
                "VALUES ('2026-01-01', 'FAKE', 100.0, 1)"
            )


def test_sql_asset_volatility_matches_pandas_sample_stdev(db):
    db_path, prices = db
    summary = database.run_query_file(db_path, "asset_summary").set_index("ticker")
    returns = metrics.daily_returns(metrics.to_wide(prices))
    for ticker in returns.columns:
        expected = returns[ticker].std(ddof=1) * np.sqrt(252) * 100
        assert summary.loc[ticker, "ann_volatility_pct"] == pytest.approx(expected, abs=0.006)


def test_connection_is_closed_after_use(db):
    db_path, _ = db
    with database.get_connection(db_path) as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_every_load_is_recorded_with_a_content_hash(tmp_path):
    prices, _ = generate_market_data(n_days=30, seed=2)
    db_path = tmp_path / "audit.db"
    for _ in range(2):
        database.initialize_database(db_path, prices, assets_frame(), weights_frame(),
                                     source="unit test")
    log = database.run_query(db_path, "SELECT * FROM load_log ORDER BY load_id")
    assert len(log) == 2  # the audit trail survives a reload; prices do not double up
    assert database.table_row_counts(db_path)["prices"] == len(prices)
    last = log.iloc[-1]
    assert last["source"] == "unit test"
    assert last["row_count"] == len(prices)
    assert (last["first_date"], last["last_date"]) == (prices["date"].min(), prices["date"].max())
    assert last["content_sha256"] == database.content_hash(prices)
    shuffled = prices.sample(frac=1.0, random_state=0)
    assert database.content_hash(shuffled) == database.content_hash(prices)


def test_failed_load_leaves_the_previous_data_in_place(tmp_path):
    good, _ = generate_market_data(n_days=30, seed=2)
    db_path = tmp_path / "atomic.db"
    database.initialize_database(db_path, good, assets_frame(), weights_frame())
    bad = good.copy()
    bad.loc[bad.index[-1], "close"] = -1.0  # violates CHECK (close > 0)
    with pytest.raises(sqlite3.IntegrityError):
        database.initialize_database(db_path, bad, assets_frame(), weights_frame())
    counts = database.table_row_counts(db_path)
    assert counts["prices"] == len(good)
    assert counts["load_log"] == 1
