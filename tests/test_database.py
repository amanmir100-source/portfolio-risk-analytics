"""The key idea here: every SQL result is cross-validated against an
independent pandas implementation of the same statistic."""
import sqlite3

import numpy as np
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
