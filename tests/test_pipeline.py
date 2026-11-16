"""End-to-end runs of the pipeline in a temporary folder (no network)."""
from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk import config, live_data, metrics, pipeline, snapshot
from portfolio_risk.data_generator import generate_market_data

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [str(REPO / "portfolios" / "qqq_to_credit.json"),
              str(REPO / "portfolios" / "qqq_to_gold.json")]


@pytest.fixture(scope="module")
def long_history():
    history, _ = generate_market_data(n_days=5200, seed=4)  # reaches back past 2007
    return history


def test_demo_run_writes_every_output(tmp_path):
    results = pipeline.main(tmp_path, ["--sims", "300", "--quiet", "--compare", *CANDIDATES])
    reports = tmp_path / "reports"
    assert (tmp_path / "data" / "demo_prices.csv").exists()  # generated on first run
    for name in ("summary_metrics", "risk_contributions", "portfolio_drawdowns",
                 "mc_comparison", "var_backtest", "data_quality", "portfolio_comparison"):
        assert (reports / f"{name}.csv").exists(), name
    assert len(list((reports / "sql").glob("*.csv"))) == 6
    assert len(results.charts) == 11 and all(p.stat().st_size > 0 for p in results.charts)
    assert results.scenarios is None  # demo data has no 2008 to replay
    assert list(results.comparison.index) == ["current", "qqq_to_credit", "qqq_to_gold"]
    assert results.inputs.risk_free == config.RISK_FREE_RATE


def test_snapshot_run_uses_the_stored_rate_and_replays_crises(tmp_path, long_history):
    prices = long_history[long_history["date"] >= "2021-07-01"]
    saved = snapshot.save_snapshot(tmp_path / "data" / "snapshots", "2026-07-03",
                                   prices, long_history, 0.04, "unit test feed")
    results = pipeline.main(tmp_path, ["--snapshot", str(saved), "--sims", "300", "--quiet"])
    assert results.paths.reports_dir == tmp_path / "reports" / "live"
    assert results.paths.db_path.name == "portfolio_live.db"
    assert len(results.charts) == 12  # the stress chart is included
    assert len(results.scenarios) == 3
    port = metrics.portfolio_returns(results.risk.returns, config.PORTFOLIO_WEIGHTS)
    assert results.risk.summary.loc["PORTFOLIO", "sharpe"] == pytest.approx(
        metrics.sharpe_ratio(port, 0.04))
    assert results.sims.frontier.current_sharpe == pytest.approx(
        metrics.sharpe_ratio(port, 0.04), rel=1e-9)


def test_live_run_saves_a_snapshot_that_reproduces_it(tmp_path, long_history, monkeypatch):
    window = long_history[long_history["date"] >= "2021-07-01"]
    monkeypatch.setattr(live_data, "fetch_live_prices",
                        lambda tickers=None, years=5, start=None: long_history if start else window)
    monkeypatch.setattr(live_data, "fetch_risk_free_rate", lambda start, end: 0.037)
    live = pipeline.main(tmp_path, ["--live", "--sims", "300", "--quiet"])
    assert live.inputs.risk_free == 0.037
    saved = list((tmp_path / "data" / "snapshots").iterdir())
    assert len(saved) == 1
    again = pipeline.main(tmp_path, ["--snapshot", str(saved[0]), "--sims", "300", "--quiet"])
    pd.testing.assert_frame_equal(live.risk.summary, again.risk.summary)
    pd.testing.assert_frame_equal(live.backtest.table, again.backtest.table)


def test_risk_free_flag_overrides_the_default(tmp_path):
    gold = str(REPO / "portfolios" / "qqq_to_gold.json")
    results = pipeline.main(tmp_path, ["--sims", "200", "--quiet", "--risk-free", "0.05",
                                       "--portfolio", gold])
    assert results.inputs.risk_free == 0.05
    assert results.inputs.weights["GLD"] == pytest.approx(0.15)
    weights = pd.read_csv(tmp_path / "reports" / "sql" / "asset_summary.csv").set_index("ticker")
    assert weights.loc["GLD", "portfolio_weight"] == pytest.approx(0.15)


def test_source_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--live", "--regenerate"])
