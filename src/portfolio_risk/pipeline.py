"""The pipeline behind run_analysis.py, one function per stage.

    load inputs -> SQLite -> SQL analytics -> risk engine -> simulation
    -> VaR backtest -> stress tests -> portfolio comparison -> charts -> report

Each stage takes what it needs and returns what it produced, so stages can be
tested on their own and the order of work is readable in `run()`.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede any pyplot import

import pandas as pd  # noqa: E402

from . import (  # noqa: E402
    backtest,
    compare,
    config,
    data_generator,
    data_quality,
    database,
    metrics,
    risk_contribution,
    snapshot,
    stress,
    visualization,
)
from .monte_carlo import METHODS as MC_METHODS  # noqa: E402
from .monte_carlo import MonteCarloResult, simulate_portfolio  # noqa: E402
from .optimization import FrontierResult, efficient_frontier_analysis  # noqa: E402

log = logging.getLogger("portfolio_risk")

DESCRIPTION = """Portfolio Risk Analytics - end-to-end pipeline.

    python run_analysis.py                     # bundled demo dataset
    python run_analysis.py --regenerate        # rebuild the synthetic dataset
    python run_analysis.py --live              # real prices via yfinance (saves a snapshot)
    python run_analysis.py --snapshot DIR      # re-run a saved live pull, offline
    python run_analysis.py --compare portfolios/qqq_to_credit.json portfolios/qqq_to_gold.json
"""


# ----------------------------------------------------------------------------
# Inputs and intermediate results
# ----------------------------------------------------------------------------
@dataclass
class Inputs:
    prices: pd.DataFrame                 # long: date, ticker, close, volume
    source: str                          # human-readable description
    risk_free: float
    risk_free_source: str
    weights: dict[str, float]
    real_data: bool                      # live pull or snapshot (vs synthetic demo)
    history: pd.DataFrame | None = None  # long prices back to 2007, for stress tests


@dataclass
class Paths:
    repo_root: Path
    data_dir: Path
    db_path: Path
    reports_dir: Path
    figures_dir: Path
    sql_out_dir: Path

    @property
    def label(self) -> str:
        return self.reports_dir.relative_to(self.repo_root).as_posix()


@dataclass
class RiskResults:
    returns: pd.DataFrame
    port_r: pd.Series
    summary: pd.DataFrame
    episodes: pd.DataFrame
    contributions: pd.DataFrame


@dataclass
class SimulationResults:
    mc: MonteCarloResult
    mc_models: dict[str, MonteCarloResult]
    frontier: FrontierResult


@dataclass
class Results:
    inputs: Inputs
    paths: Paths
    sql: dict[str, pd.DataFrame]
    risk: RiskResults
    sims: SimulationResults
    backtest: backtest.BacktestResult
    scenarios: pd.DataFrame | None
    comparison: pd.DataFrame | None
    charts: list[Path] = field(default_factory=list)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--live", action="store_true",
                        help="pull real market data via yfinance and save a dated snapshot")
    source.add_argument("--snapshot", type=Path, metavar="DIR",
                        help="re-run a saved live pull from data/snapshots/<date>/ (offline)")
    source.add_argument("--regenerate", action="store_true",
                        help="regenerate the synthetic demo dataset before running")
    parser.add_argument("--sims", type=int, default=10_000,
                        help="number of Monte Carlo paths (default 10,000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the synthetic data generator")
    parser.add_argument("--portfolio", type=Path, metavar="FILE",
                        help="JSON file of weights to analyse (default: the model portfolio)")
    parser.add_argument("--compare", type=Path, nargs="+", metavar="FILE", default=[],
                        help="candidate portfolio files to compare against the analysed one")
    parser.add_argument("--risk-free", type=float, default=None, metavar="RATE",
                        help="annual risk-free rate as a fraction, e.g. 0.04 "
                             "(default: 2%% for demo data, average T-bill yield for real data)")
    volume = parser.add_mutually_exclusive_group()
    volume.add_argument("--quiet", action="store_true", help="only print warnings and errors")
    volume.add_argument("--verbose", action="store_true", help="print debug detail")
    return parser.parse_args(argv)


def configure_logging(args: argparse.Namespace) -> None:
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", force=True)
    for noisy in ("matplotlib", "PIL", "yfinance", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def resolve_paths(repo_root: Path, real_data: bool) -> Paths:
    # Real-data runs write to reports/live/ and their own database, so they
    # never overwrite the reproducible demo outputs that CI is built on.
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports" / "live" if real_data else repo_root / "reports"
    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        db_path=data_dir / ("portfolio_live.db" if real_data else "portfolio.db"),
        reports_dir=reports_dir,
        figures_dir=reports_dir / "figures",
        sql_out_dir=reports_dir / "sql",
    )


def load_inputs(args: argparse.Namespace, repo_root: Path) -> Inputs:
    """Stage 1: prices, weights and the risk-free rate, from whichever source."""
    weights = config.load_portfolio(args.portfolio) if args.portfolio else dict(
        config.PORTFOLIO_WEIGHTS)
    data_dir = repo_root / "data"

    if args.live:
        from . import live_data
        prices = live_data.fetch_live_prices()
        history = live_data.fetch_live_prices(start=stress.HISTORY_START)
        if args.risk_free is not None:
            rf, rf_source = args.risk_free, "set with --risk-free"
        else:
            rf = live_data.fetch_risk_free_rate(prices["date"].min(), prices["date"].max())
            rf_source = (f"average 3-month T-bill yield ({live_data.RISK_FREE_TICKER}) "
                         "over the window")
        as_of = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        source = "live market data (yfinance, adjusted closes)"
        saved = snapshot.save_snapshot(data_dir / "snapshots", as_of, prices, history, rf, source)
        log.info("      Snapshot   : %s (re-run it with --snapshot)",
                 saved.relative_to(repo_root).as_posix())
        return Inputs(prices, source, rf, rf_source, weights, True, history)

    if args.snapshot:
        snap = snapshot.load_snapshot(args.snapshot)
        if args.risk_free is not None:
            rf, rf_source = args.risk_free, "set with --risk-free"
        else:
            rf, rf_source = snap.risk_free_rate, "stored in the snapshot"
        return Inputs(snap.prices, f"snapshot of {snap.source}, pulled {snap.as_of}",
                      rf, rf_source, weights, True, snap.history)

    demo_csv = data_dir / "demo_prices.csv"
    if args.regenerate or not demo_csv.exists():
        seed = args.seed if args.seed is not None else data_generator.DEFAULT_SEED
        data_generator.generate_and_save(data_dir, seed=seed)
    if args.risk_free is not None:
        rf, rf_source = args.risk_free, "set with --risk-free"
    else:
        rf, rf_source = config.RISK_FREE_RATE, "fixed assumption for the synthetic demo data"
    return Inputs(pd.read_csv(demo_csv), f"demo dataset ({demo_csv.name})",
                  rf, rf_source, weights, False)


def load_database(inputs: Inputs, paths: Paths) -> dict[str, int]:
    """Stage 2: full, transactional reload of SQLite, recorded in load_log."""
    database.initialize_database(
        paths.db_path, inputs.prices, data_generator.assets_frame(),
        data_generator.weights_frame(inputs.weights), source=inputs.source,
    )
    return database.table_row_counts(paths.db_path)


def run_sql_analytics(inputs: Inputs, paths: Paths) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Stage 3: run every query in sql/ and build the data-quality report."""
    sql_results = database.run_all_analytics(paths.db_path, out_dir=paths.sql_out_dir)
    dq = data_quality.warnings_report(inputs.prices, sql_results["data_quality"])
    dq.to_csv(paths.reports_dir / "data_quality.csv", index=False)
    return sql_results, dq


def compute_risk(inputs: Inputs, paths: Paths) -> RiskResults:
    """Stage 4: returns, summary metrics, drawdown episodes, risk contributions."""
    returns = metrics.daily_returns(metrics.to_wide(inputs.prices))
    port_r = metrics.portfolio_returns(returns, inputs.weights)
    summary = metrics.summary_table(returns, inputs.weights, risk_free=inputs.risk_free)
    episodes = metrics.drawdown_episodes((1.0 + port_r).cumprod(), top=3)
    contributions = risk_contribution.risk_contributions(returns, inputs.weights)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(paths.reports_dir / "summary_metrics.csv")
    contributions.round(4).to_csv(paths.reports_dir / "risk_contributions.csv",
                                  index_label="ticker")
    episodes.round(4).to_csv(paths.reports_dir / "portfolio_drawdowns.csv", index=False)
    return RiskResults(returns, port_r, summary, episodes, contributions)


def run_simulations(inputs: Inputs, risk: RiskResults, paths: Paths, n_sims: int
                    ) -> SimulationResults:
    """Stage 5: Monte Carlo under three shock models, plus the efficient frontier."""
    mc_models = {m: simulate_portfolio(risk.returns, inputs.weights, n_sims=n_sims, method=m)
                 for m in MC_METHODS}
    table = pd.DataFrame([{
        "method": m,
        "var_95": r.var_amount,
        "cvar_95": r.cvar_amount,
        "prob_loss": r.prob_loss,
        "median_terminal": r.summary["median_terminal"],
        "detail": (f"dof {r.summary['student_t_dof']:.1f}" if m == "student_t" else
                   "21-day blocks" if m == "bootstrap" else ""),
    } for m, r in mc_models.items()])
    table.round(4).to_csv(paths.reports_dir / "mc_comparison.csv", index=False)
    frontier = efficient_frontier_analysis(risk.returns, inputs.weights,
                                           risk_free=inputs.risk_free)
    return SimulationResults(mc_models["normal"], mc_models, frontier)


def run_backtest(risk: RiskResults, paths: Paths) -> backtest.BacktestResult:
    """Stage 6: out-of-sample VaR backtest (coverage and independence)."""
    bt = backtest.backtest_var(risk.port_r)
    bt.table.round(4).to_csv(paths.reports_dir / "var_backtest.csv", index=False)
    return bt


def run_stress(inputs: Inputs, paths: Paths) -> pd.DataFrame | None:
    """Stage 7: replay past crises. Needs history back to 2007, so real data only."""
    if inputs.history is None:
        return None
    scenarios = stress.run_scenarios(metrics.to_wide(inputs.history), inputs.weights)
    scenarios.round(4).to_csv(paths.reports_dir / "stress_scenarios.csv")
    return scenarios


def run_comparison(args: argparse.Namespace, inputs: Inputs, risk: RiskResults, paths: Paths
                   ) -> pd.DataFrame | None:
    """Stage 8: the analysed portfolio next to any --compare candidates."""
    if not args.compare:
        return None
    portfolios = {"current": inputs.weights}
    for path in args.compare:
        portfolios[path.stem] = config.load_portfolio(path)
    history = metrics.to_wide(inputs.history) if inputs.history is not None else None
    table = compare.compare_portfolios(risk.returns, portfolios, inputs.risk_free, history)
    table.round(4).to_csv(paths.reports_dir / "portfolio_comparison.csv")
    return table


def draw_charts(results: Results) -> list[Path]:
    """Stage 9: the report figures."""
    risk, sims, fig = results.risk, results.sims, results.paths.figures_dir
    port = risk.summary.loc["PORTFOLIO"]
    charts = [
        visualization.plot_cumulative_growth(risk.returns, risk.port_r,
                                             fig / "01_cumulative_growth.png"),
        visualization.plot_correlation_heatmap(risk.returns.corr(),
                                               fig / "02_correlation_heatmap.png"),
        visualization.plot_rolling_volatility(results.sql["rolling_volatility"],
                                              fig / "03_rolling_volatility_sql.png"),
        visualization.plot_drawdown(metrics.drawdown_series(risk.port_r),
                                    fig / "04_portfolio_drawdown.png"),
        visualization.plot_return_distribution(
            risk.port_r,
            var_95=port["var_95_daily"],
            var_99=port["var_99_daily"],
            cvar_95=port["cvar_95_daily"],
            path=fig / "05_return_distribution_var.png"),
        visualization.plot_monte_carlo(sims.mc, fig / "06_monte_carlo.png"),
        visualization.plot_efficient_frontier(sims.frontier, fig / "07_efficient_frontier.png"),
        visualization.plot_monthly_heatmap(risk.port_r, fig / "08_monthly_returns_heatmap.png"),
        visualization.plot_var_backtest(results.backtest, fig / "09_var_backtest.png"),
        visualization.plot_mc_comparison(sims.mc_models, fig / "11_mc_shock_models.png"),
        visualization.plot_risk_contributions(risk.contributions,
                                              fig / "12_risk_contributions.png"),
    ]
    if results.scenarios is not None:
        charts.append(visualization.plot_stress_scenarios(
            results.scenarios, results.inputs.weights, fig / "10_stress_scenarios.png"))
    return charts


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def log_report(results: Results) -> None:
    """The plain-text summary printed at the end of a run."""
    inputs, risk, sims, bt = results.inputs, results.risk, results.sims, results.backtest
    port, mc, frontier = risk.summary.loc["PORTFOLIO"], sims.mc, sims.frontier
    out = results.paths.label

    log.info("\n%s\n  KEY PORTFOLIO RESULTS\n%s", "-" * 64, "-" * 64)
    log.info("  Risk-free rate         : %8.2f%%  (%s)",
             inputs.risk_free * 100, inputs.risk_free_source)
    log.info("  Annualised return      : %8.2f%%", port["ann_return"] * 100)
    log.info("  Annualised volatility  : %8.2f%%", port["ann_volatility"] * 100)
    log.info("  Sharpe ratio           : %8.2f", port["sharpe"])
    log.info("  Sortino ratio          : %8.2f", port["sortino"])
    log.info("  Max drawdown           : %8.2f%%", port["max_drawdown"] * 100)
    for e in risk.episodes.itertuples(index=False):
        back = (f"recovered {e.recovery_date} ({e.days_to_recover} days)"
                if pd.notna(e.recovery_date) else "not yet recovered")
        log.info("    %7.1f%%  peak %s → low %s, %s",
                 e.drawdown * 100, e.peak_date, e.trough_date, back)
    log.info("  Daily VaR (95%%)        : %8.2f%%", port["var_95_daily"] * 100)
    log.info("  Daily CVaR (95%%)       : %8.2f%%", port["cvar_95_daily"] * 100)
    top = (risk.contributions.drop(index="TOTAL")
           .sort_values("vol_share", ascending=False).head(3))
    log.info("  Biggest risk sources   : %s", " | ".join(
        f"{t} {r.weight:.0%} of weight → {r.vol_share:.0%} of vol, {r.cvar_share:.0%} of tail"
        for t, r in top.iterrows()))
    log.info("  1-yr MC VaR (95%%)      : $%10s on $100,000 (%.1f%%)",
             f"{mc.var_amount:,.0f}", mc.summary["var_95_pct"] * 100)
    log.info("  P(loss) over 1 yr      : %8.1f%%", mc.prob_loss * 100)
    log.info("  1-yr MC VaR by model   : %s", " | ".join(
        f"{m} ${r.var_amount:,.0f}" for m, r in sims.mc_models.items()))
    top_weights = frontier.max_sharpe_weights[frontier.max_sharpe_weights > 0.01]
    log.info("  Max-Sharpe (long-only) : {%s}",
             ", ".join(f"{t}: {float(v):.0%}" for t, v in top_weights.items()))
    log.info("    Sharpe %.2f exact vs %.2f best of 20,000 random (model portfolio %.2f)",
             frontier.max_sharpe, frontier.sampled_max_sharpe, frontier.current_sharpe)

    log.info("\n  VaR backtest at 5% significance "
             "(coverage = Kupiec, clustering = Christoffersen):")
    for row in bt.table.itertuples():
        log.info("    %-10s %.0f%%: %3d breaches vs %5.1f expected | "
                 "coverage p = %.3f → %s | clustering p = %.3f → %s",
                 row.method, row.confidence * 100, row.actual_breaches, row.expected_breaches,
                 row.p_value, row.result, row.independence_p, row.independence_result)

    if results.scenarios is not None:
        log.info("\n  Stress scenarios (today's weights, held through the crisis):")
        for name, row in results.scenarios.iterrows():
            log.info("    %-22s %s → %s: %7.1f%%",
                     name, row["start"], row["end"], row["portfolio_return"] * 100)

    if results.comparison is not None:
        log.info("\n  Portfolio comparison (also in %s/portfolio_comparison.csv):\n", out)
        log.info("%s", format_comparison(results.comparison).to_string())

    log.info("\n  Full per-asset table (also in %s/summary_metrics.csv):\n", out)
    log.info("%s", format_summary(risk.summary).to_string())


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.astype(object)
    pct_cols = ["weight", "ann_return", "ann_volatility", "max_drawdown",
                "var_95_daily", "cvar_95_daily", "var_99_daily"]
    for col in pct_cols:
        display[col] = summary[col].map(lambda v: f"{v:.1%}")
    for col in ("sharpe", "sortino", "calmar", f"beta_vs_{config.BENCHMARK_TICKER}"):
        display[col] = summary[col].map(lambda v: f"{v:.2f}")
    return display


def format_comparison(table: pd.DataFrame) -> pd.DataFrame:
    display = table.astype(object)
    for col in table.columns:
        if col == "largest_risk_source":
            continue
        if col == "sharpe":
            display[col] = table[col].map(lambda v: f"{v:.2f}")
        elif col in ("var_95_daily", "cvar_95_daily"):
            display[col] = table[col].map(lambda v: f"{v:.2%}")
        else:
            display[col] = table[col].map(lambda v: f"{v:.1%}")
    return display


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace, repo_root: Path) -> Results:
    log.info("%s\n  PORTFOLIO RISK ANALYTICS PIPELINE\n%s", "=" * 64, "=" * 64)

    inputs = load_inputs(args, repo_root)
    paths = resolve_paths(repo_root, inputs.real_data)
    out = paths.label
    prices = inputs.prices
    log.info("[1/9] Data       : %s", inputs.source)
    log.info("                 %s rows | %d tickers | %s → %s", f"{len(prices):,}",
             prices["ticker"].nunique(), prices["date"].min(), prices["date"].max())
    if args.portfolio:
        log.warning("      Portfolio  : weights from %s; outputs in %s/ now describe that "
                    "portfolio, not the default model", args.portfolio, out)
    data_quality.check_structure(prices)  # stops the run if the data is unusable

    counts = load_database(inputs, paths)
    log.info("[2/9] SQLite     : %s reloaded (%s)", paths.db_path.relative_to(repo_root).as_posix(),
             ", ".join(f"{t}={n:,}" for t, n in counts.items()))

    sql_results, dq = run_sql_analytics(inputs, paths)
    log.info("[3/9] SQL layer  : %d analytical queries (window functions, CTEs) → %s/sql/*.csv",
             len(sql_results), out)
    if dq.empty:
        log.info("      Data quality: clean (no gaps, extreme moves or stale prices)")
    else:
        log.warning("      Data quality: %d warning(s) → %s/data_quality.csv", len(dq), out)
        for r in dq.itertuples(index=False):
            log.warning("        %-13s %s  %s: %s", r.check, r.date, r.ticker, r.detail)

    risk = compute_risk(inputs, paths)
    log.info("[4/9] Risk engine: metrics for %d rows → %s/summary_metrics.csv",
             len(risk.summary), out)

    sims = run_simulations(inputs, risk, paths, args.sims)
    log.info("[5/9] Simulation : Monte Carlo %s paths x %d shock models | "
             "frontier cloud 20,000 portfolios", f"{sims.mc.n_sims:,}", len(sims.mc_models))

    bt = run_backtest(risk, paths)
    log.info("[6/9] Backtest   : VaR on %s out-of-sample days (%d-day window) "
             "→ %s/var_backtest.csv", f"{bt.table['test_days'].iloc[0]:,}", bt.window, out)

    scenarios = run_stress(inputs, paths)
    if scenarios is None:
        log.info("[7/9] Stress     : skipped (needs price history back to 2007; "
                 "use --live or --snapshot)")
    else:
        log.info("[7/9] Stress     : %d historical crises replayed → %s/stress_scenarios.csv",
                 len(scenarios), out)

    comparison = run_comparison(args, inputs, risk, paths)
    if comparison is None:
        log.info("[8/9] Comparison : skipped (pass --compare FILE ... to test candidate weights)")
    else:
        log.info("[8/9] Comparison : %d portfolios → %s/portfolio_comparison.csv",
                 len(comparison), out)

    results = Results(inputs, paths, sql_results, risk, sims, bt, scenarios, comparison)
    results.charts = draw_charts(results)
    log.info("[9/9] Charts     : %d figures → %s/figures/", len(results.charts), out)

    log_report(results)
    return results


def main(repo_root: Path, argv: list[str] | None = None) -> Results:
    args = parse_args(argv)
    configure_logging(args)
    started = time.perf_counter()
    results = run(args, repo_root)
    log.info("\nDone in %.1fs.", time.perf_counter() - started)
    return results
