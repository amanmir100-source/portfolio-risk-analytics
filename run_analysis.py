#!/usr/bin/env python3
"""Portfolio Risk Analytics — end-to-end pipeline.

    data (synthetic or live) -> SQLite -> SQL analytics -> NumPy/pandas
    risk engine -> matplotlib report

Usage
-----
    python run_analysis.py                 # bundled demo dataset
    python run_analysis.py --regenerate    # fresh synthetic dataset
    python run_analysis.py --live          # real prices via yfinance
    python run_analysis.py --sims 50000    # heavier Monte Carlo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede any pyplot import

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

try:
    import portfolio_risk  # noqa: F401  (installed via `pip install -e .`)
except ImportError:  # fallback: run straight from the repo without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from portfolio_risk import (backtest, config, data_generator, data_quality, database, metrics,
                            stress, visualization)
from portfolio_risk.monte_carlo import METHODS as MC_METHODS
from portfolio_risk.monte_carlo import simulate_portfolio
from portfolio_risk.optimization import efficient_frontier_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true",
                        help="pull real market data via yfinance instead of the demo dataset")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the synthetic demo dataset before running")
    parser.add_argument("--sims", type=int, default=10_000,
                        help="number of Monte Carlo paths (default 10,000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the synthetic data generator")
    return parser.parse_args()


def load_prices(args: argparse.Namespace, data_dir: Path) -> tuple[pd.DataFrame, str]:
    demo_csv = data_dir / "demo_prices.csv"
    if args.live:
        from portfolio_risk.live_data import fetch_live_prices
        return fetch_live_prices(), "live market data (yfinance, adjusted closes)"
    if args.regenerate or not demo_csv.exists():
        seed = args.seed if args.seed is not None else data_generator.DEFAULT_SEED
        data_generator.generate_and_save(data_dir, seed=seed)
    return pd.read_csv(demo_csv), f"demo dataset ({demo_csv.name})"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    data_dir = REPO_ROOT / "data"
    db_path = data_dir / "portfolio.db"
    # Live runs write to reports/live/ so they never overwrite the reproducible
    # demo outputs that the README gallery and CI are built on.
    reports_dir = REPO_ROOT / "reports" / "live" if args.live else REPO_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    sql_out_dir = reports_dir / "sql"
    out = reports_dir.relative_to(REPO_ROOT).as_posix()

    print("=" * 64)
    print("  PORTFOLIO RISK ANALYTICS PIPELINE")
    print("=" * 64)

    # ------------------------------------------------------------------ data
    prices, source = load_prices(args, data_dir)
    tickers = sorted(prices["ticker"].unique())
    print(f"[1/8] Data       : {source}")
    print(f"                 {len(prices):,} rows | {len(tickers)} tickers | "
          f"{prices['date'].min()} → {prices['date'].max()}")
    data_quality.check_structure(prices)  # stops the run if the data is unusable

    # ---------------------------------------------------------------- sqlite
    database.initialize_database(
        db_path, prices, data_generator.assets_frame(), data_generator.weights_frame()
    )
    counts = database.table_row_counts(db_path)
    print(f"[2/8] SQLite     : {db_path.relative_to(REPO_ROOT)} loaded "
          f"({', '.join(f'{t}={n:,}' for t, n in counts.items())})")

    # ----------------------------------------------------------- sql analytics
    sql_results = database.run_all_analytics(db_path, out_dir=sql_out_dir)
    print(f"[3/8] SQL layer  : {len(sql_results)} analytical queries "
          f"(window functions, CTEs) → {out}/sql/*.csv")
    dq = data_quality.warnings_report(prices, sql_results["data_quality"])
    dq.to_csv(reports_dir / "data_quality.csv", index=False)
    if dq.empty:
        print("      Data quality: clean (no gaps, extreme moves or stale prices)")
    else:
        print(f"      Data quality: {len(dq)} warning(s) → {out}/data_quality.csv")
        for r in dq.itertuples(index=False):
            print(f"        {r.check:<13} {r.date}  {r.ticker}: {r.detail}")

    # ------------------------------------------------------------- risk engine
    wide = metrics.to_wide(prices)
    returns = metrics.daily_returns(wide)
    port_r = metrics.portfolio_returns(returns, config.PORTFOLIO_WEIGHTS)
    summary = metrics.summary_table(returns, config.PORTFOLIO_WEIGHTS)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(reports_dir / "summary_metrics.csv")
    port_episodes = metrics.drawdown_episodes((1.0 + port_r).cumprod(), top=3)
    port_episodes.round(4).to_csv(reports_dir / "portfolio_drawdowns.csv", index=False)
    print(f"[4/8] Risk engine: metrics for {len(summary)} rows → {out}/summary_metrics.csv")

    # -------------------------------------------------------------- simulations
    mc = simulate_portfolio(returns, config.PORTFOLIO_WEIGHTS, n_sims=args.sims)
    mc_models = {m: mc if m == "normal" else
                 simulate_portfolio(returns, config.PORTFOLIO_WEIGHTS, n_sims=args.sims, method=m)
                 for m in MC_METHODS}
    mc_table = pd.DataFrame([{
        "method": m,
        "var_95": r.var_amount,
        "cvar_95": r.cvar_amount,
        "prob_loss": r.prob_loss,
        "median_terminal": r.summary["median_terminal"],
        "detail": (f"dof {r.summary['student_t_dof']:.1f}" if m == "student_t" else
                   "21-day blocks" if m == "bootstrap" else ""),
    } for m, r in mc_models.items()])
    mc_table.round(4).to_csv(reports_dir / "mc_comparison.csv", index=False)
    frontier = efficient_frontier_analysis(returns, config.PORTFOLIO_WEIGHTS)
    print(f"[5/8] Simulation : Monte Carlo {mc.n_sims:,} paths x {len(mc_models)} shock models | "
          f"frontier cloud 20,000 portfolios")

    # ------------------------------------------------------------- var backtest
    bt = backtest.backtest_var(port_r)
    bt.table.round(4).to_csv(reports_dir / "var_backtest.csv", index=False)
    print(f"[6/8] Backtest   : VaR on {bt.table['test_days'].iloc[0]:,} out-of-sample days "
          f"({bt.window}-day window) → {out}/var_backtest.csv")

    # ------------------------------------------------------------ stress tests
    scenarios = None
    if args.live:
        from portfolio_risk.live_data import fetch_live_prices
        history = metrics.to_wide(fetch_live_prices(start=stress.HISTORY_START))
        scenarios = stress.run_scenarios(history, config.PORTFOLIO_WEIGHTS)
        scenarios.round(4).to_csv(reports_dir / "stress_scenarios.csv")
        print(f"[7/8] Stress     : {len(scenarios)} historical crises replayed "
              f"→ {out}/stress_scenarios.csv")
    else:
        print("[7/8] Stress     : skipped (needs price history back to 2007; use --live)")

    # ------------------------------------------------------------------ charts
    port_summary = summary.loc["PORTFOLIO"]
    charts = [
        visualization.plot_cumulative_growth(returns, port_r,
                                             figures_dir / "01_cumulative_growth.png"),
        visualization.plot_correlation_heatmap(returns.corr(),
                                               figures_dir / "02_correlation_heatmap.png"),
        visualization.plot_rolling_volatility(sql_results["rolling_volatility"],
                                              figures_dir / "03_rolling_volatility_sql.png"),
        visualization.plot_drawdown(metrics.drawdown_series(port_r),
                                    figures_dir / "04_portfolio_drawdown.png"),
        visualization.plot_return_distribution(
            port_r,
            var_95=port_summary["var_95_daily"],
            var_99=port_summary["var_99_daily"],
            cvar_95=port_summary["cvar_95_daily"],
            path=figures_dir / "05_return_distribution_var.png"),
        visualization.plot_monte_carlo(mc, figures_dir / "06_monte_carlo.png"),
        visualization.plot_efficient_frontier(frontier,
                                              figures_dir / "07_efficient_frontier.png"),
        visualization.plot_monthly_heatmap(port_r,
                                           figures_dir / "08_monthly_returns_heatmap.png"),
        visualization.plot_var_backtest(bt, figures_dir / "09_var_backtest.png"),
        visualization.plot_mc_comparison(mc_models, figures_dir / "11_mc_shock_models.png"),
    ]
    if scenarios is not None:
        charts.append(visualization.plot_stress_scenarios(
            scenarios, config.PORTFOLIO_WEIGHTS, figures_dir / "10_stress_scenarios.png"))
    print(f"[8/8] Charts     : {len(charts)} figures → {out}/figures/")

    # ------------------------------------------------------------------ report
    print("\n" + "-" * 64)
    print("  KEY PORTFOLIO RESULTS")
    print("-" * 64)
    print(f"  Annualised return      : {port_summary['ann_return']:>8.2%}")
    print(f"  Annualised volatility  : {port_summary['ann_volatility']:>8.2%}")
    print(f"  Sharpe ratio           : {port_summary['sharpe']:>8.2f}")
    print(f"  Sortino ratio          : {port_summary['sortino']:>8.2f}")
    print(f"  Max drawdown           : {port_summary['max_drawdown']:>8.2%}")
    for e in port_episodes.itertuples(index=False):
        back = (f"recovered {e.recovery_date} ({e.days_to_recover} days)"
                if pd.notna(e.recovery_date) else "not yet recovered")
        print(f"    {e.drawdown:>7.1%}  peak {e.peak_date} → low {e.trough_date}, {back}")
    print(f"  Daily VaR (95%)        : {port_summary['var_95_daily']:>8.2%}")
    print(f"  Daily CVaR (95%)       : {port_summary['cvar_95_daily']:>8.2%}")
    print(f"  1-yr MC VaR (95%)      : ${mc.var_amount:>10,.0f} on $100,000 "
          f"({mc.summary['var_95_pct']:.1%})")
    print(f"  P(loss) over 1 yr      : {mc.prob_loss:>8.1%}")
    print("  1-yr MC VaR by model   : " + " | ".join(
        f"{m} ${r.var_amount:,.0f}" for m, r in mc_models.items()))
    top_weights = frontier.max_sharpe_weights[frontier.max_sharpe_weights > 0.01]
    print(f"  Max-Sharpe (long-only) : "
          f"{{{', '.join(f'{t}: {float(v):.0%}' for t, v in top_weights.items())}}}")

    print("\n  VaR backtest (Kupiec test, 5% significance):")
    for row in bt.table.itertuples():
        print(f"    {row.method:<10} {row.confidence:.0%}: {row.actual_breaches:>3} breaches "
              f"vs {row.expected_breaches:>5.1f} expected | p = {row.p_value:.3f} → {row.result}")

    if scenarios is not None:
        print("\n  Stress scenarios (today's weights, held through the crisis):")
        for name, row in scenarios.iterrows():
            print(f"    {name:<22} {row['start']} → {row['end']}: {row['portfolio_return']:>7.1%}")

    print(f"\n  Full per-asset table (also in {out}/summary_metrics.csv):\n")
    display = summary.copy()
    pct_cols = ["weight", "ann_return", "ann_volatility", "max_drawdown",
                "var_95_daily", "cvar_95_daily", "var_99_daily"]
    for col in pct_cols:
        display[col] = display[col].map(lambda v: f"{v:.1%}")
    for col in ("sharpe", "sortino", "calmar", f"beta_vs_{config.BENCHMARK_TICKER}"):
        display[col] = display[col].map(lambda v: f"{v:.2f}")
    print(display.to_string())
    print(f"\nDone in {time.perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()
