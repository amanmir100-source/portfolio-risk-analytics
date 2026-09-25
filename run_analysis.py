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

from portfolio_risk import config, data_generator, database, metrics, visualization
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
    print(f"[1/6] Data       : {source}")
    print(f"                 {len(prices):,} rows | {len(tickers)} tickers | "
          f"{prices['date'].min()} → {prices['date'].max()}")

    # ---------------------------------------------------------------- sqlite
    database.initialize_database(
        db_path, prices, data_generator.assets_frame(), data_generator.weights_frame()
    )
    counts = database.table_row_counts(db_path)
    print(f"[2/6] SQLite     : {db_path.relative_to(REPO_ROOT)} loaded "
          f"({', '.join(f'{t}={n:,}' for t, n in counts.items())})")

    # ----------------------------------------------------------- sql analytics
    sql_results = database.run_all_analytics(db_path, out_dir=sql_out_dir)
    print(f"[3/6] SQL layer  : {len(sql_results)} analytical queries "
          f"(window functions, CTEs) → {out}/sql/*.csv")

    # ------------------------------------------------------------- risk engine
    wide = metrics.to_wide(prices)
    returns = metrics.daily_returns(wide)
    port_r = metrics.portfolio_returns(returns, config.PORTFOLIO_WEIGHTS)
    summary = metrics.summary_table(returns, config.PORTFOLIO_WEIGHTS)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(reports_dir / "summary_metrics.csv")
    print(f"[4/6] Risk engine: metrics for {len(summary)} rows → {out}/summary_metrics.csv")

    # -------------------------------------------------------------- simulations
    mc = simulate_portfolio(returns, config.PORTFOLIO_WEIGHTS, n_sims=args.sims)
    frontier = efficient_frontier_analysis(returns, config.PORTFOLIO_WEIGHTS)
    print(f"[5/6] Simulation : Monte Carlo {mc.n_sims:,} paths | "
          f"frontier cloud 20,000 portfolios")

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
    ]
    print(f"[6/6] Charts     : {len(charts)} figures → {out}/figures/")

    # ------------------------------------------------------------------ report
    print("\n" + "-" * 64)
    print("  KEY PORTFOLIO RESULTS")
    print("-" * 64)
    print(f"  Annualised return      : {port_summary['ann_return']:>8.2%}")
    print(f"  Annualised volatility  : {port_summary['ann_volatility']:>8.2%}")
    print(f"  Sharpe ratio           : {port_summary['sharpe']:>8.2f}")
    print(f"  Sortino ratio          : {port_summary['sortino']:>8.2f}")
    print(f"  Max drawdown           : {port_summary['max_drawdown']:>8.2%}")
    print(f"  Daily VaR (95%)        : {port_summary['var_95_daily']:>8.2%}")
    print(f"  Daily CVaR (95%)       : {port_summary['cvar_95_daily']:>8.2%}")
    print(f"  1-yr MC VaR (95%)      : ${mc.var_amount:>10,.0f} on $100,000 "
          f"({mc.summary['var_95_pct']:.1%})")
    print(f"  P(loss) over 1 yr      : {mc.prob_loss:>8.1%}")
    top_weights = frontier.max_sharpe_weights[frontier.max_sharpe_weights > 0.01]
    print(f"  Max-Sharpe (long-only) : "
          f"{{{', '.join(f'{t}: {float(v):.0%}' for t, v in top_weights.items())}}}")

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
