# Changelog

## 1.1.0 - 2026-11-16

### Changed
- Real-data runs use the average 3-month T-bill yield over the analysis window
  as the risk-free rate instead of a flat 2%. The synthetic demo keeps 2%.
  `--risk-free` overrides either.
- The pipeline moved from `run_analysis.py` into `portfolio_risk/pipeline.py`,
  one function per stage, and reports progress through `logging`.
- SQLite is reloaded in a single transaction and connections are closed after
  use. A failed load now leaves the previous data in place.
- `asset_summary.sql` uses the sample standard deviation, matching the Python
  engine (it used the population formula before).
- The demo dataset's end date is pinned, so `--regenerate` rebuilds
  `data/demo_prices.csv` byte for byte instead of shifting its dates to today.
- CI installs pinned dependencies and runs ruff, mypy and a 90% coverage floor.

### Added
- Christoffersen's independence test and the conditional-coverage p-value in
  the VaR backtest, so breach clustering is tested as well as the breach count.
- `--live` saves a dated, hash-checked snapshot of the pull; `--snapshot DIR`
  re-runs it offline.
- `--portfolio FILE` and `--compare FILE ...`: weights from JSON files, and
  candidate portfolios compared side by side.
- A `load_log` table recording each load's source, date range, row count and
  content hash.
- 34 more tests (97 in total, 99% line coverage), covering the charts, the
  live loader against a stand-in for yfinance, and end-to-end runs.

### Not yet updated
- The live Sharpe, Sortino and max-Sharpe figures in the README come from the
  25 Sep 2026 pull at a 2% risk-free rate. They change on the next `--live` run.

## 1.0.0 - 2026-09-28

First complete version: data to SQLite to SQL analytics to risk engine to
report, with a VaR backtest, stress tests, Monte Carlo and the efficient frontier.
