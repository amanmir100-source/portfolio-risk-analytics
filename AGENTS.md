# AGENTS.md — project context for AI coding assistants

## What this is

End-to-end portfolio risk analytics pipeline: market data (synthetic by default,
yfinance with `--live`) → SQLite → analytical SQL → NumPy/pandas risk engine →
matplotlib report. Stack is deliberately minimal: numpy, pandas, matplotlib,
stdlib sqlite3. No SciPy (normal quantiles come from `statistics.NormalDist`).

## Commands

```bash
pip install -e ".[dev]"            # or: pip install -r requirements-dev.txt
python run_analysis.py             # full pipeline on the bundled demo dataset (~10s)
python run_analysis.py --live      # real market data via yfinance (+ stress tests); saves a snapshot
python run_analysis.py --snapshot data/snapshots/<date>   # re-run a saved live pull, offline
python run_analysis.py --compare portfolios/qqq_to_credit.json portfolios/qqq_to_gold.json
python run_analysis.py --regenerate --seed 40   # rebuild the demo dataset
pytest                             # 98 tests, ~25s
ruff check . && mypy && pytest --cov   # what CI runs (coverage floor 90%)
```

CI installs the pinned versions: `pip install -r requirements-lock.txt && pip install --no-deps -e .`.
After changing dependencies in `pyproject.toml`, regenerate the lock file with
`uv pip compile pyproject.toml --extra live --extra dev --universal --python-version 3.10 -o requirements-lock.txt`.

## Architecture map

- `sql/` — ALL analytical SQL lives here as standalone .sql files (schema,
  daily returns, rolling volatility, monthly performance, drawdown episodes,
  asset summary, data-quality gaps). Never inline SQL strings in Python; add a new .sql file and
  register it in `database.ANALYTICS_QUERIES`.
- `src/portfolio_risk/data_quality.py` — runs on load. Duplicates and bad
  closes raise DataQualityError (the DB constraints would reject them anyway);
  gaps, 25%+ moves and 5-day stale runs are warnings in data_quality.csv.
  `missing_closes()` is the pandas twin of `sql/data_quality.sql`.
- `src/portfolio_risk/config.py` — asset universe, portfolio weights, regime
  parameters. Correlation matrices are order-sensitive (follow `TICKERS`).
- `src/portfolio_risk/data_generator.py` — regime-switching correlated GBM.
  `DEFAULT_SEED = 40` is what produced the committed `data/demo_prices.csv`;
  don't change the seed without regenerating data, reports and README numbers.
- `src/portfolio_risk/metrics.py` — risk/performance metrics. Conventions:
  252 trading days, VaR/CVaR reported as POSITIVE losses. The risk-free rate
  is passed in: 2% (`config.RISK_FREE_RATE`) for the synthetic demo, the
  window-average 3-month T-bill yield for real data, or `--risk-free`;
  annualised return and Calmar are geometric; Sharpe and Sortino use the
  arithmetic mean x 252 so they match the frontier (one Sharpe everywhere).
- `src/portfolio_risk/monte_carlo.py`, `optimization.py` — simulation and
  closed-form Markowitz frontier (pure linear algebra, no optimiser).
  Long-only max-Sharpe is exact: tangency weights on every asset subset,
  best all-positive one wins (2^n subsets, fine for 8 assets).
  `simulate_portfolio(method=...)` takes "normal" (default; keep its output
  unchanged), "student_t" or "bootstrap". Each method must stay reproducible
  and give the same result for any chunk_size.
- `src/portfolio_risk/backtest.py` — rolling out-of-sample VaR backtest with
  Kupiec's coverage test and Christoffersen's independence test. Forecasts must only use past days; `shift(1)` is what enforces it.
- `src/portfolio_risk/stress.py` — replays `SCENARIOS` (S&P 500 peak/trough
  dates) on current weights. Needs history back to 2007, so it only runs with
  `--live`; the demo run skips it.
- `src/portfolio_risk/risk_contribution.py` — Euler volatility and CVaR
  contributions per asset; both must sum exactly to the portfolio totals.
- `src/portfolio_risk/pipeline.py` — the run itself, one function per stage
  (load inputs, load database, SQL, risk, simulations, backtest, stress,
  comparison, charts, report). Output goes through `logging`, not `print`.
  Writes `reports/` (demo) or `reports/live/` (`--live` / `--snapshot`), and
  real data uses its own `data/portfolio_live.db`, so a real-data run never
  overwrites the demo outputs. `run_analysis.py` is only the entry point.
- `src/portfolio_risk/snapshot.py` — dated, hash-checked copies of a live pull
  under `data/snapshots/<date>/` (prices, long history, risk-free rate).
  Snapshots are committed; they are what makes live results repeatable.
- `src/portfolio_risk/compare.py` + `portfolios/*.json` — candidate weightings
  side by side. `config.load_portfolio` validates a file (known tickers,
  non-negative, sums to 1).
- `src/portfolio_risk/live_data.py` — yfinance loader and
  `fetch_risk_free_rate` (`^IRX`). Tests replace yfinance with a stand-in;
  never add a test that needs the network.

## Testing rules

Every SQL query has a pytest cross-check against an independent pandas
implementation (`tests/test_database.py`). If you change a query, update its
pandas twin in the same PR — a query without a cross-check is a regression.
Everything is seeded; tests must stay deterministic.

## Gotchas

- SQLite has no `STDDEV`: rolling vol derives Bessel-corrected stdev from
  AVG(r²) − AVG(r)² inside the window; keep the `MAX(..., 0.0)` guard. Its
  output is rounded in Python (`database.ROUND_DECIMALS`), not with SQL
  ROUND(), which differs between SQLite versions.
- `reports/` (demo) and `reports/live/` (real data) are both committed. The
  README headline and gallery embed `reports/live/`; regenerate with
  `python run_analysis.py` and `--live` after any change that shifts the numbers,
  then update the README figures and data window.
- The database is reloaded in full on every run, inside one transaction
  (`database.initialize_database`), and each load is recorded in `load_log`.
  Do not switch to appending new dates: adjusted closes are restated after
  every dividend, so an append would splice prices adjusted on different days.
  Inserts use `executemany`, not `to_sql`, which commits per table.
- `database.get_connection` is a context manager that closes the connection;
  always use it in a `with` block.
- The demo dataset's end date is pinned (`data_generator.DEFAULT_END_DATE`), so
  `--regenerate` reproduces `data/demo_prices.csv` byte for byte; CI checks this.
- Running the pipeline on Windows rewrites `reports/*.csv` with CRLF line
  endings and re-renders the PNGs. Unless the numbers changed, discard those
  with `git restore reports` rather than committing them.
- `data/*.db` files are disposable and gitignored; `data/demo_prices.csv` is
  the frozen source of truth for the demo.
