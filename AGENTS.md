# AGENTS.md — project context for AI coding assistants

## What this is

End-to-end portfolio risk analytics pipeline: market data (synthetic by default,
yfinance with `--live`) → SQLite → analytical SQL → NumPy/pandas risk engine →
matplotlib report. Stack is deliberately minimal: numpy, pandas, matplotlib,
stdlib sqlite3. No SciPy (normal quantiles come from `statistics.NormalDist`).

## Commands

```bash
pip install -e ".[dev]"            # or: pip install -r requirements-dev.txt
python run_analysis.py             # full pipeline on the bundled demo dataset (~4s)
python run_analysis.py --live      # real market data via yfinance
python run_analysis.py --regenerate --seed 40   # rebuild the demo dataset
pytest                             # 29 tests, ~2s
```

## Architecture map

- `sql/` — ALL analytical SQL lives here as standalone .sql files (schema,
  daily returns, rolling volatility, monthly performance, drawdown ranking,
  asset summary). Never inline SQL strings in Python; add a new .sql file and
  register it in `database.ANALYTICS_QUERIES`.
- `src/portfolio_risk/config.py` — asset universe, portfolio weights, regime
  parameters. Correlation matrices are order-sensitive (follow `TICKERS`).
- `src/portfolio_risk/data_generator.py` — regime-switching correlated GBM.
  `DEFAULT_SEED = 40` is what produced the committed `data/demo_prices.csv`;
  don't change the seed without regenerating data, reports and README numbers.
- `src/portfolio_risk/metrics.py` — risk/performance metrics. Conventions:
  252 trading days, 2% risk-free, VaR/CVaR reported as POSITIVE losses,
  annualised return is geometric.
- `src/portfolio_risk/monte_carlo.py`, `optimization.py` — simulation and
  closed-form Markowitz frontier (pure linear algebra, no optimiser).
- `run_analysis.py` — CLI orchestrator; writes `reports/` (demo) or
  `reports/live/` (`--live`), so a live run never overwrites the demo outputs.

## Testing rules

Every SQL query has a pytest cross-check against an independent pandas
implementation (`tests/test_database.py`). If you change a query, update its
pandas twin in the same PR — a query without a cross-check is a regression.
Everything is seeded; tests must stay deterministic.

## Gotchas

- SQLite has no `STDDEV`: rolling vol derives Bessel-corrected stdev from
  AVG(r²) − AVG(r)² inside the window; keep the `MAX(..., 0.0)` guard.
- `reports/figures/*.png` are committed (embedded in README); regenerate them
  with `python run_analysis.py` after any change that shifts the numbers.
- `data/portfolio.db` is disposable and gitignored; `data/demo_prices.csv` is
  the frozen source of truth for the demo.
