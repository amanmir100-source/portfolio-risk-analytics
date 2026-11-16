#!/usr/bin/env python3
"""Portfolio Risk Analytics - command-line entry point.

    data (synthetic, live or snapshot) -> SQLite -> SQL analytics
    -> NumPy/pandas risk engine -> matplotlib report

Usage
-----
    python run_analysis.py                 # bundled demo dataset
    python run_analysis.py --regenerate    # rebuild the synthetic dataset
    python run_analysis.py --live          # real prices via yfinance (saves a snapshot)
    python run_analysis.py --snapshot data/snapshots/<date>   # re-run a saved pull
    python run_analysis.py --sims 50000    # heavier Monte Carlo
    python run_analysis.py --help          # every option

The work itself lives in src/portfolio_risk/pipeline.py, one function per stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

try:
    import portfolio_risk  # noqa: F401  (installed via `pip install -e .`)
except ImportError:  # fallback: run straight from the repo without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from portfolio_risk.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main(REPO_ROOT)
