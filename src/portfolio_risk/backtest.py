"""Out-of-sample VaR backtest: Kupiec coverage + Christoffersen independence.

Each day's VaR forecast uses only the previous `window` days. A breach is a
day whose return falls below -VaR. Two questions are tested:

* Coverage (Kupiec POF): is the NUMBER of breaches plausible for the
  confidence level? At 99% over ~1,000 days we expect ~10.
* Independence (Christoffersen): do breaches arrive one at a time, or in
  clusters? A model can have the right count and still fail here, which is
  what happens when volatility jumps and the VaR is slow to catch up.

Both statistics are chi-squared(1), so each p-value is erfc(sqrt(LR / 2)).
Their sum is the conditional-coverage statistic, chi-squared(2), whose
p-value is exp(-LR / 2). All from the standard library - no SciPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 250  # one trading year, as in Basel
METHODS = ("historical", "parametric")


def rolling_var_forecasts(
    r: pd.Series,
    window: int = DEFAULT_WINDOW,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Daily VaR forecasts, each from the `window` days before it.

    Same numbers as metrics.historical_var / parametric_var on the trailing
    window, just vectorised. shift(1) is what stops look-ahead.
    """
    r = pd.Series(r).dropna()
    if len(r) <= window:
        raise ValueError(f"need more than {window} returns to backtest, got {len(r)}")
    rolling = r.rolling(window)
    if method == "historical":
        var = -rolling.quantile(1.0 - confidence, interpolation="linear")
    elif method == "parametric":
        z = NormalDist().inv_cdf(1.0 - confidence)  # negative quantile
        var = -(rolling.mean() + rolling.std(ddof=1) * z)
    else:
        raise ValueError(f"unknown VaR method {method!r}; expected one of {METHODS}")
    return var.shift(1).iloc[window:].rename(f"var_{method}_{round(confidence * 100)}")


def _log_likelihood(n_obs: int, n_breaches: int, prob: float) -> float:
    """Binomial log-likelihood of the breach count, using 0 * log(0) = 0."""
    ll = 0.0
    if n_obs - n_breaches:
        ll += (n_obs - n_breaches) * math.log(1.0 - prob)
    if n_breaches:
        ll += n_breaches * math.log(prob)
    return ll


def kupiec_pof(n_obs: int, n_breaches: int, confidence: float) -> tuple[float, float]:
    """Kupiec POF test. Returns (LR statistic, p-value); low p = reject."""
    if not 0 <= n_breaches <= n_obs or n_obs == 0:
        raise ValueError("need 0 <= n_breaches <= n_obs and n_obs > 0")
    p = 1.0 - confidence
    observed = n_breaches / n_obs
    lr = -2.0 * (_log_likelihood(n_obs, n_breaches, p)
                 - _log_likelihood(n_obs, n_breaches, observed))
    lr = max(lr, 0.0)  # float rounding can give -1e-16
    return lr, math.erfc(math.sqrt(lr / 2.0))


def christoffersen_independence(breaches) -> tuple[float, float]:
    """Christoffersen's independence test. Returns (LR statistic, p-value).

    Compares a first-order Markov model, where the chance of a breach depends
    on whether yesterday was one, against a model where it does not. A low
    p-value means breaches cluster. With no breaches at all, or no pair of
    consecutive days to compare, there is no evidence of clustering: (0, 1).
    """
    b = np.asarray(breaches, dtype=bool)
    if b.size < 2 or not b.any():
        return 0.0, 1.0
    prev, cur = b[:-1], b[1:]
    n00 = int((~prev & ~cur).sum())
    n01 = int((~prev & cur).sum())
    n10 = int((prev & ~cur).sum())
    n11 = int((prev & cur).sum())
    after_calm, after_breach = n00 + n01, n10 + n11
    pooled = (n01 + n11) / (after_calm + after_breach)
    ll_independent = _log_likelihood(after_calm + after_breach, n01 + n11, pooled)
    ll_markov = 0.0
    if after_calm:
        ll_markov += _log_likelihood(after_calm, n01, n01 / after_calm)
    if after_breach:
        ll_markov += _log_likelihood(after_breach, n11, n11 / after_breach)
    lr = max(-2.0 * (ll_independent - ll_markov), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2.0))


def conditional_coverage_p_value(lr_coverage: float, lr_independence: float) -> float:
    """p-value of the joint test (right count AND no clustering), chi-squared(2)."""
    return math.exp(-(lr_coverage + lr_independence) / 2.0)


@dataclass
class BacktestResult:
    returns: pd.Series
    forecasts: pd.DataFrame  # one column per method/confidence
    table: pd.DataFrame      # one row per method/confidence
    window: int


def backtest_var(
    r: pd.Series,
    window: int = DEFAULT_WINDOW,
    confidences: tuple[float, ...] = (0.95, 0.99),
    methods: tuple[str, ...] = METHODS,
    significance: float = 0.05,
) -> BacktestResult:
    """Backtest each VaR method at each confidence level on the series ``r``."""
    r = pd.Series(r).dropna()
    forecasts, rows = {}, []
    for method in methods:
        for conf in confidences:
            var = rolling_var_forecasts(r, window, conf, method)
            realised = r.loc[var.index]
            n_obs = len(var)
            breaches = (realised < -var).to_numpy()
            n_breaches = int(breaches.sum())
            lr, p_value = kupiec_pof(n_obs, n_breaches, conf)
            lr_ind, p_ind = christoffersen_independence(breaches)
            forecasts[var.name] = var
            rows.append({
                "method": method,
                "confidence": conf,
                "test_days": n_obs,
                "expected_breaches": round(n_obs * (1.0 - conf), 1),
                "actual_breaches": n_breaches,
                "breach_rate": n_breaches / n_obs,
                "kupiec_lr": lr,
                "p_value": p_value,
                "result": "pass" if p_value >= significance else "reject",
                "independence_lr": lr_ind,
                "independence_p": p_ind,
                "independence_result": "pass" if p_ind >= significance else "reject",
                "conditional_coverage_p": conditional_coverage_p_value(lr, lr_ind),
            })
    frame = pd.DataFrame(forecasts)
    return BacktestResult(
        returns=r.loc[frame.index],
        forecasts=frame,
        table=pd.DataFrame(rows),
        window=window,
    )
