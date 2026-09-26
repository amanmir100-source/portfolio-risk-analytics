"""Publication-quality matplotlib charts for the risk report.

Every function takes prepared data plus an output path, saves a 150-dpi PNG
and returns the path. Styling is centralised in _apply_style() so all the
figures share one visual language.
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .backtest import BacktestResult
from .monte_carlo import MonteCarloResult
from .optimization import FrontierResult

PORTFOLIO_COLOR = "#14213d"
ACCENT = "#fca311"
RED = "#d62828"

_TICKER_COLORS = {
    "SPY": "#1f77b4", "QQQ": "#9467bd", "IWM": "#17becf", "EFA": "#2ca02c",
    "TLT": "#8c564b", "LQD": "#7f7f7f", "GLD": "#bcbd22", "VNQ": "#e377c2",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 6),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _dollars(x: float, _pos=None) -> str:
    return f"${x:,.0f}"


def plot_cumulative_growth(
    returns: pd.DataFrame, port_returns: pd.Series, path: str | Path,
    initial: float = 10_000.0,
) -> Path:
    """Growth of $10,000: every asset (thin) vs the portfolio (bold)."""
    _apply_style()
    fig, ax = plt.subplots()
    for ticker in returns.columns:
        wealth = initial * (1.0 + returns[ticker]).cumprod()
        ax.plot(wealth.index, wealth, lw=1.1, alpha=0.75,
                color=_TICKER_COLORS.get(ticker), label=ticker)
    port_wealth = initial * (1.0 + port_returns).cumprod()
    ax.plot(port_wealth.index, port_wealth, lw=2.8, color=PORTFOLIO_COLOR,
            label="PORTFOLIO")
    ax.yaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.set_title("Growth of $10,000 — assets vs. multi-asset portfolio")
    ax.set_ylabel("Value")
    ax.legend(ncols=3, loc="upper left")
    return _save(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: str | Path) -> Path:
    """Annotated correlation matrix of daily returns."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else "black")
    ax.grid(False)
    ax.set_title("Daily return correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _save(fig, path)


def plot_rolling_volatility(
    sql_rolling_vol: pd.DataFrame, path: str | Path,
    tickers: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD"),
) -> Path:
    """21-day annualised volatility — computed by the SQL window-function query."""
    _apply_style()
    pivot = sql_rolling_vol.pivot(index="date", columns="ticker",
                                  values="ann_volatility_21d")
    pivot.index = pd.to_datetime(pivot.index)
    fig, ax = plt.subplots()
    for ticker in tickers:
        ax.plot(pivot.index, pivot[ticker], lw=1.3,
                color=_TICKER_COLORS.get(ticker), label=ticker)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("21-day rolling volatility, annualised (computed in SQL)")
    ax.set_ylabel("Annualised volatility")
    ax.legend(ncols=4)
    return _save(fig, path)


def plot_drawdown(drawdown: pd.Series, path: str | Path) -> Path:
    """Underwater plot of the portfolio, worst point annotated."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(drawdown.index, drawdown, 0, color=RED, alpha=0.35)
    ax.plot(drawdown.index, drawdown, color=RED, lw=1.1)
    trough_date = drawdown.idxmin()
    trough = float(drawdown.min())
    ax.annotate(f"Max drawdown: {trough:.1%}",
                xy=(trough_date, trough),
                xytext=(0.60, 0.15), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Portfolio drawdown from running peak")
    ax.set_ylabel("Drawdown")
    return _save(fig, path)


def plot_return_distribution(
    port_returns: pd.Series, var_95: float, var_99: float, cvar_95: float,
    path: str | Path,
) -> Path:
    """Histogram vs fitted normal, with VaR/CVaR thresholds marked."""
    _apply_style()
    fig, ax = plt.subplots()
    r = port_returns.dropna()
    ax.hist(r, bins=70, density=True, alpha=0.55, color=PORTFOLIO_COLOR,
            label="Daily returns")
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    xs = np.linspace(r.min(), r.max(), 400)
    pdf = [NormalDist(mu, sigma).pdf(x) for x in xs]
    ax.plot(xs, pdf, color=ACCENT, lw=2, label="Fitted normal")
    for value, label, color, style in (
        (-var_95, f"95% VaR  ({var_95:.2%})", RED, "--"),
        (-var_99, f"99% VaR  ({var_99:.2%})", "#7a0c0c", ":"),
        (-cvar_95, f"95% CVaR ({cvar_95:.2%})", "black", "-."),
    ):
        ax.axvline(value, color=color, ls=style, lw=1.6, label=label)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_title("Portfolio daily return distribution — fat tails vs the normal fit")
    ax.set_ylabel("Density")
    ax.legend()
    return _save(fig, path)


def plot_monte_carlo(mc: MonteCarloResult, path: str | Path) -> Path:
    """Simulated 1-year wealth fan chart + terminal value distribution."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    bands = mc.percentile_bands
    for row in mc.sample_paths:
        ax1.plot(range(1, mc.horizon_days + 1), row, color="grey",
                 alpha=0.10, lw=0.6)
    ax1.fill_between(bands.index, bands["p5"], bands["p95"], color=PORTFOLIO_COLOR,
                     alpha=0.18, label="5th–95th percentile")
    ax1.fill_between(bands.index, bands["p25"], bands["p75"], color=PORTFOLIO_COLOR,
                     alpha=0.32, label="25th–75th percentile")
    ax1.plot(bands.index, bands["p50"], color=PORTFOLIO_COLOR, lw=2.2,
             label="Median path")
    ax1.axhline(mc.initial_value, color="black", lw=1, ls="--", alpha=0.7)
    ax1.yaxis.set_major_formatter(FuncFormatter(_dollars))
    ax1.set_title(f"Monte Carlo: {mc.n_sims:,} simulated 1-year paths")
    ax1.set_xlabel("Trading day")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(loc="upper left")

    ax2.hist(mc.terminal_values, bins=60, orientation="horizontal",
             color=PORTFOLIO_COLOR, alpha=0.6)
    p5 = mc.initial_value - mc.var_amount
    ax2.axhline(mc.initial_value, color="black", lw=1, ls="--",
                label=f"Initial ({_dollars(mc.initial_value)})")
    ax2.axhline(p5, color=RED, lw=1.6, ls="--",
                label=f"5th pct → VaR {_dollars(mc.var_amount)}")
    ax2.yaxis.set_major_formatter(FuncFormatter(_dollars))
    ax2.set_title("Terminal value distribution")
    ax2.set_xlabel("Simulations")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_efficient_frontier(fr: FrontierResult, path: str | Path) -> Path:
    """Random-portfolio cloud, analytic frontier, and the three key points."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(fr.cloud["vol"], fr.cloud["ret"], c=fr.cloud["sharpe"],
                    cmap="viridis", s=5, alpha=0.45)
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    ax.plot(fr.frontier["vol"], fr.frontier["ret"], color=PORTFOLIO_COLOR,
            lw=2.4, label="Efficient frontier (analytic)")
    for (vol, ret), label, marker, color in (
        (fr.min_var_point, "Min variance", "D", "#1f77b4"),
        (fr.max_sharpe_point, "Max Sharpe (long-only)", "*", ACCENT),
        (fr.current_point, "Model portfolio", "o", RED),
    ):
        ax.scatter([vol], [ret], marker=marker, s=220 if marker == "*" else 110,
                   color=color, edgecolor="black", zorder=5, label=label)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return")
    ax.set_title("Efficient frontier — 20,000 random portfolios vs closed-form solution")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_monthly_heatmap(port_returns: pd.Series, path: str | Path) -> Path:
    """Calendar heatmap (year x month) of portfolio monthly returns."""
    _apply_style()
    monthly = port_returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    table = (
        pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.to_numpy(),
        })
        .pivot(index="year", columns="month", values="ret")
        .reindex(columns=range(1, 13))
    )
    bound = float(np.nanmax(np.abs(table.to_numpy())))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=-bound, vmax=bound,
                   aspect="auto")
    ax.set_xticks(range(12),
                  ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(table)), table.index)
    for i in range(table.shape[0]):
        for j in range(12):
            value = table.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value * 100:.1f}", ha="center", va="center",
                        fontsize=8.5)
    ax.grid(False)
    ax.set_title("Portfolio monthly returns (%)")
    fig.colorbar(im, ax=ax, shrink=0.8,
                 format=PercentFormatter(xmax=1.0, decimals=0))
    return _save(fig, path)


def plot_var_backtest(bt: BacktestResult, path: str | Path, confidence: float = 0.99) -> Path:
    """Realised daily returns against rolling 99% VaR forecasts, breaches marked."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    r = bt.returns
    ax.plot(r.index, r, color="#9aa0a6", lw=0.6, label="Realised daily return")
    level = round(confidence * 100)
    # historical breaches as rings, parametric as dots, so a day both models
    # missed shows as a dot inside a ring
    for method, color, style, marker in (
        ("historical", PORTFOLIO_COLOR, "-",
         dict(s=70, facecolor="none", edgecolor=PORTFOLIO_COLOR, linewidth=1.4)),
        ("parametric", ACCENT, "--", dict(s=20, color=ACCENT)),
    ):
        var = bt.forecasts[f"var_{method}_{level}"]
        row = bt.table.query("method == @method and confidence == @confidence").iloc[0]
        ax.plot(var.index, -var, color=color, ls=style, lw=1.5,
                label=f"{method.capitalize()} {level}% VaR: {row['actual_breaches']} breaches "
                      f"(expected {row['expected_breaches']:.0f}, Kupiec p = {row['p_value']:.2f})")
        breaches = r[r < -var]
        ax.scatter(breaches.index, breaches, zorder=3, **marker)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"VaR backtest: {level}% forecasts from the previous {bt.window} days "
                 "vs what happened")
    ax.set_ylabel("Daily return")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=9.5)
    return _save(fig, path)
