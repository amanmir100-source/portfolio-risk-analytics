"""The live loader, tested against a stand-in for yfinance (no network)."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import config, live_data


def _install_fake_yfinance(monkeypatch, download):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))


def _price_frame(tickers, n=6):
    dates = pd.bdate_range("2026-01-05", periods=n, name="Date")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], tickers], names=["Price", "Ticker"])
    data = np.arange(1, n * len(columns) + 1, dtype=float).reshape(n, len(columns))
    return pd.DataFrame(data, index=dates, columns=columns)


def test_fetch_live_prices_returns_the_generator_schema(monkeypatch):
    calls = {}

    def download(**kwargs):
        calls.update(kwargs)
        frame = _price_frame(config.TICKERS)
        frame.loc[frame.index[2], ("Close", "EFA")] = np.nan  # a gap in the feed
        return frame

    _install_fake_yfinance(monkeypatch, download)
    prices = live_data.fetch_live_prices()
    assert list(prices.columns) == ["date", "ticker", "close", "volume"]
    assert set(prices["ticker"]) == set(config.TICKERS)
    assert len(prices) == 6 * len(config.TICKERS) - 1  # the missing close is dropped
    assert prices["date"].iloc[0] == "2026-01-05"
    assert str(prices["volume"].dtype) == "int64"
    assert calls["period"] == "5y" and calls["auto_adjust"] is True


def test_fetch_live_prices_passes_a_start_date_for_long_history(monkeypatch):
    calls = {}

    def download(**kwargs):
        calls.update(kwargs)
        return _price_frame(["SPY", "TLT"])

    _install_fake_yfinance(monkeypatch, download)
    live_data.fetch_live_prices(["SPY", "TLT"], start="2007-01-01")
    assert calls["start"] == "2007-01-01" and "period" not in calls


def test_empty_download_is_an_error(monkeypatch):
    _install_fake_yfinance(monkeypatch, lambda **kwargs: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        live_data.fetch_live_prices()
    with pytest.raises(RuntimeError, match="--risk-free"):
        live_data.fetch_risk_free_rate("2026-01-01", "2026-02-01")


def test_risk_free_rate_is_the_window_average_as_a_fraction(monkeypatch):
    calls = {}

    def download(**kwargs):
        calls.update(kwargs)
        dates = pd.bdate_range("2026-01-05", periods=4, name="Date")
        columns = pd.MultiIndex.from_product([["Close"], ["^IRX"]])
        return pd.DataFrame([[4.0], [4.2], [np.nan], [4.4]], index=dates, columns=columns)

    _install_fake_yfinance(monkeypatch, download)
    rate = live_data.fetch_risk_free_rate("2026-01-05", "2026-01-08")
    assert rate == pytest.approx(0.042)  # quoted in percent, NaN ignored
    assert calls["end"] == "2026-01-09"  # yfinance's end date is exclusive


def test_implausible_risk_free_rate_is_rejected(monkeypatch):
    frame = pd.DataFrame({"Close": [420.0, 421.0]},
                         index=pd.bdate_range("2026-01-05", periods=2, name="Date"))
    _install_fake_yfinance(monkeypatch, lambda **kwargs: frame)
    with pytest.raises(RuntimeError, match="implausible"):
        live_data.fetch_risk_free_rate("2026-01-05", "2026-01-06")
