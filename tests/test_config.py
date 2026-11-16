import json
from pathlib import Path

import pytest

from portfolio_risk import config

PORTFOLIOS = Path(__file__).resolve().parents[1] / "portfolios"


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_file_matches_the_default_weights():
    assert config.load_portfolio(PORTFOLIOS / "model.json") == config.PORTFOLIO_WEIGHTS


@pytest.mark.parametrize("path", sorted(PORTFOLIOS.glob("*.json")), ids=lambda p: p.stem)
def test_every_shipped_portfolio_is_valid(path):
    weights = config.load_portfolio(path)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("payload, message", [
    ({"weights": {"SPY": 0.5, "QQQ": 0.4}}, "sum to"),
    ({"weights": {"SPY": 0.5, "FAKE": 0.5}}, "unknown tickers"),
    ({"weights": {"SPY": 1.2, "QQQ": -0.2}}, "non-negative"),
    ({"weights": {"SPY": "lots", "QQQ": 0.5}}, "must be numbers"),
    ({"weights": {}}, "non-empty"),
    (["SPY"], "non-empty"),
])
def test_bad_portfolio_files_are_rejected(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        config.load_portfolio(_write(tmp_path, payload))
