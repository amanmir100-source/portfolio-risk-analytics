import pandas as pd
import pytest

from portfolio_risk import snapshot
from portfolio_risk.data_generator import generate_market_data


def test_snapshot_round_trip(tmp_path):
    prices, _ = generate_market_data(n_days=40, seed=1)
    history, _ = generate_market_data(n_days=90, seed=1)
    saved = snapshot.save_snapshot(tmp_path, "2026-11-16", prices, history, 0.041, "unit test")
    assert saved == tmp_path / "2026-11-16"
    snap = snapshot.load_snapshot(saved)
    assert (snap.as_of, snap.source, snap.risk_free_rate) == ("2026-11-16", "unit test", 0.041)
    pd.testing.assert_frame_equal(snap.prices, prices.reset_index(drop=True))
    assert len(snap.history) == len(history)


def test_snapshot_refuses_edited_data(tmp_path):
    prices, _ = generate_market_data(n_days=40, seed=1)
    saved = snapshot.save_snapshot(tmp_path, "2026-11-16", prices, prices, 0.04, "unit test")
    edited = prices.copy()
    edited.loc[0, "close"] += 1.0
    edited.to_csv(saved / "prices.csv", index=False)
    with pytest.raises(ValueError, match="changed after the snapshot"):
        snapshot.load_snapshot(saved)
