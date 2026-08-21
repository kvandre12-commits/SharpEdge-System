from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from volume_weighted_rsi import build_volume_weighted_rsi


def _bar(idx: int, close: float, volume: int = 1000) -> tuple:
    open_ = close - 0.05
    return (idx, open_, close + 0.08, close - 0.08, close, volume)


def test_volume_weighted_rsi_inactive_when_volume_missing():
    bars = [_bar(idx, 100 + idx * 0.1, volume=0) for idx in range(20)]

    packet = build_volume_weighted_rsi(bars)

    assert packet["active"] is False
    assert packet["state"] == "inactive"
    assert packet["volume_quality"] == "missing"
    assert packet["advisory_only"] is True


def test_volume_weighted_rsi_confirms_upside_pressure():
    closes = [100, 100.1, 100.0, 100.2, 100.15, 100.3, 100.25, 100.4]
    closes += [100.45, 100.55, 100.7, 100.9, 101.1, 101.35, 101.6, 101.9]
    bars = [
        _bar(idx, close, volume=1000 + idx * 20) for idx, close in enumerate(closes)
    ]

    packet = build_volume_weighted_rsi(bars)

    assert packet["active"] is True
    assert packet["bias"] in {"CALLS", "NEUTRAL"}
    assert packet["value"] > 50
    assert packet["volume_quality"] == "usable"


def test_volume_weighted_rsi_flags_bullish_divergence():
    closes = [
        100.0,
        99.8,
        99.6,
        99.4,
        99.2,
        99.0,
        98.8,
        98.6,
        98.4,
        98.2,
        98.0,
        97.8,
        97.6,
        97.4,
        97.2,
        97.0,
        97.3,
        97.1,
        97.4,
        96.95,
    ]
    bars = []
    for idx, close in enumerate(closes):
        volume = 300 if idx < 16 else 2200
        bars.append(_bar(idx, close, volume=volume))

    packet = build_volume_weighted_rsi(bars, divergence_lookback=8)

    assert packet["active"] is True
    assert packet["state"] == "bullish_divergence"
    assert packet["bias"] == "CALLS"
    assert "price pressed a low" in packet["reason"]
