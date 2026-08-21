from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from fair_value_gap import build_fair_value_gap_map


def test_bullish_fvg_stays_open_until_price_trades_back_through_it():
    bars = [
        (0, 100.0, 100.2, 99.8, 100.1, 1000),
        (1, 100.1, 100.75, 100.0, 100.7, 1100),
        (2, 100.8, 101.0, 100.7, 100.9, 1200),
        (3, 100.9, 101.1, 100.74, 101.0, 1300),
    ]

    packet = build_fair_value_gap_map(bars, spot=101.0)

    assert packet["gap_count"] == 1
    assert packet["open_gap_count"] == 1
    assert packet["latest_bullish_gap"] == {
        "direction": "bullish",
        "start_index": 0,
        "created_index": 2,
        "minute": 2,
        "gap_low": 100.2,
        "gap_high": 100.7,
        "midpoint": 100.45,
        "size": 0.5,
        "size_pct": 0.498,
        "fill_state": "open",
        "fill_pct": 0.0,
        "age_bars": 1,
        "distance_from_spot": 0.55,
        "position_vs_spot": "below",
        "fill_direction": "down",
    }
    assert packet["nearest_open_gap_below"]["direction"] == "bullish"


def test_bearish_fvg_can_be_partially_filled_without_full_closeout():
    bars = [
        (0, 100.8, 101.0, 100.6, 100.9, 1000),
        (1, 100.9, 101.1, 100.45, 100.8, 1100),
        (2, 100.2, 100.3, 100.0, 100.1, 1200),
        (3, 100.1, 100.5, 100.05, 100.4, 1300),
    ]

    packet = build_fair_value_gap_map(bars, spot=100.6)
    gap = packet["latest_bearish_gap"]

    assert gap["direction"] == "bearish"
    assert gap["gap_low"] == 100.3
    assert gap["gap_high"] == 100.6
    assert gap["fill_state"] == "partial"
    assert gap["fill_pct"] == 66.7
    assert packet["open_gap_count"] == 1
