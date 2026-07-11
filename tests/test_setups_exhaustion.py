from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from setups import detect_exhaustion  # noqa: E402


FLAT_BARS = [
    (0, 100.0, 100.05, 99.95, 100.0, 1000),
    (1, 100.0, 100.05, 99.95, 100.0, 1000),
    (2, 100.0, 100.05, 99.95, 100.0, 1000),
    (3, 100.0, 100.05, 99.95, 100.0, 1000),
    (4, 100.0, 100.05, 99.95, 100.0, 1000),
    (5, 100.0, 100.05, 99.95, 100.0, 1000),
    (6, 100.0, 100.05, 99.95, 100.0, 1000),
    (7, 100.0, 100.05, 99.95, 100.0, 1000),
    (8, 100.0, 100.05, 99.95, 100.0, 1000),
]


def _downside_bars() -> list[tuple[int, float, float, float, float, int]]:
    return [*FLAT_BARS, (9, 100.0, 100.05, 99.0, 100.0, 1000)]


def _upside_bars() -> list[tuple[int, float, float, float, float, int]]:
    return [*FLAT_BARS, (9, 100.0, 101.0, 99.95, 100.0, 1000)]


def test_detect_exhaustion_flags_downside_pressing_edge_with_strict_stretch():
    cards = detect_exhaustion(
        _downside_bars(),
        {"spot": 100.0, "vwap": 100.41, "vs_vwap": -0.41, "rng_pos": 22.0},
    )

    assert len(cards) == 1
    assert cards[0]["tag"] == "DOWNSIDE EXHAUSTION"
    assert "at day lows" in cards[0]["detail"]
    assert "stretched -0.41% from VWAP" in cards[0]["detail"]


def test_detect_exhaustion_keeps_setup_owned_stricter_than_canonical_stretch():
    cards = detect_exhaustion(
        _downside_bars(),
        {"spot": 100.0, "vwap": 100.37, "vs_vwap": -0.37, "rng_pos": 22.0},
    )

    assert cards == []


def test_detect_exhaustion_flags_upside_pressing_edge_from_canonical_posture():
    cards = detect_exhaustion(
        _upside_bars(),
        {"spot": 100.0, "vwap": 99.59, "vs_vwap": 0.41, "rng_pos": 78.0},
    )

    assert len(cards) == 1
    assert cards[0]["tag"] == "UPSIDE EXHAUSTION"
    assert "at day highs" in cards[0]["detail"]
