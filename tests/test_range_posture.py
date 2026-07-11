from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from range_posture import build_range_posture  # noqa: E402


def test_range_posture_classifies_balanced_near_value_and_emerging_displacement():
    near_value = build_range_posture(
        {"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.06, "rng_pos": 52.0}
    )
    emerging = build_range_posture(
        {"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.12, "rng_pos": 67.0}
    )

    assert near_value["range_state"] == "balanced_middle"
    assert near_value["is_near_value"] is True
    assert near_value["displacement_state"] == "near_value"

    assert emerging["range_state"] == "upper_range"
    assert emerging["is_upper_half"] is True
    assert emerging["is_emerging_from_value"] is True
    assert emerging["displacement_state"] == "emerging_from_value"


def test_range_posture_classifies_terminal_extreme_and_stretch_from_value():
    packet = build_range_posture(
        {"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.37, "rng_pos": 91.0}
    )

    assert packet["range_state"] == "terminal_high"
    assert packet["is_terminal_extreme"] is True
    assert packet["is_extreme"] is True
    assert packet["is_stretched_from_value"] is True
    assert packet["displacement_state"] == "stretched_from_value"
