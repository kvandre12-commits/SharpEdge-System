from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from vwap_posture import build_vwap_posture  # noqa: E402


def test_vwap_posture_distinguishes_hugging_near_and_directional_control():
    hugging = build_vwap_posture({"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.03})
    near = build_vwap_posture({"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.06})
    above = build_vwap_posture({"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.18})
    below = build_vwap_posture({"spot": 100.0, "vwap": 100.0, "vs_vwap": -0.18})

    assert hugging["state"] == "hugging_vwap"
    assert hugging["posture"] == "magnet_chop"
    assert hugging["is_range_like"] is True

    assert near["state"] == "near_vwap"
    assert near["posture"] == "wait_for_acceptance"
    assert near["is_range_like"] is True
    assert near["bias"] == "CALLS"

    assert above["state"] == "above_vwap"
    assert above["has_upside_control"] is True
    assert above["bias"] == "CALLS"

    assert below["state"] == "below_vwap"
    assert below["has_downside_control"] is True
    assert below["bias"] == "PUTS"


def test_vwap_posture_flags_stretched_extension_and_recent_acceptance():
    bars = [
        (0, 99.8, 100.0, 99.7, 99.95, 1000),
        (1, 99.95, 100.4, 99.9, 100.25, 1100),
        (2, 100.25, 100.7, 100.2, 100.55, 1200),
    ]
    packet = build_vwap_posture(
        {"spot": 100.55, "vwap": 100.0, "vs_vwap": 0.55},
        bars,
        acceptance_window=3,
        min_acceptance_closes=2,
    )

    assert packet["state"] == "stretched_above"
    assert packet["posture"] == "upside_extension"
    assert packet["is_stretched"] is True
    assert packet["accepted_above_vwap"] is True
    assert packet["accepted_below_vwap"] is False
