from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_state_scores import score_location_state  # noqa: E402
from location_state_engine import build_location_state  # noqa: E402
from trade_permission_context import BULLISH, NEUTRAL  # noqa: E402


def test_location_state_detects_at_reference():
    packet = build_location_state(100.0, {"ORH": 100.05, "VWAP": 99.7})

    assert packet["state"] == "at_reference"
    assert packet["nearest_reference"]["reference_name"] == "ORH"
    assert packet["reference_relations"]["ORH"] == "at_reference"


def test_location_state_detects_between_references():
    packet = build_location_state(100.0, {"PDC": 99.8, "VWAP": 100.4})

    assert packet["state"] == "between_references"
    assert packet["nearest_below_reference"]["reference_name"] == "PDC"
    assert packet["nearest_above_reference"]["reference_name"] == "VWAP"
    assert "between PDC 99.80 and VWAP 100.40" in packet["reason"]


def test_location_state_detects_above_all_references():
    packet = build_location_state(101.0, {"ORH": 100.2, "VWAP": 100.5})

    assert packet["state"] == "above_all_references"
    assert packet["bias"] == "CALLS"
    assert "above all tracked references" in packet["reason"]


def test_location_score_derives_from_pure_spatial_state():
    at_ref = score_location_state(build_location_state(100.0, {"ORH": 100.05}))
    above_all = score_location_state(
        build_location_state(101.0, {"ORH": 100.2, "VWAP": 100.5})
    )
    between = score_location_state(
        build_location_state(100.0, {"PDC": 99.8, "VWAP": 100.4})
    )
    missing = score_location_state(build_location_state(100.0, {}))

    assert at_ref.score == 82
    assert at_ref.bias == NEUTRAL
    assert above_all.score == 58
    assert above_all.bias == BULLISH
    assert between.score == 42
    assert between.bias == NEUTRAL
    assert missing.score == 34
    assert missing.bias == NEUTRAL


def test_location_state_detects_near_reference_before_directional_extremes():
    packet = build_location_state(100.0, {"PDC": 100.18, "VWAP": 99.2})

    assert packet["state"] == "near_reference"
    assert packet["bias"] == "NEUTRAL"
    assert packet["nearest_reference"]["reference_name"] == "PDC"
    assert packet["nearest_reference"]["relation"] == "below"


def test_location_state_geometry_does_not_depend_on_semantic_reference_names():
    semantic = build_location_state(100.0, {"FAILED_BREAKDOWN_RECLAIMED": 100.18})
    plain = build_location_state(100.0, {"VWAP": 100.18})

    assert semantic["state"] == plain["state"] == "near_reference"
    assert semantic["bias"] == plain["bias"] == "NEUTRAL"
    assert (
        semantic["nearest_reference"]["reference_price"]
        == plain["nearest_reference"]["reference_price"]
    )
    assert (
        semantic["nearest_reference"]["relation"]
        == plain["nearest_reference"]["relation"]
        == "below"
    )
