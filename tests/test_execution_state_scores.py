from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_state_scores import (  # noqa: E402
    score_acceptance_state,
    score_dealer_state,
    score_location_state,
    score_structure_state,
    score_time_state,
    score_trend_state,
)
from trade_permission_context import BEARISH, BULLISH, NEUTRAL  # noqa: E402


def test_score_structure_state_maps_confirmed_and_weak_sequences():
    confirmed = score_structure_state(
        {
            "state": "bullish_sequence",
            "sequence_quality": "confirmed",
            "reason": "HH/HL structure intact",
        }
    )
    weak = score_structure_state(
        {
            "state": "bearish_sequence",
            "sequence_quality": "weak",
            "reason": "LH/LL structure intact, but pivot spacing is tight",
        }
    )

    assert confirmed.score == 82
    assert confirmed.bias == BULLISH
    assert weak.score == 68
    assert weak.bias == BEARISH


def test_score_acceptance_state_maps_directional_and_neutral_states():
    accepted = score_acceptance_state(
        {
            "state": "accepted_above_level",
            "reason": "3 closes accepted above ORH 100.20",
        }
    )
    none = score_acceptance_state(
        {"state": "no_acceptance", "reason": "no clean level acceptance"}
    )
    insufficient = score_acceptance_state(
        {"state": "insufficient_data", "reason": "need 3 closes for acceptance"}
    )

    assert accepted.score == 78
    assert accepted.bias == BULLISH
    assert none.score == 35
    assert none.bias == NEUTRAL
    assert insufficient.score == 45


def test_score_trend_state_maps_alignment_conflict_and_unknown():
    aligned = score_trend_state(
        {"state": "aligned_down", "detail": "trend components aligned down"}
    )
    conflict = score_trend_state(
        {"state": "conflict", "detail": "trend components disagree"}
    )
    unknown = score_trend_state(
        {"state": "unknown", "detail": "trend inputs unavailable"}
    )

    assert aligned.score == 82
    assert aligned.bias == BEARISH
    assert conflict.score == 42
    assert conflict.bias == NEUTRAL
    assert unknown.score == 40


def test_score_time_state_maps_session_windows():
    opening = score_time_state(
        {"state": "opening", "detail": "opening auction: price discovery"}
    )
    power_hour = score_time_state(
        {"state": "power_hour", "detail": "power hour positioning window"}
    )
    closed = score_time_state(
        {"state": "closed_or_unknown", "detail": "outside regular session at 16:20"}
    )

    assert opening.score == 52
    assert opening.bias == NEUTRAL
    assert power_hour.score == 68
    assert closed.score == 40


def test_score_location_state_maps_spatial_states():
    near = score_location_state(
        {"state": "near_reference", "reason": "near VWAP 100.20 (0.12% away)"}
    )
    below_all = score_location_state(
        {"state": "below_all_references", "reason": "below all tracked references"}
    )
    missing = score_location_state(
        {
            "state": "insufficient_references",
            "reason": "no location reference map available",
        }
    )

    assert near.score == 68
    assert near.bias == NEUTRAL
    assert below_all.score == 58
    assert below_all.bias == BEARISH
    assert missing.score == 34


def test_score_dealer_state_maps_shared_dealer_states():
    gravity = score_dealer_state(
        {
            "state": "positive_gamma_gravity",
            "bias": "PUTS",
            "reason": "positive gamma pinning near call wall",
        }
    )
    expansion = score_dealer_state(
        {
            "state": "negative_gamma_expansion",
            "bias": "CALLS",
            "reason": "negative gamma supports expansion",
        }
    )
    unknown = score_dealer_state(
        {
            "state": "dealer_unknown",
            "bias": "CALLS",
            "reason": "dealer unknown: gamma data quality is weak or unknown",
        }
    )

    assert gravity.score == 38
    assert gravity.bias == BEARISH
    assert expansion.score == 72
    assert expansion.bias == BULLISH
    assert unknown.score == 40
    assert unknown.bias == NEUTRAL
