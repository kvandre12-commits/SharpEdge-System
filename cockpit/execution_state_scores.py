"""Shared state-packet to score adapters for SharpEdge execution vectors.

These adapters translate canonical state packets into score doctrine so the
cockpit does not duplicate state interpretation across authority surfaces.
"""

from __future__ import annotations

from typing import Any

from trade_permission_context import BEARISH, BULLISH, NEUTRAL, ScorePart

DEALER_STATE_SCORE_MAP = {
    "positive_gamma_gravity": 38,
    "positive_gamma_context": 48,
    "negative_gamma_expansion": 72,
}

LOCATION_STATE_SCORE_MAP = {
    "at_reference": (82, NEUTRAL),
    "near_reference": (68, NEUTRAL),
    "above_all_references": (58, BULLISH),
    "below_all_references": (58, BEARISH),
    "between_references": (42, NEUTRAL),
}

TIME_STATE_SCORE_MAP = {
    "opening": 52,
    "morning": 74,
    "midday": 42,
    "power_hour": 68,
    "afternoon": 58,
}

TREND_STATE_SCORE_MAP = {
    "aligned_up": (82, BULLISH),
    "aligned_down": (82, BEARISH),
    "conflict": (42, NEUTRAL),
    "neutral": (50, NEUTRAL),
    "insufficient": (45, NEUTRAL),
}


def label_to_bias(label: str) -> int:
    return {"CALLS": BULLISH, "PUTS": BEARISH}.get(str(label).upper(), NEUTRAL)


def score_structure_state(structure_state: dict[str, Any] | None) -> ScorePart:
    structure = structure_state or {}
    state = str(structure.get("state") or "insufficient_sequence")
    reason = str(structure.get("reason") or "structure unavailable")
    quality = str(structure.get("sequence_quality") or "insufficient")
    if state == "bullish_sequence":
        return ScorePart(82 if quality == "confirmed" else 68, BULLISH, reason)
    if state == "bearish_sequence":
        return ScorePart(82 if quality == "confirmed" else 68, BEARISH, reason)
    if state == "mixed_sequence":
        return ScorePart(46, NEUTRAL, reason)
    return ScorePart(40, NEUTRAL, reason)


def score_acceptance_state(acceptance_state: dict[str, Any] | None) -> ScorePart:
    acceptance = acceptance_state or {}
    state = str(acceptance.get("state") or "insufficient_data")
    reason = str(acceptance.get("reason") or "acceptance unavailable")
    if state == "accepted_above_level":
        return ScorePart(78, BULLISH, reason)
    if state == "accepted_below_level":
        return ScorePart(78, BEARISH, reason)
    if state == "insufficient_data":
        return ScorePart(45, NEUTRAL, reason)
    return ScorePart(35, NEUTRAL, reason)


def score_trend_state(trend_state: dict[str, Any] | None) -> ScorePart:
    trend = trend_state or {}
    state = str(trend.get("state") or "insufficient")
    reason = str(
        trend.get("detail") or trend.get("reason") or "trend alignment unavailable"
    )
    score_and_bias = TREND_STATE_SCORE_MAP.get(state)
    if score_and_bias is None:
        return ScorePart(40, NEUTRAL, reason)
    score, bias = score_and_bias
    return ScorePart(score, bias, reason)


def score_time_state(time_state: dict[str, Any] | None) -> ScorePart:
    time_state = time_state or {}
    state = str(time_state.get("state") or "closed_or_unknown")
    reason = str(
        time_state.get("detail")
        or time_state.get("reason")
        or "time context unavailable"
    )
    return ScorePart(TIME_STATE_SCORE_MAP.get(state, 40), NEUTRAL, reason)


def score_location_state(location_state: dict[str, Any] | None) -> ScorePart:
    location = location_state or {}
    state = str(location.get("state") or "insufficient_references")
    reason = str(location.get("reason") or "location unavailable")
    score_and_bias = LOCATION_STATE_SCORE_MAP.get(state)
    if score_and_bias is None:
        return ScorePart(34, NEUTRAL, reason)
    score, bias = score_and_bias
    return ScorePart(score, bias, reason)


def score_dealer_state(dealer_state: dict[str, Any] | None) -> ScorePart:
    dealer = dealer_state or {}
    state = str(dealer.get("state") or "dealer_unknown")
    reason = str(dealer.get("reason") or "dealer state unavailable")
    score = DEALER_STATE_SCORE_MAP.get(state)
    if score is None:
        return ScorePart(40, NEUTRAL, reason)
    bias_label = str(dealer.get("bias") or "NEUTRAL")
    return ScorePart(score, label_to_bias(bias_label), reason)


__all__ = [
    "DEALER_STATE_SCORE_MAP",
    "LOCATION_STATE_SCORE_MAP",
    "TIME_STATE_SCORE_MAP",
    "TREND_STATE_SCORE_MAP",
    "label_to_bias",
    "score_acceptance_state",
    "score_dealer_state",
    "score_location_state",
    "score_structure_state",
    "score_time_state",
    "score_trend_state",
]
