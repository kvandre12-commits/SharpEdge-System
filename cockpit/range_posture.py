"""Canonical range/extreme/stretch posture for SharpEdge."""

from __future__ import annotations

from typing import Any

from vwap_posture import build_vwap_posture

BALANCED_LOWER_PCT = 35.0
BALANCED_UPPER_PCT = 65.0
EDGE_LOWER_PCT = 22.0
EDGE_UPPER_PCT = 78.0
EXTREME_LOWER_PCT = 18.0
EXTREME_UPPER_PCT = 82.0
TERMINAL_LOWER_PCT = 12.0
TERMINAL_UPPER_PCT = 88.0
EMERGING_DISPLACEMENT_MIN_PCT = 0.05
EMERGING_DISPLACEMENT_MAX_PCT = 0.18
EXTENDED_FROM_VALUE_PCT = 0.28
TREND_TAIL_DISPLACEMENT_PCT = 0.30
STRETCHED_FROM_VALUE_PCT = 0.35


def _range_state(rng_pos: float) -> str:
    if rng_pos >= TERMINAL_UPPER_PCT:
        return "terminal_high"
    if rng_pos >= EXTREME_UPPER_PCT:
        return "extreme_high"
    if rng_pos >= EDGE_UPPER_PCT:
        return "edge_high"
    if rng_pos >= BALANCED_UPPER_PCT:
        return "upper_range"
    if rng_pos <= TERMINAL_LOWER_PCT:
        return "terminal_low"
    if rng_pos <= EXTREME_LOWER_PCT:
        return "extreme_low"
    if rng_pos <= EDGE_LOWER_PCT:
        return "edge_low"
    if rng_pos <= BALANCED_LOWER_PCT:
        return "lower_range"
    return "balanced_middle"


def _displacement_state(abs_vs_vwap_pct: float, vwap_posture: dict[str, Any]) -> str:
    if bool(vwap_posture.get("is_range_like")):
        return "near_value"
    if abs_vs_vwap_pct >= STRETCHED_FROM_VALUE_PCT:
        return "stretched_from_value"
    if abs_vs_vwap_pct >= EXTENDED_FROM_VALUE_PCT:
        return "extended_from_value"
    if EMERGING_DISPLACEMENT_MIN_PCT < abs_vs_vwap_pct <= EMERGING_DISPLACEMENT_MAX_PCT:
        return "emerging_from_value"
    return "accepted_away_from_value"


def build_range_posture(
    pa: dict[str, Any] | None,
    *,
    vwap_posture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = pa or {}
    rng_pos = float(data.get("rng_pos") or 50.0)
    vwap = vwap_posture or build_vwap_posture(data)
    abs_vs_vwap_pct = float(vwap.get("abs_vs_vwap_pct") or 0.0)
    range_state = _range_state(rng_pos)
    displacement_state = _displacement_state(abs_vs_vwap_pct, vwap)
    side = "upside" if rng_pos > 50 else "downside" if rng_pos < 50 else "center"
    return {
        "schema": "sharpedge.range_posture.v1",
        "range_position_pct": round(rng_pos, 2),
        "range_state": range_state,
        "side": side,
        "displacement_state": displacement_state,
        "vwap_state": vwap.get("state"),
        "vwap_posture": vwap.get("posture"),
        "abs_vs_vwap_pct": round(abs_vs_vwap_pct, 3),
        "is_upper_half": rng_pos >= BALANCED_UPPER_PCT,
        "is_lower_half": rng_pos <= BALANCED_LOWER_PCT,
        "is_pressing_edge": range_state
        in {
            "edge_high",
            "edge_low",
            "extreme_high",
            "extreme_low",
            "terminal_high",
            "terminal_low",
        },
        "is_extreme": range_state
        in {"extreme_high", "extreme_low", "terminal_high", "terminal_low"},
        "is_terminal_extreme": range_state in {"terminal_high", "terminal_low"},
        "is_near_value": bool(vwap.get("is_range_like")),
        "has_directional_displacement": not bool(vwap.get("is_range_like")),
        "is_emerging_from_value": displacement_state == "emerging_from_value",
        "is_extended_from_value": abs_vs_vwap_pct >= EXTENDED_FROM_VALUE_PCT,
        "is_trend_tail_displacement": abs_vs_vwap_pct >= TREND_TAIL_DISPLACEMENT_PCT,
        "is_stretched_from_value": abs_vs_vwap_pct >= STRETCHED_FROM_VALUE_PCT,
        "reason": f"range {rng_pos:.0f}% -> {range_state}; VWAP displacement {abs_vs_vwap_pct:.2f}% -> {displacement_state}",
    }


__all__ = [
    "BALANCED_LOWER_PCT",
    "BALANCED_UPPER_PCT",
    "EDGE_LOWER_PCT",
    "EDGE_UPPER_PCT",
    "EMERGING_DISPLACEMENT_MAX_PCT",
    "EMERGING_DISPLACEMENT_MIN_PCT",
    "EXTENDED_FROM_VALUE_PCT",
    "EXTREME_LOWER_PCT",
    "EXTREME_UPPER_PCT",
    "STRETCHED_FROM_VALUE_PCT",
    "TERMINAL_LOWER_PCT",
    "TERMINAL_UPPER_PCT",
    "TREND_TAIL_DISPLACEMENT_PCT",
    "build_range_posture",
]
