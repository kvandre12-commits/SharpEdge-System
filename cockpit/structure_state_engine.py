"""Pure price-structure state engine for SharpEdge.

Structure owns sequence quality only.
It does not own failed breaks, reclaims, acceptance, compression, or momentum.
"""

from __future__ import annotations

from typing import Any

import execution_vector_context as ctx

STRUCTURE_SWING_WINDOW = 2
MIN_PIVOT_SPACING_BARS = 4
MIN_SWING_AMPLITUDE_PCT = 0.12
MAX_PIVOT_AGE_BARS = 6


def _pct_change(previous: float, current: float) -> float:
    return abs(current - previous) / previous * 100 if previous else 0.0


def _sequence_flags(
    highs: list[tuple[int, float]],
    lows: list[tuple[int, float]],
) -> dict[str, bool]:
    if len(highs) < 2 or len(lows) < 2:
        return {
            "higher_high": False,
            "higher_low": False,
            "lower_high": False,
            "lower_low": False,
            "has_sequence": False,
        }
    higher_high = highs[-1][1] > highs[-2][1]
    higher_low = lows[-1][1] > lows[-2][1]
    lower_high = highs[-1][1] < highs[-2][1]
    lower_low = lows[-1][1] < lows[-2][1]
    return {
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "has_sequence": True,
    }


def _quality_packet(
    highs: list[tuple[int, float]],
    lows: list[tuple[int, float]],
    *,
    current_index: int,
) -> dict[str, Any]:
    if len(highs) < 2 or len(lows) < 2:
        return {
            "sequence_quality": "insufficient",
            "spacing_ok": False,
            "amplitude_ok": False,
            "freshness_ok": False,
            "high_spacing_bars": None,
            "low_spacing_bars": None,
            "high_amplitude_pct": None,
            "low_amplitude_pct": None,
            "latest_high_age_bars": None,
            "latest_low_age_bars": None,
            "quality_issues": [],
        }
    high_spacing_bars = int(highs[-1][0] - highs[-2][0])
    low_spacing_bars = int(lows[-1][0] - lows[-2][0])
    high_amplitude_pct = _pct_change(highs[-2][1], highs[-1][1])
    low_amplitude_pct = _pct_change(lows[-2][1], lows[-1][1])
    latest_high_age_bars = int(current_index - highs[-1][0])
    latest_low_age_bars = int(current_index - lows[-1][0])
    spacing_ok = (
        high_spacing_bars >= MIN_PIVOT_SPACING_BARS
        and low_spacing_bars >= MIN_PIVOT_SPACING_BARS
    )
    amplitude_ok = (
        high_amplitude_pct >= MIN_SWING_AMPLITUDE_PCT
        and low_amplitude_pct >= MIN_SWING_AMPLITUDE_PCT
    )
    freshness_ok = (
        latest_high_age_bars <= MAX_PIVOT_AGE_BARS
        and latest_low_age_bars <= MAX_PIVOT_AGE_BARS
    )
    quality_issues = []
    if not spacing_ok:
        quality_issues.append("pivot_spacing_tight")
    if not amplitude_ok:
        quality_issues.append("swing_amplitude_small")
    if not freshness_ok:
        quality_issues.append("pivot_freshness_stale")
    return {
        "sequence_quality": "confirmed" if not quality_issues else "weak",
        "spacing_ok": spacing_ok,
        "amplitude_ok": amplitude_ok,
        "freshness_ok": freshness_ok,
        "high_spacing_bars": high_spacing_bars,
        "low_spacing_bars": low_spacing_bars,
        "high_amplitude_pct": round(high_amplitude_pct, 4),
        "low_amplitude_pct": round(low_amplitude_pct, 4),
        "latest_high_age_bars": latest_high_age_bars,
        "latest_low_age_bars": latest_low_age_bars,
        "quality_issues": quality_issues,
    }


def _quality_reason(base_reason: str, quality: dict[str, Any]) -> str:
    issues = [
        {
            "pivot_spacing_tight": "pivot spacing is tight",
            "swing_amplitude_small": "swing amplitude is small",
            "pivot_freshness_stale": "latest pivots are getting stale",
        }[item]
        for item in (quality.get("quality_issues") or [])
    ]
    if not issues:
        return base_reason
    return f"{base_reason}, but {' and '.join(issues)}"


def build_structure_state(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    *,
    swing_window: int = STRUCTURE_SWING_WINDOW,
) -> dict[str, Any]:
    clean_bars = list(bars)
    highs, lows = ctx.swing_points(clean_bars, window=swing_window)
    flags = _sequence_flags(highs, lows)
    quality = _quality_packet(highs, lows, current_index=max(len(clean_bars) - 1, 0))
    packet = {
        "schema": "sharpedge.structure_state.v1",
        "state": "insufficient_sequence",
        "bias": "NEUTRAL",
        "reason": "not enough confirmed swing points for sequence structure",
        "swing_window": int(swing_window),
        "swing_high_count": len(highs),
        "swing_low_count": len(lows),
        "has_sequence": bool(flags["has_sequence"]),
        "higher_high": bool(flags["higher_high"]),
        "higher_low": bool(flags["higher_low"]),
        "lower_high": bool(flags["lower_high"]),
        "lower_low": bool(flags["lower_low"]),
        "latest_swing_highs": [value for _index, value in highs[-2:]],
        "latest_swing_lows": [value for _index, value in lows[-2:]],
        "latest_swing_high_indices": [index for index, _value in highs[-2:]],
        "latest_swing_low_indices": [index for index, _value in lows[-2:]],
        **quality,
    }
    if not flags["has_sequence"]:
        return packet
    if flags["higher_high"] and flags["higher_low"]:
        return {
            **packet,
            "state": "bullish_sequence",
            "bias": "CALLS",
            "reason": _quality_reason("HH/HL structure intact", quality),
        }
    if flags["lower_high"] and flags["lower_low"]:
        return {
            **packet,
            "state": "bearish_sequence",
            "bias": "PUTS",
            "reason": _quality_reason("LH/LL structure intact", quality),
        }
    return {
        **packet,
        "state": "mixed_sequence",
        "bias": "NEUTRAL",
        "reason": _quality_reason("mixed swing structure", quality),
    }


__all__ = [
    "STRUCTURE_SWING_WINDOW",
    "MIN_PIVOT_SPACING_BARS",
    "MIN_SWING_AMPLITUDE_PCT",
    "MAX_PIVOT_AGE_BARS",
    "build_structure_state",
]
