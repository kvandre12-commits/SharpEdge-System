"""Trend alignment state engine for SharpEdge.

Trend does not predict direction in isolation.
It answers whether its components align:
- short-horizon path
- VWAP relationship
- momentum
"""

from __future__ import annotations

from typing import Any

RECENT_WINDOW = 6
SLOPE_FLAT_PCT = 0.03
VWAP_FLAT_PCT = 0.05
MOMENTUM_FLAT_PCT = 0.05


def _pct_change(start: float, end: float) -> float:
    return (end - start) / start * 100 if start else 0.0


def _component_direction(value: float, threshold: float) -> str:
    if value >= threshold:
        return "up"
    if value <= -threshold:
        return "down"
    return "neutral"


def _component_reason_map(
    slope_dir: str, vwap_dir: str, momentum_dir: str
) -> dict[str, str]:
    return {
        "slope": {
            "up": "path_up",
            "down": "path_down",
            "neutral": "path_flat",
        }[slope_dir],
        "vwap": {
            "up": "above_vwap",
            "down": "below_vwap",
            "neutral": "vwap_chop",
        }[vwap_dir],
        "momentum": {
            "up": "positive_momentum",
            "down": "negative_momentum",
            "neutral": "momentum_flat",
        }[momentum_dir],
    }


def build_trend_state(
    bars: list[tuple] | None,
    pa: dict[str, Any] | None,
) -> dict[str, Any]:
    packet = {
        "schema": "sharpedge.trend_state.v1",
        "state": "insufficient",
        "bias": "NEUTRAL",
        "reason": "insufficient_bars",
        "detail": "need at least 6 bars for trend alignment",
        "component_states": {},
        "component_reasons": {},
        "up_count": 0,
        "down_count": 0,
        "neutral_count": 0,
        "slope_pct": None,
        "vs_vwap": None,
        "mom15": None,
    }
    clean_bars = bars or []
    if len(clean_bars) < RECENT_WINDOW:
        return packet

    data = pa or {}
    vs_vwap = data.get("vs_vwap")
    mom15 = data.get("mom15")
    if not isinstance(vs_vwap, (int, float)) or not isinstance(mom15, (int, float)):
        missing = []
        if not isinstance(vs_vwap, (int, float)):
            missing.append("vs_vwap")
        if not isinstance(mom15, (int, float)):
            missing.append("mom15")
        return {
            **packet,
            "state": "unknown",
            "reason": "missing_inputs",
            "detail": f"trend inputs unavailable: {', '.join(missing)} missing",
        }

    closes = [float(bar[4]) for bar in clean_bars[-RECENT_WINDOW:]]
    slope_pct = _pct_change(closes[0], closes[-1])
    slope_dir = _component_direction(slope_pct, SLOPE_FLAT_PCT)
    vwap_dir = _component_direction(float(vs_vwap), VWAP_FLAT_PCT)
    momentum_dir = _component_direction(float(mom15), MOMENTUM_FLAT_PCT)
    component_states = {
        "slope": slope_dir,
        "vwap": vwap_dir,
        "momentum": momentum_dir,
    }
    component_reasons = _component_reason_map(slope_dir, vwap_dir, momentum_dir)
    up_count = sum(direction == "up" for direction in component_states.values())
    down_count = sum(direction == "down" for direction in component_states.values())
    neutral_count = sum(
        direction == "neutral" for direction in component_states.values()
    )
    base = {
        **packet,
        "component_states": component_states,
        "component_reasons": component_reasons,
        "up_count": up_count,
        "down_count": down_count,
        "neutral_count": neutral_count,
        "slope_pct": round(slope_pct, 4),
        "vs_vwap": round(float(vs_vwap), 4),
        "mom15": round(float(mom15), 4),
    }
    if up_count >= 2 and down_count == 0:
        detail = "trend components aligned up"
        if vwap_dir == "neutral":
            detail += "; vwap_chop but path and momentum are leaning higher"
        return {
            **base,
            "state": "aligned_up",
            "bias": "CALLS",
            "reason": "full_alignment" if neutral_count == 0 else "vwap_chop",
            "detail": detail,
        }
    if down_count >= 2 and up_count == 0:
        detail = "trend components aligned down"
        if vwap_dir == "neutral":
            detail += "; vwap_chop but path and momentum are leaning lower"
        return {
            **base,
            "state": "aligned_down",
            "bias": "PUTS",
            "reason": "full_alignment" if neutral_count == 0 else "vwap_chop",
            "detail": detail,
        }
    if up_count > 0 and down_count > 0:
        if vwap_dir == "neutral":
            reason = "vwap_chop"
            detail = "trend components conflict around VWAP"
        elif slope_dir == momentum_dir != "neutral" and vwap_dir != slope_dir:
            reason = "vwap_rotation"
            detail = "path and momentum disagree with VWAP relationship"
        else:
            reason = "component_conflict"
            detail = "trend components disagree"
        return {
            **base,
            "state": "conflict",
            "reason": reason,
            "detail": detail,
        }
    if up_count == 0 and down_count == 0:
        return {
            **base,
            "state": "neutral",
            "reason": "vwap_chop",
            "detail": "trend is neutral; components are rotating around VWAP",
        }
    return {
        **base,
        "state": "neutral",
        "reason": "vwap_rotation" if vwap_dir == "neutral" else "weak_alignment",
        "detail": "trend alignment is incomplete",
    }


__all__ = ["build_trend_state"]
