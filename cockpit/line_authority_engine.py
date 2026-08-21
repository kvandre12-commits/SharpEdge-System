"""Line-authority scoring for SharpEdge cockpit graph rails.

The top graph is operator visual canon. This module treats visible reference
lines as authority rails and describes how the latest candle vector interacts
with them. It is advisory-only in v1: the packet is visible to cockpit/signal,
but it does not alter permission weighting.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim
from level_interaction_facts import level_interaction_facts_for_levels

LINE_AUTHORITY_SCHEMA = "sharpedge.line_authority.v1"
LINE_AUTHORITY_SCORE_NAME = "line_authority_score"

CORE_LEVEL_NAMES = ("VWAP", "ORH", "ORL", "PDH", "PDL", "PDC")
BALANCE_LEVEL_NAMES = ("BALANCE_HIGH", "BALANCE_LOW", "BALANCE_MID", "DAY_MID")

ROLE_BY_NAME = {
    "VWAP": "value_authority",
    "ORH": "resistance",
    "PDH": "resistance",
    "ORL": "support",
    "PDL": "support",
    "PDC": "reference",
    "BALANCE_HIGH": "channel_ceiling",
    "BALANCE_LOW": "channel_floor",
    "BALANCE_MID": "channel_midpoint",
    "DAY_MID": "range_midpoint",
}


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bias_value(label: str) -> float:
    return {"CALLS": 1.0, "PUTS": -1.0}.get(label, 0.0)


def _safe_pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def build_line_authority_levels(
    pa: dict[str, Any] | None,
    levels: dict[str, Any] | None,
) -> dict[str, float]:
    """Return visible authority rails from explicit levels + graph context."""
    data = pa or {}
    source = levels or {}
    rails: dict[str, float] = {}
    for name in CORE_LEVEL_NAMES:
        value = _num(source.get(name))
        if value is not None:
            rails[name] = value
    vwap = _num(data.get("vwap"))
    if vwap is not None:
        rails["VWAP"] = vwap
    balance_high = _num(data.get("balance_high"))
    balance_low = _num(data.get("balance_low"))
    if balance_high is not None:
        rails["BALANCE_HIGH"] = balance_high
    if balance_low is not None:
        rails["BALANCE_LOW"] = balance_low
    if balance_high is not None and balance_low is not None:
        rails["BALANCE_MID"] = (balance_high + balance_low) / 2.0
    high = _num(data.get("hi"))
    low = _num(data.get("lo"))
    if high is not None and low is not None:
        rails["DAY_MID"] = (high + low) / 2.0
    return rails


def _latest_vector(bars: list[tuple[Any, ...]] | list[list[Any]]) -> dict[str, Any]:
    if not bars:
        return {
            "direction": "flat",
            "body_pct": 0.0,
            "close_position": 0.5,
            "body": 0.0,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
        }
    body, upper_wick, lower_wick, close_pos = prim.bar_personality(bars[-1])
    open_ = float(bars[-1][1])
    close = float(bars[-1][4])
    high = float(bars[-1][2])
    low = float(bars[-1][3])
    candle_range = max(high - low, 1e-9)
    direction = "up" if close > open_ else "down" if close < open_ else "flat"
    return {
        "direction": direction,
        "body_pct": round(body / candle_range, 3),
        "close_position": round(close_pos, 3),
        "body": round(body, 4),
        "upper_wick": round(upper_wick, 4),
        "lower_wick": round(lower_wick, 4),
    }


def _event_from_facts(role: str, facts: dict[str, Any]) -> tuple[str, str, int, str]:
    close_relation = str(facts.get("current_close_relation") or "unknown")
    above = int(facts.get("closes_above_count") or 0)
    below = int(facts.get("closes_below_count") or 0)
    at_level = int(facts.get("closes_at_level_count") or 0)
    window = max(int(facts.get("acceptance_window_used") or 0), 1)
    hold_above = int(facts.get("hold_above_count") or 0)
    hold_below = int(facts.get("hold_below_count") or 0)
    reclaimed = facts.get("bars_since_reclaim_above_level")
    rejected = facts.get("bars_since_reject_below_level")
    breach_above = bool(facts.get("recent_breach_above"))
    breach_below = bool(facts.get("recent_breach_below"))

    if isinstance(reclaimed, int) and reclaimed <= int(
        facts.get("recent_window_used") or 0
    ):
        return "reclaimed", "CALLS", 78, "breached below and reclaimed the rail"
    if isinstance(rejected, int) and rejected <= int(
        facts.get("recent_window_used") or 0
    ):
        return "rejected", "PUTS", 78, "breached above and rejected back under the rail"
    if above >= min(2, window):
        score = 70 + min(hold_above * 4, 12)
        label = "CALLS" if role not in {"resistance", "channel_ceiling"} else "CALLS"
        return "accepted_above", label, score, "recent closes accepted above the rail"
    if below >= min(2, window):
        score = 70 + min(hold_below * 4, 12)
        label = "PUTS" if role not in {"support", "channel_floor"} else "PUTS"
        return "accepted_below", label, score, "recent closes accepted below the rail"
    if close_relation == "at_level" or at_level:
        return "testing", "NEUTRAL", 55, "price is testing the rail buffer"
    if breach_above and close_relation == "above":
        return "breaking_above", "CALLS", 62, "price is pressing above the rail"
    if breach_below and close_relation == "below":
        return "breaking_below", "PUTS", 62, "price is pressing below the rail"
    return "nearby", "NEUTRAL", 45, "rail is visible but not controlling latest bars"


def _line_packet(
    name: str,
    facts: dict[str, Any],
    spot: float | None,
    latest_vector: dict[str, Any],
) -> dict[str, Any]:
    level = float(facts.get("level_price") or 0.0)
    role = ROLE_BY_NAME.get(name, str(facts.get("role") or "reference"))
    event, bias, score, reason = _event_from_facts(role, facts)
    distance_pct = round(_safe_pct(level - spot, spot), 3) if spot else None
    return {
        "name": name,
        "price": round(level, 4),
        "role": role,
        "distance_pct": distance_pct,
        "event": event,
        "bias": bias,
        "score": prim.clamp(score),
        "reason": reason,
        "close_relation": facts.get("current_close_relation"),
        "acceptance_counts": {
            "above": int(facts.get("closes_above_count") or 0),
            "below": int(facts.get("closes_below_count") or 0),
            "at_level": int(facts.get("closes_at_level_count") or 0),
        },
        "vector": latest_vector,
    }


def _summary(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines:
        return {
            "bias": "NEUTRAL",
            "score": 0,
            "state": "unavailable",
            "reason": "no authority rails available",
        }
    weighted = sum((line["score"] - 50) * _bias_value(line["bias"]) for line in lines)
    decisive = [line for line in lines if line["bias"] != "NEUTRAL"]
    score = prim.clamp(50 + weighted / max(len(lines), 1))
    if weighted >= 12:
        bias = "CALLS"
    elif weighted <= -12:
        bias = "PUTS"
    else:
        bias = "NEUTRAL"
    best = sorted(lines, key=lambda row: row["score"], reverse=True)[:3]
    if not decisive:
        state = "balanced"
        reason = "authority rails are mostly being tested or rotated around"
    else:
        state = (
            "aligned" if len({row["bias"] for row in decisive}) == 1 else "conflicted"
        )
        reason = "; ".join(
            f"{row['name']} {row['event']} ({row['bias']})" for row in best
        )
    return {"bias": bias, "score": score, "state": state, "reason": reason}


def build_line_authority(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    pa: dict[str, Any] | None,
    levels: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build advisory line-authority packet for cockpit graph rails."""
    rails = build_line_authority_levels(pa, levels)
    facts_by_level = level_interaction_facts_for_levels(
        bars,
        rails,
        level_names=tuple(rails.keys()),
        recent_window=6,
        acceptance_window=3,
    )
    spot = _num((pa or {}).get("spot"))
    vector = _latest_vector(bars)
    lines = [
        _line_packet(name, facts_by_level[name], spot, vector)
        for name in rails
        if name in facts_by_level
    ]
    lines = sorted(lines, key=lambda row: abs(row.get("distance_pct") or 999.0))
    summary = _summary(lines)
    return {
        "schema": LINE_AUTHORITY_SCHEMA,
        "authority_role": "operator_visual_canon_advisory",
        "weighted_in_permission": False,
        "summary": summary,
        "lines": lines,
        "score_part": {
            "name": LINE_AUTHORITY_SCORE_NAME,
            "score": summary["score"],
            "bias": summary["bias"],
            "reason": summary["reason"],
        },
    }


__all__ = [
    "LINE_AUTHORITY_SCHEMA",
    "LINE_AUTHORITY_SCORE_NAME",
    "build_line_authority",
    "build_line_authority_levels",
]
