"""Deterministic level-state engine for SharpEdge.

This module sits between raw reference levels and higher-order setup/authority
layers. It classifies what state each level is in right now without emitting
permission, setup conviction, or trade authority.
"""

from __future__ import annotations

from typing import Any

from failed_break_facts import (
    RESISTANCE_LEVEL_NAMES,
    SUPPORT_LEVEL_NAMES,
    active_failed_break_levels,
    failed_break_facts_for_levels,
)

LEVEL_STATE_LEVEL_NAMES = ("ORH", "ORL", "PDH", "PDL", "PDC")


def _level_role(level_name: str) -> str:
    if level_name in SUPPORT_LEVEL_NAMES:
        return "support"
    if level_name in RESISTANCE_LEVEL_NAMES:
        return "resistance"
    return "reference"


def _close_relation(current_close: float | None, level_price: float, buffer: float) -> str:
    if not isinstance(current_close, (int, float)):
        return "unknown"
    if current_close > level_price + buffer:
        return "above"
    if current_close < level_price - buffer:
        return "below"
    return "at_level"


def _acceptance_state(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    buffer: float,
    *,
    acceptance_window: int = 3,
) -> dict[str, Any]:
    recent = list(bars[-max(int(acceptance_window or 0), 1) :])
    closes = [float(bar[4]) for bar in recent] if recent else []
    above = sum(1 for close in closes if close > level_price + buffer)
    below = sum(1 for close in closes if close < level_price - buffer)
    needed = min(2, len(closes))
    if needed and above >= needed:
        state = "accepted_above"
    elif needed and below >= needed:
        state = "accepted_below"
    else:
        state = "mixed"
    return {
        "window": len(closes),
        "above_count": above,
        "below_count": below,
        "state": state,
    }


def _event_state(role: str, facts: dict[str, Any], close_relation: str, acceptance: str) -> str:
    if role == "support":
        reclaim_age = facts.get("bars_since_reclaim_above_level")
        if isinstance(reclaim_age, int) and reclaim_age <= int(
            facts.get("recent_window_used") or 0
        ):
            return "failed_break_reclaimed"
        if close_relation == "below" and facts.get("recent_breach_below"):
            return "lost_support"
        if close_relation == "at_level":
            return "testing_support"
        if acceptance == "accepted_above":
            return "holding_above_support"
        return "support_in_play"

    if role == "resistance":
        reject_age = facts.get("bars_since_reject_below_level")
        if isinstance(reject_age, int) and reject_age <= int(
            facts.get("recent_window_used") or 0
        ):
            return "failed_break_rejected"
        if close_relation == "above" and facts.get("recent_breach_above"):
            return "accepted_above_resistance"
        if close_relation == "at_level":
            return "testing_resistance"
        if acceptance == "accepted_below":
            return "holding_below_resistance"
        return "resistance_in_play"

    if close_relation == "above" and acceptance == "accepted_above":
        return "accepted_above_reference"
    if close_relation == "below" and acceptance == "accepted_below":
        return "accepted_below_reference"
    if close_relation == "at_level":
        return "testing_reference"
    return "reference_in_play"


def _failed_break_candidate(role: str, event_state: str) -> str | None:
    if role == "support" and event_state == "failed_break_reclaimed":
        return "FAILED BREAKDOWN"
    if role == "resistance" and event_state == "failed_break_rejected":
        return "FAILED BREAKOUT"
    return None


def _summary(level_name: str, level_price: float, event_state: str, acceptance: str) -> str:
    label = f"{level_name} ${level_price:.2f}"
    summaries = {
        "failed_break_reclaimed": f"{label} broke down and was reclaimed; failed-break long candidate remains live",
        "lost_support": f"{label} is below and acting as lost support",
        "testing_support": f"{label} is being tested from nearby support",
        "holding_above_support": f"{label} is holding as support with recent acceptance above",
        "failed_break_rejected": f"{label} broke out and was rejected; failed-break short candidate remains live",
        "accepted_above_resistance": f"{label} has been exceeded and is no longer clean resistance",
        "testing_resistance": f"{label} is being tested from nearby resistance",
        "holding_below_resistance": f"{label} is holding as resistance with recent acceptance below",
        "accepted_above_reference": f"{label} is accepted above on recent closes",
        "accepted_below_reference": f"{label} is accepted below on recent closes",
        "testing_reference": f"{label} is being tested in the buffer",
    }
    return summaries.get(
        event_state,
        f"{label} remains in play with {acceptance.replace('_', ' ')} recent behavior",
    )


def level_state_packet(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    facts: dict[str, Any],
    *,
    acceptance_window: int = 3,
) -> dict[str, Any]:
    level_name = str(facts.get("level_name") or "UNKNOWN")
    level_price = float(facts.get("level_price") or 0.0)
    buffer = float(facts.get("buffer") or 0.0)
    current_close = facts.get("current_close")
    role = _level_role(level_name)
    close_relation = _close_relation(current_close, level_price, buffer)
    acceptance_packet = _acceptance_state(
        bars,
        level_price,
        buffer,
        acceptance_window=acceptance_window,
    )
    event_state = _event_state(
        role,
        facts,
        close_relation,
        str(acceptance_packet.get("state") or "mixed"),
    )
    failed_break_tag = _failed_break_candidate(role, event_state)
    actionable = event_state in {
        "failed_break_reclaimed",
        "failed_break_rejected",
        "testing_support",
        "testing_resistance",
        "testing_reference",
    }
    return {
        "schema": "sharpedge.level_state.v1",
        "level_name": level_name,
        "level_price": level_price,
        "role": role,
        "buffer": buffer,
        "current_close": current_close,
        "close_relation": close_relation,
        "acceptance": acceptance_packet,
        "event_state": event_state,
        "failed_break_candidate": failed_break_tag,
        "actionable": actionable,
        "summary": _summary(
            level_name,
            level_price,
            event_state,
            str(acceptance_packet.get("state") or "mixed"),
        ),
        "facts": {
            "recent_breach_above": bool(facts.get("recent_breach_above")),
            "recent_breach_below": bool(facts.get("recent_breach_below")),
            "breach_above_highest_high": facts.get("breach_above_highest_high"),
            "breach_above_extension_pct": facts.get("breach_above_extension_pct"),
            "breach_below_deepest_low": facts.get("breach_below_deepest_low"),
            "breach_below_depth_pct": facts.get("breach_below_depth_pct"),
            "reclaim_above_level_index": facts.get("reclaim_above_level_index"),
            "bars_since_reclaim_above_level": facts.get(
                "bars_since_reclaim_above_level"
            ),
            "reject_below_level_index": facts.get("reject_below_level_index"),
            "bars_since_reject_below_level": facts.get("bars_since_reject_below_level"),
        },
    }


def build_level_state_map(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    levels: dict[str, Any] | None,
    *,
    level_names: tuple[str, ...] | list[str] | None = None,
    recent_window: int = 6,
    acceptance_window: int = 3,
) -> dict[str, dict[str, Any]]:
    active_levels = active_failed_break_levels(
        levels,
        level_names=level_names or LEVEL_STATE_LEVEL_NAMES,
    )
    facts_by_level = failed_break_facts_for_levels(
        bars,
        active_levels,
        level_names=tuple(active_levels.keys()),
        recent_window=recent_window,
    )
    return {
        name: level_state_packet(
            bars,
            facts,
            acceptance_window=acceptance_window,
        )
        for name, facts in facts_by_level.items()
    }


__all__ = [
    "LEVEL_STATE_LEVEL_NAMES",
    "build_level_state_map",
    "level_state_packet",
]
