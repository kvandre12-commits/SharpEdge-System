"""Deterministic level-interaction facts for SharpEdge.

This module computes opinion-free mechanical observations from bars + levels.
It must not emit setup tags, bias labels, scores, or authority verdicts.
Different consumers can interpret the same facts packet differently.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim
from reference_geometry import relation_to_reference

LEVEL_INTERACTION_LEVEL_NAMES = ("ORH", "ORL", "PDH", "PDL", "PDC")
FAILED_BREAK_LEVEL_NAMES = ("ORH", "ORL", "PDH", "PDL")
SUPPORT_LEVEL_NAMES = ("ORL", "PDL")
RESISTANCE_LEVEL_NAMES = ("ORH", "PDH")


def active_level_interaction_levels(
    levels: dict[str, Any] | None,
    level_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    names = tuple(level_names or LEVEL_INTERACTION_LEVEL_NAMES)
    source = levels or {}
    return {
        name: float(source[name])
        for name in names
        if isinstance(source.get(name), (int, float))
    }


def active_failed_break_levels(
    levels: dict[str, Any] | None,
    level_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    return active_level_interaction_levels(
        levels,
        level_names=level_names or FAILED_BREAK_LEVEL_NAMES,
    )


def _level_role(level_name: str) -> str:
    if level_name in SUPPORT_LEVEL_NAMES:
        return "support"
    if level_name in RESISTANCE_LEVEL_NAMES:
        return "resistance"
    return "reference"


def _count_recent_close_relations(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    buffer: float,
    *,
    acceptance_window: int,
) -> dict[str, int]:
    recent = list(bars[-max(int(acceptance_window or 0), 1) :])
    closes = [float(bar[4]) for bar in recent]
    above = sum(1 for close in closes if close > level_price + buffer)
    below = sum(1 for close in closes if close < level_price - buffer)
    return {
        "acceptance_window_used": len(closes),
        "closes_above_count": above,
        "closes_below_count": below,
        "closes_at_level_count": max(0, len(closes) - above - below),
    }


def _first_close_relation_index(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    buffer: float,
    relation: str,
    *,
    start_index: int = 0,
) -> int | None:
    for index in range(max(int(start_index), 0), len(bars)):
        if (
            relation_to_reference(
                float(bars[index][4]),
                level_price,
                at_label="at_level",
                buffer=buffer,
            )
            == relation
        ):
            return index
    return None


def _first_close_above_level(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    *,
    start_index: int | None = None,
) -> int | None:
    if start_index is None:
        return None
    for index in range(max(int(start_index), 0), len(bars)):
        if float(bars[index][4]) > level_price:
            return index
    return None


def _first_close_below_level(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    *,
    start_index: int | None = None,
) -> int | None:
    if start_index is None:
        return None
    for index in range(max(int(start_index), 0), len(bars)):
        if float(bars[index][4]) < level_price:
            return index
    return None


def _consecutive_close_count_from_end(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_price: float,
    buffer: float,
    relation: str,
) -> int:
    count = 0
    for bar in reversed(bars):
        if (
            relation_to_reference(
                float(bar[4]),
                level_price,
                at_label="at_level",
                buffer=buffer,
            )
            != relation
        ):
            break
        count += 1
    return count


def level_interaction_facts(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_name: str,
    level_price: float,
    *,
    recent_window: int = 6,
    acceptance_window: int = 3,
) -> dict[str, Any]:
    recent_window = max(int(recent_window or 0), 1)
    acceptance_window = max(int(acceptance_window or 0), 1)
    level_price = float(level_price)
    buffer = prim.buffer_for_price(level_price)
    packet = {
        "schema": "sharpedge.level_interaction_facts.v1",
        "level_name": level_name,
        "level_price": level_price,
        "role": _level_role(level_name),
        "buffer": buffer,
        "total_bars": len(bars),
        "latest_bar_index": len(bars) - 1 if bars else None,
        "recent_window": recent_window,
        "recent_window_used": 0,
        "acceptance_window": acceptance_window,
        "acceptance_window_used": 0,
        "current_close": None,
        "current_high": None,
        "current_low": None,
        "current_close_relation": "unknown",
        "current_high_relation": "unknown",
        "current_low_relation": "unknown",
        "current_close_above_level": None,
        "current_close_below_level": None,
        "recent_high": None,
        "recent_low": None,
        "recent_breach_above": False,
        "recent_breach_below": False,
        "breach_above_latest_index": None,
        "breach_above_highest_high": None,
        "breach_above_extension_pct": None,
        "reject_below_level_index": None,
        "bars_since_reject_below_level": None,
        "breach_below_latest_index": None,
        "breach_below_deepest_low": None,
        "breach_below_depth_pct": None,
        "reclaim_above_level_index": None,
        "bars_since_reclaim_above_level": None,
        "closes_above_count": 0,
        "closes_below_count": 0,
        "closes_at_level_count": 0,
        "first_close_above_index": None,
        "first_close_below_index": None,
        "hold_above_count": 0,
        "hold_below_count": 0,
    }
    if not bars:
        return packet

    latest_index = len(bars) - 1
    recent = list(bars[-min(recent_window, len(bars)) :])
    current_high = float(bars[-1][2])
    current_low = float(bars[-1][3])
    current_close = float(bars[-1][4])

    breach_below = [
        (index, float(bar[3]))
        for index, bar in enumerate(bars)
        if float(bar[3]) < level_price - buffer
    ]
    breach_above = [
        (index, float(bar[2]))
        for index, bar in enumerate(bars)
        if float(bar[2]) > level_price + buffer
    ]
    breach_below_latest_index = breach_below[-1][0] if breach_below else None
    breach_above_latest_index = breach_above[-1][0] if breach_above else None
    breach_below_deepest_low = min((low for _index, low in breach_below), default=None)
    breach_above_highest_high = max(
        (high for _index, high in breach_above), default=None
    )
    reclaim_above_level_index = _first_close_above_level(
        bars,
        level_price,
        start_index=breach_below_latest_index,
    )
    reject_below_level_index = _first_close_below_level(
        bars,
        level_price,
        start_index=breach_above_latest_index,
    )
    close_counts = _count_recent_close_relations(
        bars,
        level_price,
        buffer,
        acceptance_window=acceptance_window,
    )

    packet.update(
        {
            "recent_window_used": len(recent),
            "acceptance_window_used": close_counts["acceptance_window_used"],
            "current_close": current_close,
            "current_high": current_high,
            "current_low": current_low,
            "current_close_relation": relation_to_reference(
                current_close,
                level_price,
                at_label="at_level",
                buffer=buffer,
            ),
            "current_high_relation": relation_to_reference(
                current_high,
                level_price,
                at_label="at_level",
                buffer=buffer,
            ),
            "current_low_relation": relation_to_reference(
                current_low,
                level_price,
                at_label="at_level",
                buffer=buffer,
            ),
            "current_close_above_level": current_close > level_price,
            "current_close_below_level": current_close < level_price,
            "recent_high": max(float(bar[2]) for bar in recent),
            "recent_low": min(float(bar[3]) for bar in recent),
            "recent_breach_above": any(
                float(bar[2]) > level_price + buffer for bar in recent
            ),
            "recent_breach_below": any(
                float(bar[3]) < level_price - buffer for bar in recent
            ),
            "breach_above_latest_index": breach_above_latest_index,
            "breach_above_highest_high": breach_above_highest_high,
            "breach_above_extension_pct": (
                (breach_above_highest_high - level_price) / level_price * 100
                if breach_above_highest_high is not None and level_price
                else None
            ),
            "reject_below_level_index": reject_below_level_index,
            "bars_since_reject_below_level": (
                latest_index - reject_below_level_index
                if reject_below_level_index is not None
                else None
            ),
            "breach_below_latest_index": breach_below_latest_index,
            "breach_below_deepest_low": breach_below_deepest_low,
            "breach_below_depth_pct": (
                (level_price - breach_below_deepest_low) / level_price * 100
                if breach_below_deepest_low is not None and level_price
                else None
            ),
            "reclaim_above_level_index": reclaim_above_level_index,
            "bars_since_reclaim_above_level": (
                latest_index - reclaim_above_level_index
                if reclaim_above_level_index is not None
                else None
            ),
            "closes_above_count": close_counts["closes_above_count"],
            "closes_below_count": close_counts["closes_below_count"],
            "closes_at_level_count": close_counts["closes_at_level_count"],
            "first_close_above_index": _first_close_relation_index(
                bars,
                level_price,
                buffer,
                "above",
            ),
            "first_close_below_index": _first_close_relation_index(
                bars,
                level_price,
                buffer,
                "below",
            ),
            "hold_above_count": _consecutive_close_count_from_end(
                bars,
                level_price,
                buffer,
                "above",
            ),
            "hold_below_count": _consecutive_close_count_from_end(
                bars,
                level_price,
                buffer,
                "below",
            ),
        }
    )
    return packet


def level_interaction_facts_for_levels(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    levels: dict[str, Any] | None,
    *,
    level_names: tuple[str, ...] | list[str] | None = None,
    recent_window: int = 6,
    acceptance_window: int = 3,
) -> dict[str, dict[str, Any]]:
    active_levels = active_level_interaction_levels(levels, level_names=level_names)
    return {
        name: level_interaction_facts(
            bars,
            name,
            level,
            recent_window=recent_window,
            acceptance_window=acceptance_window,
        )
        for name, level in active_levels.items()
    }


__all__ = [
    "FAILED_BREAK_LEVEL_NAMES",
    "LEVEL_INTERACTION_LEVEL_NAMES",
    "RESISTANCE_LEVEL_NAMES",
    "SUPPORT_LEVEL_NAMES",
    "active_failed_break_levels",
    "active_level_interaction_levels",
    "level_interaction_facts",
    "level_interaction_facts_for_levels",
]
