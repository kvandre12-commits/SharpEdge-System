"""Deterministic failed-break market facts for SharpEdge.

This module computes opinion-free breach/reclaim mechanics from bars + levels.
It must not emit setup tags, bias labels, scores, or authority verdicts.
Different consumers can interpret the same facts packet differently.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim

FAILED_BREAK_LEVEL_NAMES = ("ORH", "ORL", "PDH", "PDL")
SUPPORT_LEVEL_NAMES = ("ORL", "PDL")
RESISTANCE_LEVEL_NAMES = ("ORH", "PDH")


def active_failed_break_levels(
    levels: dict[str, Any] | None,
    level_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    names = tuple(level_names or FAILED_BREAK_LEVEL_NAMES)
    source = levels or {}
    return {
        name: float(source[name])
        for name in names
        if isinstance(source.get(name), (int, float))
    }


def _first_close_above(
    bars: list[tuple], start_index: int | None, level_price: float
) -> int | None:
    if start_index is None:
        return None
    for index in range(start_index, len(bars)):
        if float(bars[index][4]) > level_price:
            return index
    return None


def _first_close_below(
    bars: list[tuple], start_index: int | None, level_price: float
) -> int | None:
    if start_index is None:
        return None
    for index in range(start_index, len(bars)):
        if float(bars[index][4]) < level_price:
            return index
    return None


def failed_break_facts(
    bars: list[tuple],
    level_name: str,
    level_price: float,
    *,
    recent_window: int = 6,
) -> dict[str, Any]:
    recent_window = max(int(recent_window or 0), 1)
    packet = {
        "schema": "sharpedge.failed_break_facts.v1",
        "level_name": level_name,
        "level_price": float(level_price),
        "buffer": prim.buffer_for_price(level_price),
        "total_bars": len(bars),
        "latest_bar_index": len(bars) - 1 if bars else None,
        "recent_window": recent_window,
        "recent_window_used": 0,
        "current_close": None,
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
    }
    if not bars:
        return packet

    level_price = float(level_price)
    buf = prim.buffer_for_price(level_price)
    latest_index = len(bars) - 1
    recent = bars[-min(recent_window, len(bars)) :]
    current_close = float(bars[-1][4])

    breach_below = [
        (index, float(bar[3]))
        for index, bar in enumerate(bars)
        if float(bar[3]) < level_price - buf
    ]
    breach_above = [
        (index, float(bar[2]))
        for index, bar in enumerate(bars)
        if float(bar[2]) > level_price + buf
    ]

    breach_below_latest_index = breach_below[-1][0] if breach_below else None
    breach_above_latest_index = breach_above[-1][0] if breach_above else None
    breach_below_deepest_low = min((low for _index, low in breach_below), default=None)
    breach_above_highest_high = max(
        (high for _index, high in breach_above), default=None
    )
    reclaim_above_level_index = _first_close_above(
        bars, breach_below_latest_index, level_price
    )
    reject_below_level_index = _first_close_below(
        bars, breach_above_latest_index, level_price
    )

    packet.update(
        {
            "buffer": buf,
            "recent_window_used": len(recent),
            "current_close": current_close,
            "current_close_above_level": current_close > level_price,
            "current_close_below_level": current_close < level_price,
            "recent_high": max(float(bar[2]) for bar in recent),
            "recent_low": min(float(bar[3]) for bar in recent),
            "recent_breach_above": any(
                float(bar[2]) > level_price + buf for bar in recent
            ),
            "recent_breach_below": any(
                float(bar[3]) < level_price - buf for bar in recent
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
        }
    )
    return packet


def failed_break_facts_for_levels(
    bars: list[tuple],
    levels: dict[str, Any] | None,
    *,
    level_names: tuple[str, ...] | list[str] | None = None,
    recent_window: int = 6,
) -> dict[str, dict[str, Any]]:
    active_levels = active_failed_break_levels(levels, level_names=level_names)
    return {
        name: failed_break_facts(
            bars,
            name,
            level,
            recent_window=recent_window,
        )
        for name, level in active_levels.items()
    }


__all__ = [
    "FAILED_BREAK_LEVEL_NAMES",
    "RESISTANCE_LEVEL_NAMES",
    "SUPPORT_LEVEL_NAMES",
    "active_failed_break_levels",
    "failed_break_facts",
    "failed_break_facts_for_levels",
]
