"""Compatibility wrapper for failed-break facts.

The shared mechanical primitive now lives in ``level_interaction_facts.py``.
This module preserves the historical failed-break packet shape for callers that
still depend on ``sharpedge.failed_break_facts.v1``.
"""

from __future__ import annotations

from typing import Any

from level_interaction_facts import (
    FAILED_BREAK_LEVEL_NAMES,
    RESISTANCE_LEVEL_NAMES,
    SUPPORT_LEVEL_NAMES,
    active_failed_break_levels,
    level_interaction_facts,
)

_FAILED_BREAK_KEYS = (
    "level_name",
    "level_price",
    "buffer",
    "total_bars",
    "latest_bar_index",
    "recent_window",
    "recent_window_used",
    "current_close",
    "current_close_above_level",
    "current_close_below_level",
    "recent_high",
    "recent_low",
    "recent_breach_above",
    "recent_breach_below",
    "breach_above_latest_index",
    "breach_above_highest_high",
    "breach_above_extension_pct",
    "reject_below_level_index",
    "bars_since_reject_below_level",
    "breach_below_latest_index",
    "breach_below_deepest_low",
    "breach_below_depth_pct",
    "reclaim_above_level_index",
    "bars_since_reclaim_above_level",
)


def failed_break_facts(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    level_name: str,
    level_price: float,
    *,
    recent_window: int = 6,
) -> dict[str, Any]:
    interaction = level_interaction_facts(
        bars,
        level_name,
        level_price,
        recent_window=recent_window,
    )
    packet = {key: interaction.get(key) for key in _FAILED_BREAK_KEYS}
    packet["schema"] = "sharpedge.failed_break_facts.v1"
    return packet


def failed_break_facts_for_levels(
    bars: list[tuple[Any, ...]] | list[list[Any]],
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
