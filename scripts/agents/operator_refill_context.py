"""Compact canonical refill intelligence for operator-facing artifacts.

This module is descriptive only. It must never grant trade or broker authority.
"""

from __future__ import annotations

from typing import Any


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "mode",
            "event_date",
            "threshold_pct",
            "move_pct",
            "target",
            "elapsed_trading_days",
            "phase",
            "ema200_side",
            "ema200_distance_bucket",
        )
        if key in item
    }


def summarize_refill_context(
    signal: dict[str, Any], curator: dict[str, Any]
) -> dict[str, Any]:
    """Return a bounded operator summary without changing permission state."""
    context = signal.get("historical_refill_context") or {}
    stack = context.get("active_refill_stack") or {}
    if context.get("available") and stack:
        history = stack.get("historical_stack_surface") or {}
        return {
            "available": True,
            "authority": "diagnostic_only",
            "source": "signal.historical_refill_context",
            "surface_generated_at": context.get("surface_generated_at"),
            "stack_label": stack.get("stack_label"),
            "active_count": stack.get("active_count"),
            "oldest_event_date": stack.get("oldest_event_date"),
            "latest_event_date": stack.get("latest_event_date"),
            "nearest_target": stack.get("nearest_target"),
            "highest_target": stack.get("highest_target"),
            "items": [_compact_item(item) for item in stack.get("items", [])],
            "historical_stack_stats": history.get("overall"),
            "entry_usage": "context_and_veto_only",
            "permission_effect": "none",
        }

    curated = curator.get("cockpit_context") or {}
    if curated.get("active_stack_count"):
        return {
            "available": True,
            "authority": "diagnostic_only",
            "source": "nerv_curator.cockpit_context",
            "stack_label": curated.get("stack_label"),
            "active_count": curated.get("active_stack_count"),
            "nearest_target": curated.get("nearest_refill_target"),
            "highest_target": curated.get("highest_refill_target"),
            "items": [],
            "historical_stack_stats": None,
            "entry_usage": "context_and_veto_only",
            "permission_effect": "none",
        }

    return {
        "available": False,
        "authority": "diagnostic_only",
        "reason": context.get("reason", "active_refill_context_unavailable"),
        "entry_usage": "context_and_veto_only",
        "permission_effect": "none",
    }


__all__ = ["summarize_refill_context"]
