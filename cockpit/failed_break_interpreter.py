"""Shared failed-break interpretation helpers for SharpEdge.

These helpers consume level-state packets and emit reusable failed-break event
objects. They do not own setup-card presentation or execution-grammar authority,
but they provide one consistent semantic read for both consumers.
"""

from __future__ import annotations

from typing import Any

RECENT_BARS = 6
SUPPORT_LEVELS = ("ORL", "PDL")
RESISTANCE_LEVELS = ("ORH", "PDH")


def _support_failed_break_event(
    level_state: dict[str, Any] | None,
    *,
    recent_bars: int = RECENT_BARS,
) -> dict[str, Any] | None:
    if not level_state or level_state.get("event_state") != "failed_break_reclaimed":
        return None
    facts = level_state.get("facts") or {}
    reclaim_idx = facts.get("reclaim_above_level_index")
    bars_ago = facts.get("bars_since_reclaim_above_level")
    deepest = facts.get("breach_below_deepest_low")
    depth = facts.get("breach_below_depth_pct")
    level = level_state.get("level_price")
    name = level_state.get("level_name")
    if (
        reclaim_idx is None
        or bars_ago is None
        or deepest is None
        or depth is None
        or level is None
        or name is None
    ):
        return None
    entry_window_open = int(bars_ago) <= int(recent_bars)
    if not entry_window_open and not level_state.get("event_detected", True):
        return None
    score = float(depth) + (int(recent_bars) - int(bars_ago))
    return {
        "state": "failed_breakdown",
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS",
        "setup_bias": "CALLS (bullish)",
        "setup_kind": "ok",
        "level_name": str(name),
        "level_price": float(level),
        "trigger_price": float(deepest),
        "bars_ago": int(bars_ago),
        "event_age_bars": int(bars_ago),
        "event_detected": True,
        "entry_window_open": entry_window_open,
        "magnitude_pct": float(depth),
        "score": score,
    }


def _resistance_failed_break_event(
    level_state: dict[str, Any] | None,
    *,
    recent_bars: int = RECENT_BARS,
) -> dict[str, Any] | None:
    if not level_state or level_state.get("event_state") != "failed_break_rejected":
        return None
    facts = level_state.get("facts") or {}
    reject_idx = facts.get("reject_below_level_index")
    bars_ago = facts.get("bars_since_reject_below_level")
    highest = facts.get("breach_above_highest_high")
    ext = facts.get("breach_above_extension_pct")
    level = level_state.get("level_price")
    name = level_state.get("level_name")
    if (
        reject_idx is None
        or bars_ago is None
        or highest is None
        or ext is None
        or level is None
        or name is None
    ):
        return None
    entry_window_open = int(bars_ago) <= int(recent_bars)
    if not entry_window_open and not level_state.get("event_detected", True):
        return None
    score = float(ext) + (int(recent_bars) - int(bars_ago))
    return {
        "state": "failed_breakout",
        "tag": "FAILED BREAKOUT",
        "bias": "PUTS",
        "setup_bias": "PUTS (bearish)",
        "setup_kind": "bad",
        "level_name": str(name),
        "level_price": float(level),
        "trigger_price": float(highest),
        "bars_ago": int(bars_ago),
        "event_age_bars": int(bars_ago),
        "event_detected": True,
        "entry_window_open": entry_window_open,
        "magnitude_pct": float(ext),
        "score": score,
    }


def collect_failed_break_events(
    level_states: dict[str, dict[str, Any]] | None,
    *,
    level_order: tuple[str, ...] | list[str] | None = None,
    recent_bars: int = RECENT_BARS,
    entry_window_only: bool = False,
) -> list[dict[str, Any]]:
    packets = level_states or {}
    ordered_names = tuple(level_order or packets.keys())
    events: list[dict[str, Any]] = []
    for name in ordered_names:
        packet = packets.get(name)
        if name in SUPPORT_LEVELS:
            event = _support_failed_break_event(packet, recent_bars=recent_bars)
        elif name in RESISTANCE_LEVELS:
            event = _resistance_failed_break_event(packet, recent_bars=recent_bars)
        else:
            event = None
        if event and (event.get("entry_window_open") or not entry_window_only):
            events.append(event)
    return sorted(
        events,
        key=lambda item: float(item["score"]),
        reverse=True,
    )


def best_failed_break_event(
    level_states: dict[str, dict[str, Any]] | None,
    *,
    level_order: tuple[str, ...] | list[str] | None = None,
    recent_bars: int = RECENT_BARS,
    entry_window_only: bool = True,
) -> dict[str, Any]:
    events = collect_failed_break_events(
        level_states,
        level_order=level_order,
        recent_bars=recent_bars,
        entry_window_only=entry_window_only,
    )
    return events[0] if events else {}


def failed_break_setup_card(event: dict[str, Any]) -> dict[str, Any]:
    state = str(event.get("state") or "")
    name = str(event.get("level_name") or "")
    level = float(event.get("level_price") or 0.0)
    trigger = float(event.get("trigger_price") or 0.0)
    bars_ago = int(event.get("bars_ago") or 0)
    magnitude = float(event.get("magnitude_pct") or 0.0)
    window_text = (
        "entry window open" if event.get("entry_window_open") else "entry window stale"
    )
    if state == "failed_breakdown":
        detail = (
            f"reclaimed {name} ${level:.2f} {bars_ago}m ago after "
            f"stabbing ${trigger:.2f} (-{magnitude:.2f}% below) - bear trap; "
            f"{window_text}"
        )
    else:
        detail = (
            f"rejected {name} ${level:.2f} {bars_ago}m ago after "
            f"poking ${trigger:.2f} (+{magnitude:.2f}% above) - bull trap; "
            f"{window_text}"
        )
    return {
        "tag": str(event.get("tag") or ""),
        "bias": str(event.get("setup_bias") or "NEUTRAL"),
        "kind": str(event.get("setup_kind") or "warn"),
        "detail": detail,
        "score": float(event.get("score") or 0.0),
        "level_name": name,
        "level_price": round(level, 2),
        "trigger_price": round(trigger, 2),
        "bars_ago": bars_ago,
        "event_age_bars": int(event.get("event_age_bars") or bars_ago),
        "event_detected": bool(event.get("event_detected", True)),
        "entry_window_open": bool(event.get("entry_window_open")),
    }


def failed_break_break_state(event: dict[str, Any]) -> dict[str, Any]:
    state = str(event.get("state") or "")
    name = str(event.get("level_name") or "")
    level = float(event.get("level_price") or 0.0)
    trigger = float(event.get("trigger_price") or 0.0)
    if state == "failed_breakdown":
        reason = (
            f"sellers trapped below {name} {level:.2f}; reclaimed from {trigger:.2f}"
        )
    else:
        reason = f"buyers trapped above {name} {level:.2f}; rejected from {trigger:.2f}"
    return {
        "state": state,
        "bias": str(event.get("bias") or "NEUTRAL"),
        "level_name": name,
        "level_price": level,
        "trigger_price": trigger,
        "score": 88,
        "reason": reason,
    }


__all__ = [
    "RECENT_BARS",
    "best_failed_break_event",
    "collect_failed_break_events",
    "failed_break_break_state",
    "failed_break_setup_card",
]
