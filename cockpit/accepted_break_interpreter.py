"""Shared accepted-break interpretation helpers for SharpEdge.

These helpers consume level-state packets and emit reusable accepted-break
objects. They do not own execution authority or setup-card identity; they only
interpret accepted-break semantics consistently for downstream consumers.
"""

from __future__ import annotations

from typing import Any

SUPPORT_LEVELS = ("ORL", "PDL")
RESISTANCE_LEVELS = ("ORH", "PDH")
DEFAULT_ACCEPTANCE_CLOSES = 3


def collect_accepted_break_events(
    level_states: dict[str, dict[str, Any]] | None,
    *,
    level_order: tuple[str, ...] | list[str] | None = None,
    acceptance_closes: int = DEFAULT_ACCEPTANCE_CLOSES,
) -> list[dict[str, Any]]:
    packets = level_states or {}
    ordered_names = tuple(level_order or packets.keys())
    events: list[dict[str, Any]] = []
    for name in ordered_names:
        state = packets.get(name) or {}
        level = state.get("level_price")
        acceptance = (state.get("acceptance") or {}).get("state")
        event_state = str(state.get("event_state") or "")
        if name in RESISTANCE_LEVELS and event_state == "accepted_above_resistance":
            events.append(
                {
                    "state": "accepted_breakout",
                    "bias": "CALLS",
                    "level_name": name,
                    "level_price": float(level),
                    "score": 72,
                    "reason": f"{acceptance_closes} closes accepted above {name} {float(level):.2f}",
                }
            )
        elif (
            name in SUPPORT_LEVELS
            and acceptance == "accepted_below"
            and state.get("close_relation") == "below"
        ):
            events.append(
                {
                    "state": "accepted_breakdown",
                    "bias": "PUTS",
                    "level_name": name,
                    "level_price": float(level),
                    "score": 72,
                    "reason": f"{acceptance_closes} closes accepted below {name} {float(level):.2f}",
                }
            )
    return sorted(events, key=lambda item: float(item["score"]), reverse=True)


def best_accepted_break_event(
    level_states: dict[str, dict[str, Any]] | None,
    *,
    level_order: tuple[str, ...] | list[str] | None = None,
    acceptance_closes: int = DEFAULT_ACCEPTANCE_CLOSES,
) -> dict[str, Any]:
    events = collect_accepted_break_events(
        level_states,
        level_order=level_order,
        acceptance_closes=acceptance_closes,
    )
    return events[0] if events else {}


def accepted_break_break_state(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": str(event.get("state") or "no_active_break"),
        "bias": str(event.get("bias") or "NEUTRAL"),
        "level_name": event.get("level_name"),
        "level_price": event.get("level_price"),
        "score": int(event.get("score") or 0),
        "reason": str(event.get("reason") or ""),
    }


__all__ = [
    "DEFAULT_ACCEPTANCE_CLOSES",
    "accepted_break_break_state",
    "best_accepted_break_event",
    "collect_accepted_break_events",
]
