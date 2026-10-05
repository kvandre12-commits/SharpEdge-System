"""Gate classification helpers for SharpEdge setup/context cards."""

from __future__ import annotations

from typing import Any

_GATE_MAP = {
    "FAILED BREAKDOWN": {
        "gate_id": "failed_breakdown_reclaim",
        "gate_family": "reclaim",
        "workflow": "mean_reversion_reclaim",
        "actionable": True,
    },
    "FAILED BREAKOUT": {
        "gate_id": "failed_breakout_reject",
        "gate_family": "reject",
        "workflow": "mean_reversion_reject",
        "actionable": True,
    },
    "DOWNSIDE EXHAUSTION": {
        "gate_id": "downside_exhaustion_fade",
        "gate_family": "exhaustion",
        "workflow": "exhaustion_fade",
        "actionable": True,
    },
    "UPSIDE EXHAUSTION": {
        "gate_id": "upside_exhaustion_fade",
        "gate_family": "exhaustion",
        "workflow": "exhaustion_fade",
        "actionable": True,
    },
    "EXHAUSTION -> RUNNER HANDOFF": {
        "gate_id": "exhaustion_runner_handoff",
        "gate_family": "handoff",
        "workflow": "directional_continuation",
        "actionable": True,
    },
    "POST-SELLOFF COIL": {
        "gate_id": "post_selloff_coil_break",
        "gate_family": "compression",
        "workflow": "continuation_breakout",
        "actionable": True,
    },
    "STICKY DAY (CALM/CHOP)": {
        "gate_id": "sticky_day_magnet_fade",
        "gate_family": "day_type",
        "workflow": "magnet_fade",
        "actionable": False,
    },
    "RUNNER DAY (WHEEE)": {
        "gate_id": "runner_day_directional_continuation",
        "gate_family": "day_type",
        "workflow": "directional_continuation",
        "actionable": False,
    },
}


def _tag(setup: dict[str, Any] | None) -> str:
    return str((setup or {}).get("tag", "")).upper()


def _entry_window_allows_action(setup: dict[str, Any], tag: str) -> bool:
    if tag not in {"FAILED BREAKDOWN", "FAILED BREAKOUT"}:
        return True
    return bool(setup.get("entry_window_open", True))


def gate_metadata(setup: dict[str, Any] | None) -> dict[str, Any]:
    setup = setup or {}
    tag = _tag(setup)
    meta = _GATE_MAP.get(tag, {})
    actionable = bool(meta.get("actionable", False)) and _entry_window_allows_action(
        setup, tag
    )
    return {
        "tag": setup.get("tag"),
        "bias": setup.get("bias"),
        "kind": setup.get("kind"),
        "gate_id": meta.get("gate_id"),
        "gate_family": meta.get("gate_family"),
        "workflow": meta.get("workflow"),
        "actionable": actionable,
        "event_detected": setup.get("event_detected"),
        "event_age_bars": setup.get("event_age_bars"),
        "entry_window_open": setup.get("entry_window_open"),
        "level_name": setup.get("level_name"),
        "level_price": setup.get("level_price"),
        "trigger_price": setup.get("trigger_price"),
        "bars_ago": setup.get("bars_ago"),
    }


def is_actionable_setup(setup: dict[str, Any] | None) -> bool:
    return bool(gate_metadata(setup).get("actionable"))


def primary_trade_setup(setups: list[dict[str, Any]] | None) -> dict[str, Any]:
    for setup in setups or []:
        if is_actionable_setup(setup):
            return setup
    return (setups or [{}])[0] if setups else {}


def primary_context_setup(setups: list[dict[str, Any]] | None) -> dict[str, Any]:
    for setup in setups or []:
        if not is_actionable_setup(setup):
            return setup
    return (setups or [{}])[0] if setups else {}


__all__ = [
    "gate_metadata",
    "is_actionable_setup",
    "primary_context_setup",
    "primary_trade_setup",
]
