"""Local shadow-policy state manager for SharpEdge edge tokens.

EdgeTokenPosition models local policy state, not broker state. It is the highest
protocol in the internal reasoning stack; external execution begins after it and
must add its own approval/broker authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from setup_event_lifecycle import ACTIVE_EVENT_STATUSES

CLEAR_EVENT_STATUSES = {"invalidated", "expired"}
_ENTRY_ACTIONS = {"confirmed"}
_POLICY = {
    "contracts_per_token": 1,
    "entry_on_status": sorted(_ENTRY_ACTIONS),
    "watch_only_status": ["candidate"],
    "exit_on_status": sorted(CLEAR_EVENT_STATUSES),
    "replacement_policy": "close_first_no_same_tick_flip",
    "max_concurrent_tokens": 1,
}


def load_previous_edge_token_position(signal_path: Path) -> dict[str, Any]:
    if not signal_path.exists():
        return {}
    try:
        payload = json.loads(signal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    edge_token = payload.get("edge_token_position")
    return edge_token if isinstance(edge_token, dict) else {}


def _direction_from_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).upper()
    if "CALLS" in text or "BULLISH" in text:
        return "CALLS"
    if "PUTS" in text or "BEARISH" in text:
        return "PUTS"
    return "NEUTRAL"


def _event_lookup(decision_receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(event.get("event_id")): event
        for event in (decision_receipt.get("setup_events") or [])
        if isinstance(event, dict) and event.get("event_id")
    }


def _token_summary(event: dict[str, Any], side: str) -> dict[str, Any]:
    level = event.get("level") or {}
    return {
        "token_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "event_scope": event.get("event_scope"),
        "gate_family": event.get("gate_family"),
        "workflow": event.get("workflow"),
        "status": event.get("status"),
        "side": side,
        "observation_count": event.get("observation_count"),
        "confidence": event.get("confidence"),
        "level_name": level.get("name"),
        "level_price": level.get("price"),
        "first_seen_ts": event.get("first_seen_ts"),
        "last_seen_ts": event.get("last_seen_ts"),
        "last_confirmed_ts": event.get("last_confirmed_ts"),
    }


def _previous_open_token(previous_state: dict[str, Any]) -> dict[str, Any]:
    if str(previous_state.get("position_state") or "").lower() != "open":
        return {}
    token = previous_state.get("current_token")
    return token if isinstance(token, dict) and token.get("token_id") else {}


def _closing_token(
    previous_token: dict[str, Any], current_events: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not previous_token:
        return {}
    token_id = str(previous_token.get("token_id") or "")
    current_event = current_events.get(token_id) or {}
    status = str(current_event.get("status") or "").lower()
    if status in CLEAR_EVENT_STATUSES:
        closing = _token_summary(
            current_event, str(previous_token.get("side") or "NEUTRAL")
        )
        closing["clear_reason"] = status
        return closing
    return {
        **previous_token,
        "clear_reason": "replaced",
    }


def _active_directional_token(
    permission: dict[str, Any],
    decision_receipt: dict[str, Any],
) -> dict[str, Any]:
    event = decision_receipt.get("primary_setup_event") or {}
    status = str(event.get("status") or "").lower()
    scope = str(event.get("event_scope") or "").lower()
    if status not in ACTIVE_EVENT_STATUSES or scope != "setup":
        return {}
    side = _direction_from_text(
        decision_receipt.get("bias"),
        permission.get("bias"),
        event.get("bias"),
        event.get("event_type"),
    )
    if side not in {"CALLS", "PUTS"}:
        return {}
    return _token_summary(event, side)


def _enter_action(side: str) -> str:
    return "enter_call" if side == "CALLS" else "enter_put"


def _is_entry_ready(token: dict[str, Any]) -> bool:
    return str(token.get("status") or "").lower() in _ENTRY_ACTIONS


def build_edge_token_position(
    signal_ts: str,
    permission: dict[str, Any] | None = None,
    decision_receipt: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    permission = permission or {}
    decision_receipt = decision_receipt or {}
    previous_state = previous_state or {}

    current_events = _event_lookup(decision_receipt)
    previous_token = _previous_open_token(previous_state)
    current_token = _active_directional_token(permission, decision_receipt)

    result = {
        "schema": "sharpedge.edge_token_position.v1",
        "ts": signal_ts,
        "policy": _POLICY,
        "position_state": "flat",
        "contracts_held": 0,
        "contracts_delta": 0,
        "suggested_action": "stand_down",
        "recommended_actions": ["stand_down"],
        "action_reason": "no active directional edge token",
        "token_status": "none",
        "current_token": None,
        "closing_token": None,
        "pending_token": None,
    }

    if (
        current_token
        and previous_token
        and current_token["token_id"] == previous_token.get("token_id")
    ):
        return {
            **result,
            "position_state": "open",
            "contracts_held": 1,
            "suggested_action": "hold",
            "recommended_actions": ["hold"],
            "action_reason": "edge token is still active; shadow policy keeps one-contract exposure marked as open.",
            "token_status": "active",
            "current_token": current_token,
        }

    if current_token and previous_token:
        closing_token = _closing_token(previous_token, current_events)
        return {
            **result,
            "contracts_delta": -1,
            "suggested_action": "close_position",
            "recommended_actions": ["close_position"],
            "action_reason": (
                "previous edge token cleared or got replaced; shadow policy marks "
                "prior exposure for closure before considering any replacement."
            ),
            "token_status": "reset_required",
            "closing_token": closing_token,
            "pending_token": current_token,
        }

    if current_token and not _is_entry_ready(current_token):
        return {
            **result,
            "suggested_action": "stand_down",
            "recommended_actions": ["stand_down"],
            "action_reason": (
                "directional setup is only a candidate; wait for confirmation before "
                "minting an actionable edge token."
            ),
            "token_status": "pending_confirmation",
            "pending_token": current_token,
        }

    if current_token:
        side = current_token["side"]
        return {
            **result,
            "position_state": "open",
            "contracts_held": 1,
            "contracts_delta": 1,
            "suggested_action": _enter_action(side),
            "recommended_actions": [_enter_action(side)],
            "action_reason": (
                f"confirmed {side} edge token is active; shadow policy marks "
                "one-contract exposure as eligible while this token remains active."
            ),
            "token_status": "active",
            "current_token": current_token,
        }

    if previous_token:
        closing_token = _closing_token(previous_token, current_events)
        clear_reason = closing_token.get("clear_reason") or "cleared"
        return {
            **result,
            "contracts_delta": -1,
            "suggested_action": "close_position",
            "recommended_actions": ["close_position"],
            "action_reason": f"edge token cleared ({clear_reason}); shadow policy marks prior exposure for closure.",
            "token_status": "cleared",
            "closing_token": closing_token,
        }

    return result


__all__ = [
    "ACTIVE_EVENT_STATUSES",
    "CLEAR_EVENT_STATUSES",
    "build_edge_token_position",
    "load_previous_edge_token_position",
]
