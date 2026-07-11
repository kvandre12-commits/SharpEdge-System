"""Setup-event lifecycle metadata for SharpEdge.

This module tracks persistence, confirmation, invalidation, and expiry for
setup events across ticks. It is not a fresh setup detector and should not be
used as a substitute for current-bar evidence.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from gate_workflows import gate_metadata
from setup_conviction import sync_setup_evidence_fields

ACTIVE_EVENT_STATUSES = {"candidate", "confirmed"}
_STATUS_ORDER = {"confirmed": 0, "candidate": 1, "invalidated": 2, "expired": 3}


def _is_active_status(status: Any) -> bool:
    return str(status or "").lower() in ACTIVE_EVENT_STATUSES


def _clamp(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _slug(text: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in text]
    return "".join(cleaned).strip("_") or "event"


def _event_scope(meta: dict[str, Any]) -> str:
    return "setup" if meta.get("actionable") else "context"


def _event_id(tag: str, level_name: str = "", level_price: float | None = None) -> str:
    parts = [_slug(tag)]
    if level_name:
        parts.append(_slug(level_name))
    if isinstance(level_price, (int, float)):
        parts.append(f"{level_price:.2f}")
    return ":".join(parts)


def _family_key(
    tag: str,
    meta: dict[str, Any],
    level_name: str = "",
    level_price: float | None = None,
) -> str:
    family = str(meta.get("gate_family") or _slug(tag))
    scope = _event_scope(meta)
    if level_name:
        return f"{scope}:{family}:{_slug(level_name)}"
    if isinstance(level_price, (int, float)):
        return f"{scope}:{family}:{level_price:.2f}"
    return f"{scope}:{family}"


def _confidence(setup: dict[str, Any], meta: dict[str, Any]) -> int:
    raw = setup.get("score")
    if isinstance(raw, (int, float)) and raw >= 20:
        return _clamp(int(round(raw)))
    return 70 if meta.get("actionable") else 48


def _normalize_setup_event(setup: dict[str, Any]) -> dict[str, Any]:
    meta = gate_metadata(setup)
    tag = str(setup.get("tag") or meta.get("tag") or "UNKNOWN EVENT")
    level_name = str(setup.get("level_name") or "")
    level_price = setup.get("level_price")
    return {
        "event_id": _event_id(tag, level_name, level_price),
        "event_type": tag,
        "event_scope": _event_scope(meta),
        "family_key": _family_key(tag, meta, level_name, level_price),
        "bias": setup.get("bias"),
        "kind": setup.get("kind"),
        "detail": setup.get("detail"),
        "confidence": _confidence(setup, meta),
        "gate_id": meta.get("gate_id"),
        "gate_family": meta.get("gate_family"),
        "workflow": meta.get("workflow"),
        "level": {
            "name": level_name or None,
            "price": level_price if isinstance(level_price, (int, float)) else None,
        },
        "trigger_price": setup.get("trigger_price"),
    }


def _transition_record(
    event: dict[str, Any],
    *,
    signal_ts: str,
    previous_status: str | None,
    transition: str,
) -> dict[str, Any]:
    level = event.get("level") or {}
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "event_scope": event.get("event_scope"),
        "status": event.get("status"),
        "previous_status": previous_status,
        "transition": transition,
        "confidence": event.get("confidence"),
        "level_name": level.get("name"),
        "level_price": level.get("price"),
        "ts": signal_ts,
    }


def build_setup_event_lifecycle(
    signal_ts: str,
    setups: list[dict[str, Any]] | None = None,
    previous_receipt: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    previous_events = {
        str(event.get("event_id")): event
        for event in (previous_receipt or {}).get("setup_events", [])
        if isinstance(event, dict) and event.get("event_id")
    }
    current_events: dict[str, dict[str, Any]] = {}
    observed_family_keys: dict[str, str] = {}
    transitions: list[dict[str, Any]] = []

    for setup in setups or []:
        observed = _normalize_setup_event(setup)
        event_id = str(observed["event_id"])
        observed_family_keys[str(observed["family_key"])] = event_id
        previous = previous_events.get(event_id)
        previous_status = str((previous or {}).get("status") or "candidate").lower()
        if previous and previous_status in ACTIVE_EVENT_STATUSES:
            observation_count = int(previous.get("observation_count") or 1) + 1
            status = "confirmed" if observation_count >= 2 else "candidate"
            event = {
                **observed,
                "status": status,
                "first_seen_ts": previous.get("first_seen_ts") or signal_ts,
                "last_seen_ts": signal_ts,
                "last_confirmed_ts": (
                    signal_ts
                    if status == "confirmed"
                    else previous.get("last_confirmed_ts")
                ),
                "observation_count": observation_count,
            }
            if status != previous_status:
                transitions.append(
                    _transition_record(
                        event,
                        signal_ts=signal_ts,
                        previous_status=previous_status,
                        transition="promoted",
                    )
                )
        else:
            previous = None
            event = {
                **observed,
                "status": "candidate",
                "first_seen_ts": signal_ts,
                "last_seen_ts": signal_ts,
                "last_confirmed_ts": None,
                "observation_count": 1,
            }
            transitions.append(
                _transition_record(
                    event,
                    signal_ts=signal_ts,
                    previous_status=None,
                    transition="new",
                )
            )
        current_events[event_id] = event

    observed_actionable_setup = any(
        str(event.get("event_scope") or "").lower() == "setup"
        for event in current_events.values()
    )

    for event_id, previous in previous_events.items():
        if event_id in current_events:
            continue
        previous_status = str(previous.get("status") or "candidate").lower()
        if previous_status not in ACTIVE_EVENT_STATUSES:
            continue
        replacement_id = observed_family_keys.get(str(previous.get("family_key") or ""))
        previous_scope = str(previous.get("event_scope") or "").lower()
        if (
            previous_scope == "setup"
            and previous_status == "confirmed"
            and current_events
            and not observed_actionable_setup
            and not replacement_id
        ):
            current_events[event_id] = {
                **deepcopy(previous),
                "last_status_ts": signal_ts,
                "persisted_without_fresh_trigger": True,
            }
            continue
        status = (
            "invalidated"
            if replacement_id and replacement_id != event_id
            else "expired"
        )
        event = {
            **deepcopy(previous),
            "status": status,
            "last_status_ts": signal_ts,
        }
        current_events[event_id] = event
        transitions.append(
            _transition_record(
                event,
                signal_ts=signal_ts,
                previous_status=previous_status,
                transition=status,
            )
        )

    events = sorted(
        current_events.values(),
        key=lambda event: (
            _STATUS_ORDER.get(str(event.get("status") or "").lower(), 9),
            str(event.get("event_scope") or ""),
            str(event.get("event_type") or ""),
            str(event.get("event_id") or ""),
        ),
    )
    return events, transitions


def primary_setup_event(
    setup_events: list[dict[str, Any]] | None,
    setup_tag: str | None,
) -> dict[str, Any]:
    if not setup_tag:
        return {}
    active = [
        event
        for event in (setup_events or [])
        if _is_active_status(event.get("status"))
    ]
    for event in active:
        if str(event.get("event_type") or "") == str(setup_tag):
            return event
    for event in setup_events or []:
        if str(event.get("event_type") or "") == str(setup_tag):
            return event
    return {}


def primary_actionable_setup_event(
    setup_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    for event in setup_events or []:
        if (
            _is_active_status(event.get("status"))
            and str(event.get("event_scope") or "").lower() == "setup"
        ):
            return event
    return {}


def setup_dict_from_event(event: dict[str, Any] | None) -> dict[str, Any]:
    event = event or {}
    level = event.get("level") or {}
    return {
        "tag": event.get("event_type"),
        "bias": event.get("bias"),
        "kind": event.get("kind"),
        "detail": event.get("detail"),
        "score": event.get("confidence"),
        "level_name": level.get("name"),
        "level_price": level.get("price"),
        "trigger_price": event.get("trigger_price"),
    }


def annotate_setup_conviction(
    permission: dict[str, Any],
    setup_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Attach lifecycle metadata without recomputing authority.

    Contract doctrine:
    - setup-event lifecycle can preserve a persisted thesis after fresh setup
      evidence has disappeared.
    - lifecycle metadata is descriptive and presentation-facing; it does not
      re-run vector scoring or alter the already-computed authority score.
    """
    setup_conviction = (permission or {}).get("setup_conviction") or {}
    if not setup_conviction:
        return permission

    promoted = primary_actionable_setup_event(setup_events)
    current_gate = str(setup_conviction.get("setup_gate") or "NONE").upper()
    if promoted and current_gate in {"NONE", "CONTEXT"}:
        promoted_setup = setup_dict_from_event(promoted)
        confidence = int(promoted.get("confidence") or 0)
        setup_conviction.update(
            {
                "setup_conviction_score": confidence,
                "setup_gate": (
                    "ACTIONABLE"
                    if confidence >= 72
                    else "EMERGING"
                    if confidence >= 58
                    else "WATCH"
                ),
                "bias": promoted_setup.get("bias"),
                "setup_tag": promoted_setup.get("tag"),
                "reason": promoted_setup.get("detail")
                or setup_conviction.get("reason")
                or promoted_setup.get("tag"),
                "entry_gate": gate_metadata(promoted_setup),
            }
        )

    event = primary_setup_event(setup_events, setup_conviction.get("setup_tag"))
    if not event:
        return sync_setup_evidence_fields(permission)
    level = event.get("level") or {}
    persisted_without_fresh_trigger = bool(event.get("persisted_without_fresh_trigger"))
    setup_conviction["event_lifecycle"] = {
        "status": event.get("status"),
        "confidence": event.get("confidence"),
        "first_seen_ts": event.get("first_seen_ts"),
        "last_seen_ts": event.get("last_seen_ts"),
        "last_confirmed_ts": event.get("last_confirmed_ts"),
        "observation_count": event.get("observation_count"),
        "level_name": level.get("name"),
        "level_price": level.get("price"),
        "persisted_without_fresh_trigger": persisted_without_fresh_trigger,
    }
    setup_conviction["persisted_setup_thesis"] = {
        "source": "setup_event_lifecycle",
        "active": persisted_without_fresh_trigger,
        "setup_tag": setup_conviction.get("setup_tag"),
        "event_status": event.get("status"),
        "persisted_without_fresh_trigger": persisted_without_fresh_trigger,
        "first_seen_ts": event.get("first_seen_ts"),
        "last_seen_ts": event.get("last_seen_ts"),
        "last_confirmed_ts": event.get("last_confirmed_ts"),
        "observation_count": event.get("observation_count"),
    }
    return sync_setup_evidence_fields(permission)


__all__ = [
    "ACTIVE_EVENT_STATUSES",
    "annotate_setup_conviction",
    "build_setup_event_lifecycle",
    "primary_actionable_setup_event",
    "primary_setup_event",
    "setup_dict_from_event",
]
