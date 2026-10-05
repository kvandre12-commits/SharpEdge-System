"""Durable chart markers for setup observations and confirmations.

The cockpit refreshes constantly, but important setup events should not vanish
when the fresh-entry window closes. This module stores observed events and then
attaches execution confirmation separately when higher-level evidence supports
trading them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MARKER_SCHEMA = "sharpedge.setup_markers.v1"
MARKABLE_EVENTS = {
    "DOWNSIDE EXHAUSTION",
    "UPSIDE EXHAUSTION",
    "FAILED BREAKDOWN",
    "FAILED BREAKOUT",
    "EXHAUSTION -> RUNNER HANDOFF",
}
MARKABLE_CANDIDATE_EVENTS = {
    "DOWNSIDE EXHAUSTION",
    "FAILED BREAKDOWN",
    "FAILED BREAKOUT",
}
FAILED_BREAK_EVENTS = {"FAILED BREAKDOWN", "FAILED BREAKOUT"}
FAILED_BREAK_EXECUTION_BIAS = {
    "FAILED BREAKDOWN": "CALLS",
    "FAILED BREAKOUT": "PUTS",
}
TRAP_CONFIRMATION_SCORE = 70


def load_setup_markers(
    path: Path, session_date: str | None = None
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    markers = payload.get("markers") if isinstance(payload, dict) else []
    if not isinstance(markers, list):
        return []
    out = [m for m in markers if isinstance(m, dict)]
    if session_date:
        out = [m for m in out if _marker_session_date(m) == session_date]
    return out


def _write_setup_markers(path: Path, markers: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": MARKER_SCHEMA, "markers": markers}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _receipt_by_ts(receipts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(receipt.get("ts")): receipt
        for receipt in receipts
        if isinstance(receipt, dict) and receipt.get("ts")
    }


def _price_for_event(event: dict[str, Any], receipt: dict[str, Any]) -> float | None:
    event_type = str(event.get("event_type") or "")
    level = event.get("level") or {}
    level_price = level.get("price")
    if event_type in {"FAILED BREAKDOWN", "FAILED BREAKOUT"} and isinstance(
        level_price, (int, float)
    ):
        return float(level_price)
    spot = receipt.get("spot")
    return float(spot) if isinstance(spot, (int, float)) else None


def _normalized_bias(value: Any) -> str:
    text = str(value or "").upper()
    if "CALL" in text or "BULL" in text or "RECLAIM" in text:
        return "CALLS"
    if "PUT" in text or "BEAR" in text or "REJECT" in text:
        return "PUTS"
    return "NEUTRAL"


def _execution_confirmation_for_event(
    receipt: dict[str, Any], event_type: str
) -> dict[str, Any] | None:
    if event_type not in FAILED_BREAK_EVENTS:
        return None
    conviction = receipt.get("setup_conviction") or {}
    if not isinstance(conviction, dict):
        conviction = {}
    trap = conviction.get("live_trap_corroboration") or {}
    if not isinstance(trap, dict):
        trap = {}
    score = int(trap.get("trap_score") or 0)
    bias = _normalized_bias(trap.get("trap_bias"))
    required_bias = FAILED_BREAK_EXECUTION_BIAS[event_type]
    confirmed = score >= TRAP_CONFIRMATION_SCORE and bias == required_bias
    return {
        "source": "setup_conviction.live_trap_corroboration.trap_score",
        "confirmed": confirmed,
        "score": score,
        "bias": bias,
        "required_bias": required_bias,
        "threshold": TRAP_CONFIRMATION_SCORE,
        "reason": str(trap.get("trap_reason") or ""),
    }


def _marker_color(event_type: str, status: str = "confirmed") -> str:
    # Failed-break color is directional/contextual, not lifecycle. Generic
    # candidates remain amber because they are not confirmed market facts yet.
    if str(status).lower() == "candidate":
        return "#d29922"
    if event_type in {"FAILED BREAKDOWN", "DOWNSIDE EXHAUSTION"}:
        return "#26a641"
    if event_type in {"FAILED BREAKOUT", "UPSIDE EXHAUSTION"}:
        return "#f85149"
    if event_type == "EXHAUSTION -> RUNNER HANDOFF":
        return "#58a6ff"
    return "#7d8590"


def _slug(text: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in str(text)]
    return "".join(cleaned).strip("_") or "event"


def _event_id(event_type: str, level_name: Any = None, level_price: Any = None) -> str:
    parts = [_slug(event_type)]
    if level_name:
        parts.append(_slug(str(level_name)))
    if isinstance(level_price, (int, float)):
        parts.append(f"{level_price:.2f}")
    return ":".join(parts)


def _date_from_ts(value: Any) -> str:
    text = str(value or "")
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return ""


def _marker_session_date(marker: dict[str, Any]) -> str:
    # Session ownership is a market-data fact, not a wall-clock timestamp fact.
    # Event timestamps can cross midnight while Yahoo is still replaying the
    # prior trading session. Prefer the explicit receipt/marker session date.
    return str(marker.get("session_date") or "") or _date_from_ts(marker.get("ts"))


def _session_date_source(receipt: dict[str, Any]) -> str:
    return str(receipt.get("session_date_source") or "legacy")


def _setup_conviction_event(receipt: dict[str, Any]) -> dict[str, Any] | None:
    conviction = receipt.get("setup_conviction") or {}
    if not isinstance(conviction, dict):
        return None
    event_type = str(conviction.get("setup_tag") or "")
    if event_type not in MARKABLE_EVENTS:
        return None
    lifecycle = conviction.get("event_lifecycle") or {}
    thesis = conviction.get("persisted_setup_thesis") or {}
    status = str(lifecycle.get("status") or thesis.get("event_status") or "").lower()
    if status not in {"confirmed", "candidate"}:
        return None
    if status == "candidate" and event_type not in MARKABLE_CANDIDATE_EVENTS:
        return None
    level_name = lifecycle.get("level_name")
    level_price = lifecycle.get("level_price")
    return {
        "event_id": _event_id(event_type, level_name, level_price),
        "event_type": event_type,
        "status": status,
        "last_confirmed_ts": (
            lifecycle.get("last_confirmed_ts")
            or thesis.get("last_confirmed_ts")
            or lifecycle.get("last_seen_ts")
            or thesis.get("last_seen_ts")
            or receipt.get("ts")
        ),
        "bias": conviction.get("bias"),
        "detail": conviction.get("reason"),
        "confidence": conviction.get("setup_conviction_score"),
        "level": {"name": level_name, "price": level_price},
        "event_detected": lifecycle.get("event_detected"),
        "event_age_bars": lifecycle.get("event_age_bars"),
        "entry_window_open": lifecycle.get("entry_window_open"),
    }


def _marker_has_execution_confirmation(marker: dict[str, Any], event_type: str) -> bool:
    confirmation = marker.get("execution_confirmation") or {}
    if not isinstance(confirmation, dict):
        return False
    return (
        bool(confirmation.get("confirmed"))
        and int(confirmation.get("score") or 0) >= TRAP_CONFIRMATION_SCORE
        and _normalized_bias(confirmation.get("bias"))
        == FAILED_BREAK_EXECUTION_BIAS[event_type]
    )


def _prune_unconfirmed_failed_break_markers(
    markers: list[dict[str, Any]], session_date: str
) -> list[dict[str, Any]]:
    # Historical name retained for compatibility with the caller. We no longer
    # prune mechanically observed failed breaks just because execution evidence
    # is not strong enough. Observation and execution confirmation are separate.
    return markers


def _confirmed_marker_event(
    receipt: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any] | None:
    event_type = str(event.get("event_type") or "")
    confirmation = _execution_confirmation_for_event(receipt, event_type)
    if confirmation:
        status = "execution_confirmed" if confirmation["confirmed"] else "observed"
        return {
            **event,
            "execution_confirmation": confirmation,
            "setup_confirmation_status": status,
        }
    return event


def _marker_status(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    if event_type in FAILED_BREAK_EVENTS:
        confirmation = event.get("execution_confirmation") or {}
        if isinstance(confirmation, dict) and confirmation.get("confirmed"):
            return "confirmed"
        return "observed"
    return str(event.get("status") or "confirmed").lower()


def _normalized_existing_marker(marker: dict[str, Any]) -> dict[str, Any]:
    event_type = str(marker.get("event_type") or "")
    status = _marker_status(marker)
    normalized = {
        **marker,
        "status": status,
        "color": _marker_color(event_type, status),
    }
    if event_type in FAILED_BREAK_EVENTS and not normalized.get(
        "setup_confirmation_status"
    ):
        normalized["setup_confirmation_status"] = (
            "execution_confirmed" if status == "confirmed" else "observed"
        )
    return normalized


def _marker_events(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in receipt.get("setup_events") or []:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type") or "")
        if event_type not in MARKABLE_EVENTS:
            continue
        status = str(event.get("status") or "").lower()
        if status == "confirmed" or (
            status == "candidate" and event_type in MARKABLE_CANDIDATE_EVENTS
        ):
            confirmed_event = _confirmed_marker_event(receipt, event)
            if confirmed_event:
                events.append(confirmed_event)
    conviction_event = _setup_conviction_event(receipt)
    if conviction_event:
        confirmed_event = _confirmed_marker_event(receipt, conviction_event)
        if confirmed_event:
            events.append(confirmed_event)
    return events


def update_setup_markers(
    path: Path,
    *,
    decision_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Persist newly confirmed setup markers and return today's marker list.

    A marker is written when a markable event reaches confirmed status. If the
    marker file is introduced mid-session, confirmed events can be backfilled
    from the receipt's last_confirmed_ts and the matching receipt spot.
    """
    session_date = str(decision_receipt.get("session_date") or "")
    require_verified_source = _session_date_source(decision_receipt) == "price_source"
    markers = [
        _normalized_existing_marker(marker)
        for marker in _prune_unconfirmed_failed_break_markers(
            load_setup_markers(path), session_date
        )
    ]
    seen = {str(marker.get("marker_id")) for marker in markers}
    receipts = list(prior_receipts or []) + [decision_receipt]
    receipts_by_ts = _receipt_by_ts(receipts)

    for source_receipt in receipts:
        if str(source_receipt.get("session_date") or "") != session_date:
            continue
        if (
            source_receipt is not decision_receipt
            and require_verified_source
            and _session_date_source(source_receipt) != "price_source"
        ):
            continue
        for event in _marker_events(source_receipt):
            event_type = str(event.get("event_type") or "")
            event_id = str(event.get("event_id") or event_type)
            marker_status = _marker_status(event)
            marker_ts = str(
                event.get("last_confirmed_ts")
                or event.get("last_seen_ts")
                or source_receipt.get("ts")
            )
            marker_session_date = str(
                source_receipt.get("session_date") or session_date
            )
            marker_session_date = marker_session_date or _date_from_ts(marker_ts)
            marker_id = f"{marker_session_date}:{event_id}:{marker_status}"
            if marker_id in seen:
                continue
            receipt = receipts_by_ts.get(marker_ts) or source_receipt
            marker_price = _price_for_event(event, receipt)
            if marker_price is None:
                continue
            marker = {
                "marker_id": marker_id,
                "session_date": marker_session_date,
                "session_date_source": _session_date_source(source_receipt),
                "event_id": event_id,
                "event_type": event_type,
                "status": marker_status,
                "ts": marker_ts,
                "price": marker_price,
                "spot": receipt.get("spot"),
                "bias": event.get("bias"),
                "detail": event.get("detail"),
                "confidence": event.get("confidence"),
                "event_detected": event.get("event_detected"),
                "event_age_bars": event.get("event_age_bars"),
                "entry_window_open": event.get("entry_window_open"),
                "setup_confirmation_status": event.get("setup_confirmation_status"),
                "color": _marker_color(event_type, marker_status),
            }
            if event.get("execution_confirmation"):
                marker["execution_confirmation"] = event["execution_confirmation"]
            markers.append(marker)
            seen.add(marker_id)

    markers = [_normalized_existing_marker(marker) for marker in markers]
    markers = sorted(markers, key=lambda marker: str(marker.get("ts") or ""))
    _write_setup_markers(path, markers)
    todays_markers = load_setup_markers(path, session_date=session_date)
    if require_verified_source:
        todays_markers = [
            marker
            for marker in todays_markers
            if _session_date_source(marker) == "price_source"
        ]
    if todays_markers:
        return todays_markers
    latest_marker_date = ""
    for marker in reversed(markers):
        latest_marker_date = _marker_session_date(marker)
        if latest_marker_date:
            break
    if latest_marker_date:
        return load_setup_markers(path, session_date=latest_marker_date)
    return []


__all__ = ["load_setup_markers", "update_setup_markers"]
