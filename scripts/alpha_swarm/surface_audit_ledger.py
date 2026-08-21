"""Append-only recommendation-change ledger for the surface audit supervisor."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "sharpedge.surface_audit_recommendation_event.v1"


class LedgerError(ValueError):
    """Raised when an existing recommendation ledger cannot be trusted."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def recommendation_material(recommendation: dict[str, Any]) -> dict[str, Any]:
    """Exclude volatile observations from a material recommendation identity."""
    return {
        "recommendation_key": recommendation["recommendation_key"],
        "rule_id": recommendation["rule_id"],
        "category": recommendation["category"],
        "severity": recommendation["severity"],
        "why_code": recommendation["why_code"],
        "steps": recommendation.get("steps", []),
        "validation": recommendation.get("validation", []),
        "guardrails": recommendation.get("guardrails", []),
    }


def recommendation_fingerprint(recommendation: dict[str, Any]) -> str:
    return sha256(
        canonical_json(recommendation_material(recommendation)).encode()
    ).hexdigest()


def recommendation_set_fingerprint(recommendations: list[dict[str, Any]]) -> str:
    fingerprints = sorted(recommendation_fingerprint(item) for item in recommendations)
    return sha256(canonical_json(fingerprints).encode()).hexdigest()


def _event_hash(event: dict[str, Any]) -> str:
    material = {key: value for key, value in event.items() if key != "event_hash"}
    return sha256(canonical_json(material).encode()).hexdigest()


def read_verified_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink():
        raise LedgerError("recommendation ledger must not be a symlink")
    events = []
    previous_hash = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"invalid ledger JSON at line {line_number}") from exc
        if event.get("schema") != LEDGER_SCHEMA:
            raise LedgerError(f"invalid ledger schema at line {line_number}")
        if event.get("previous_event_hash") != previous_hash:
            raise LedgerError(f"broken ledger chain at line {line_number}")
        if event.get("event_hash") != _event_hash(event):
            raise LedgerError(f"invalid ledger event hash at line {line_number}")
        events.append(event)
        previous_hash = event["event_hash"]
    return events


def plan_event(
    recommendations: list[dict[str, Any]],
    previous_event: dict[str, Any] | None,
    generated_at: datetime,
) -> dict[str, Any] | None:
    current = {
        item["recommendation_key"]: {
            **recommendation_material(item),
            "fingerprint": recommendation_fingerprint(item),
        }
        for item in recommendations
    }
    previous = {
        item["recommendation_key"]: item
        for item in (previous_event or {}).get("active_recommendations", [])
    }
    set_fingerprint = recommendation_set_fingerprint(recommendations)
    if previous_event and previous_event.get("set_fingerprint") == set_fingerprint:
        return None
    opened = sorted(current.keys() - previous.keys())
    resolved = sorted(previous.keys() - current.keys())
    persistent = sorted(
        key
        for key in current.keys() & previous.keys()
        if current[key]["fingerprint"] == previous[key].get("fingerprint")
    )
    superseded = sorted(
        key
        for key in current.keys() & previous.keys()
        if current[key]["fingerprint"] != previous[key].get("fingerprint")
    )
    event = {
        "schema": LEDGER_SCHEMA,
        "event_id": f"surface-audit-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}",
        "generated_at": generated_at.isoformat(),
        "previous_set_fingerprint": (previous_event or {}).get("set_fingerprint"),
        "set_fingerprint": set_fingerprint,
        "opened": opened,
        "persistent": persistent,
        "superseded": superseded,
        "resolved": resolved,
        "active_recommendations": [current[key] for key in sorted(current)],
        "previous_event_hash": (previous_event or {}).get("event_hash"),
    }
    event["event_hash"] = _event_hash(event)
    return event


def append_if_changed(
    path: Path,
    recommendations: list[dict[str, Any]],
    generated_at: datetime,
) -> dict[str, Any]:
    events = read_verified_events(path)
    previous = events[-1] if events else None
    event = plan_event(recommendations, previous, generated_at)
    if event is None:
        return {
            "verified": True,
            "changed": False,
            "event_id": (previous or {}).get("event_id"),
            "set_fingerprint": (previous or {}).get("set_fingerprint"),
            "transitions": {"opened": [], "superseded": [], "resolved": []},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise LedgerError("recommendation ledger must not be a symlink")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "verified": True,
        "changed": True,
        "event_id": event["event_id"],
        "event_hash": event["event_hash"],
        "set_fingerprint": event["set_fingerprint"],
        "transitions": {
            "opened": event["opened"],
            "persistent": event["persistent"],
            "superseded": event["superseded"],
            "resolved": event["resolved"],
        },
    }
