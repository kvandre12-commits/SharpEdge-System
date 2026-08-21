#!/usr/bin/env python3
"""Deterministic one-shot lifecycle planner for the paper-only Alpha Swarm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    validate_manifest,
)

PLAN_SCHEMA = "sharpedge.alpha_swarm.operational_plan.v1"
EVENT_SCHEMA = "sharpedge.alpha_swarm.operational_event.v1"
STAGES = (
    ("eligibility_declared_at", 10, "acquire_research_snapshot", "always"),
    ("eligibility_declared_at", 20, "publish_data_eligibility", "always"),
    ("prediction_ts", 30, "publish_hypothesis", "eligible_only"),
    ("expression_ts", 40, "acquire_option_snapshot", "directional_candidate_only"),
    ("expression_ts", 50, "publish_options_expression", "candidate_only"),
    ("expression_ts", 60, "publish_skeptic_review", "phase4_publication_exists"),
    ("entry_ts", 70, "acquire_entry_mark", "accepted_expression_only"),
    ("exit_ts", 80, "acquire_exit_mark", "accepted_expression_only"),
    ("label_available_ts", 90, "publish_evaluation_receipt", "phase5_terminal"),
)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _event_time(slot: dict[str, Any], field: str) -> datetime:
    if field == "expression_ts":
        return parse_timestamp(slot["prediction_ts"], "prediction_ts") + timedelta(
            minutes=1
        )
    return parse_timestamp(slot[field], field)


def _event(
    manifest: dict[str, Any],
    slot: dict[str, Any],
    *,
    field: str,
    order: int,
    action: str,
    condition: str,
) -> dict[str, Any]:
    scheduled = _event_time(slot, field)
    event_id = f"{slot['slot_id']}:{order:02d}:{action}"
    artifact_root = (
        f"outputs/alpha_swarm_pilot/{slot['session_date']}/{slot['slot_id']}"
    )
    return {
        "schema": EVENT_SCHEMA,
        "event_id": event_id,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "scheduled_at": scheduled.isoformat(),
        "stage_order": order,
        "action": action,
        "condition": condition,
        "artifact_root": artifact_root,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "aggregate_score_computed": False,
    }


def build_schedule(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand immutable manifest timestamps into a stable lifecycle schedule."""
    validate_manifest(manifest)
    events = [
        _event(
            manifest,
            slot,
            field=field,
            order=order,
            action=action,
            condition=condition,
        )
        for slot in manifest["slots"]
        for field, order, action, condition in STAGES
    ]
    return sorted(
        events,
        key=lambda event: (
            parse_timestamp(event["scheduled_at"], "scheduled_at"),
            event["stage_order"],
            event["slot_id"],
        ),
    )


def build_plan(
    manifest: dict[str, Any],
    *,
    now: datetime,
    completed_event_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Classify one tick. Past events are missed, never silently replayed."""
    completed = completed_event_ids or set()
    scheduled = build_schedule(manifest)
    planned = []
    counts = {"completed": 0, "due": 0, "missed": 0, "pending": 0}
    for event in scheduled:
        timestamp = parse_timestamp(event["scheduled_at"], "scheduled_at")
        if event["event_id"] in completed:
            status = "completed"
        elif timestamp < now:
            status = "missed"
        elif timestamp == now:
            status = "due"
        else:
            status = "pending"
        counts[status] += 1
        planned.append({**event, "status": status})
    due = [event for event in planned if event["status"] == "due"]
    pending = [event for event in planned if event["status"] == "pending"]
    next_time = pending[0]["scheduled_at"] if pending else None
    next_events = (
        [event for event in pending if event["scheduled_at"] == next_time]
        if next_time
        else []
    )
    return {
        "schema": PLAN_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "orchestrator_source_sha256": _source_sha256(),
        "planned_at": now.isoformat(),
        "mode": "one_shot_plan_only",
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "background_scheduler_started": False,
        "aggregate_score_computed": False,
        "catch_up_allowed": False,
        "event_counts": counts,
        "due_events": due,
        "next_scheduled_at": next_time,
        "next_events": next_events,
        "events": planned,
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--now", required=True)
    parser.add_argument("--completed-events", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit("operational planning requires --no-network")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    completed: set[str] = set()
    if args.completed_events:
        payload = json.loads(args.completed_events.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise SystemExit("--completed-events must contain a JSON string list")
        completed = set(payload)
    plan = build_plan(
        manifest,
        now=parse_timestamp(args.now, "now"),
        completed_event_ids=completed,
    )
    write_once(args.output, plan)
    print(
        json.dumps(
            {
                "event_counts": plan["event_counts"],
                "next_scheduled_at": plan["next_scheduled_at"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
