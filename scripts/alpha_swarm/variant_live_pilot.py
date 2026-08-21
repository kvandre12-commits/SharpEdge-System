#!/usr/bin/env python3
"""Isolated live worker for fixed paper-equity Alpha Swarm variants."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
)
from scripts.alpha_swarm.data_steward import ELIGIBILITY_SCHEMA  # noqa: E402
from scripts.alpha_swarm.live_pilot_sources import fetch_price_capture  # noqa: E402
from scripts.alpha_swarm.snapshot_acquirer import SNAPSHOT_SCHEMA  # noqa: E402
from scripts.alpha_swarm.variant_equity import (  # noqa: E402
    build_evaluation_publication,
    build_shared_capture,
)
from scripts.alpha_swarm.variant_manifest import (  # noqa: E402
    DEFAULT_BASE_MANIFEST,
    DEFAULT_OUTPUT as DEFAULT_VARIANT_MANIFEST,
    validate_variant_manifest,
    variant_manifest_sha256,
)
from scripts.alpha_swarm.variant_rules import VARIANTS, build_publication  # noqa: E402

STATE_SCHEMA = "sharpedge.alpha_swarm.variant_worker_state.v1"
EVENT_SCHEMA = "sharpedge.alpha_swarm.variant_event_receipt.v1"
EVIDENCE_SCHEMA = "sharpedge.alpha_swarm.shared_variant_evidence.v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "alpha_swarm_pltr_variant_pilot"
HEARTBEAT_SECONDS = 5
POLL_SECONDS = 0.5
DEPENDENCIES = {
    "publish_variants": ("attach_evidence",),
    "publish_receipts": ("publish_variants", "capture_entry", "capture_exit"),
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(payload) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing append-only artifact differs: {path}") from None


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _slot_root(output_root: Path, slot: dict[str, Any]) -> Path:
    return output_root / slot["session_date"] / slot["slot_id"]


def _base_slot_root(base_root: Path, slot: dict[str, Any]) -> Path:
    return base_root / slot["session_date"] / slot["slot_id"]


def _paths(output_root: Path, slot: dict[str, Any]) -> dict[str, Path]:
    root = _slot_root(output_root, slot)
    return {
        "evidence": root / "shared" / "evidence_ref.json",
        "entry": root / "shared" / "entry_equity_capture.json",
        "exit": root / "shared" / "exit_equity_capture.json",
        "variants": root / "variants",
    }


def event_schedule(
    variant_manifest: dict[str, Any], base_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_variant_manifest(variant_manifest, base_manifest, verify_sources=True)
    schedule = variant_manifest["schedule"]
    events = []
    for slot in base_manifest["slots"]:
        eligibility = parse_timestamp(
            slot["eligibility_declared_at"], "eligibility_declared_at"
        )
        prediction = parse_timestamp(slot["prediction_ts"], "prediction_ts")
        entry = parse_timestamp(slot["entry_ts"], "entry_ts")
        exit_time = parse_timestamp(slot["exit_ts"], "exit_ts")
        label = parse_timestamp(slot["label_available_ts"], "label_available_ts")
        specs = (
            (
                eligibility
                + timedelta(
                    seconds=schedule["shared_evidence_attach_seconds_after_eligibility"]
                ),
                "attach_evidence",
                schedule["capture_lateness_seconds"],
            ),
            (
                prediction,
                "publish_variants",
                schedule["candidate_publication_lateness_seconds"],
            ),
            (
                entry
                + timedelta(
                    seconds=schedule["entry_capture_seconds_after_locked_time"]
                ),
                "capture_entry",
                schedule["capture_lateness_seconds"],
            ),
            (
                exit_time
                + timedelta(seconds=schedule["exit_capture_seconds_after_locked_time"]),
                "capture_exit",
                schedule["capture_lateness_seconds"],
            ),
            (
                label,
                "publish_receipts",
                schedule["receipt_publication_lateness_seconds"],
            ),
        )
        for scheduled_at, action, tolerance in specs:
            events.append(
                {
                    "event_id": f"{slot['slot_id']}:{action}",
                    "scheduled_at": scheduled_at,
                    "action": action,
                    "tolerance_seconds": tolerance,
                    "slot": slot,
                }
            )
    return sorted(events, key=lambda item: (item["scheduled_at"], item["event_id"]))


def build_evidence_ref(
    *,
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    slot: dict[str, Any],
    snapshot: dict[str, Any],
    steward: dict[str, Any],
    attached_at: str,
) -> dict[str, Any]:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("base snapshot schema is invalid")
    if (
        steward.get("schema") != ELIGIBILITY_SCHEMA
        or steward.get("state") != "eligible"
    ):
        raise ValueError("base Data Steward did not publish eligible evidence")
    if steward.get("eligible") is not True:
        raise ValueError("base Data Steward eligibility must be true")
    expected = {
        "run_id": base_manifest["run_id"],
        "manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
    }
    for field, value in expected.items():
        if snapshot.get(field) != value or steward.get(field) != value:
            raise ValueError(f"shared evidence {field} differs from the base slot")
    snapshot_hash = payload_sha256(snapshot)
    evidence = steward.get("snapshot_evidence") or {}
    if evidence.get("snapshot_sha256") != snapshot_hash:
        raise ValueError("Data Steward snapshot hash does not match shared evidence")
    if parse_timestamp(
        snapshot["feature_available_ts"], "feature_available_ts"
    ) > parse_timestamp(slot["prediction_ts"], "prediction_ts"):
        raise ValueError("shared feature evidence became available after prediction")
    if parse_timestamp(attached_at, "attached_at") > parse_timestamp(
        slot["prediction_ts"], "prediction_ts"
    ):
        raise ValueError("shared evidence cannot attach after prediction")
    return {
        "schema": EVIDENCE_SCHEMA,
        "variant_run_id": variant_manifest["run_id"],
        "variant_manifest_sha256": variant_manifest_sha256(variant_manifest),
        "base_run_id": base_manifest["run_id"],
        "base_manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "attached_at": attached_at,
        "prediction_ts": slot["prediction_ts"],
        "feature_available_ts": snapshot["feature_available_ts"],
        "feature_names": sorted(str(name) for name in snapshot["features"]),
        "snapshot_sha256": snapshot_hash,
        "data_steward_sha256": payload_sha256(steward),
        "source_refs": list(snapshot.get("source_refs") or []),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
    }


def _attach_evidence(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    event: dict[str, Any],
    output_root: Path,
    base_root: Path,
) -> None:
    slot = event["slot"]
    base = _base_slot_root(base_root, slot)
    snapshot = read_json(base / "research_snapshot.json")
    steward = read_json(base / "phase2_eligibility.json")
    payload = build_evidence_ref(
        variant_manifest=variant_manifest,
        base_manifest=base_manifest,
        slot=slot,
        snapshot=snapshot,
        steward=steward,
        attached_at=event["scheduled_at"].isoformat(),
    )
    write_once(_paths(output_root, slot)["evidence"], payload)


def _publish_variants(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    event: dict[str, Any],
    output_root: Path,
    base_root: Path,
) -> None:
    slot = event["slot"]
    paths = _paths(output_root, slot)
    base = _base_slot_root(base_root, slot)
    evidence = read_json(paths["evidence"])
    snapshot = read_json(base / "research_snapshot.json")
    steward = read_json(base / "phase2_eligibility.json")
    if payload_sha256(snapshot) != evidence["snapshot_sha256"]:
        raise ValueError("shared snapshot changed after evidence attachment")
    if payload_sha256(steward) != evidence["data_steward_sha256"]:
        raise ValueError("Data Steward artifact changed after evidence attachment")
    locked_hash = variant_manifest_sha256(variant_manifest)
    for variant in VARIANTS:
        publication = build_publication(
            base_manifest=base_manifest,
            slot=slot,
            snapshot=snapshot,
            steward=steward,
            evidence_ref=evidence,
            variant=variant,
            variant_manifest_sha256=locked_hash,
            observed_at=event["scheduled_at"].isoformat(),
        )
        write_once(
            paths["variants"] / variant["variant_id"] / "candidate.json",
            publication,
        )


def _capture_equity(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    event: dict[str, Any],
    output_root: Path,
) -> None:
    slot = event["slot"]
    phase = "entry" if event["action"] == "capture_entry" else "exit"
    observed_at = utc_now()
    provider = fetch_price_capture(
        slot["symbol"], slot["session_date"], observed_at=observed_at
    )
    shared = build_shared_capture(
        base_manifest=base_manifest,
        slot=slot,
        phase=phase,
        provider_capture=provider,
        variant_manifest_sha256=variant_manifest_sha256(variant_manifest),
    )
    write_once(_paths(output_root, slot)[phase], shared)


def _publish_receipts(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    event: dict[str, Any],
    output_root: Path,
) -> None:
    slot = event["slot"]
    paths = _paths(output_root, slot)
    entry = read_json(paths["entry"])
    exit_capture = read_json(paths["exit"])
    for variant in VARIANTS:
        root = paths["variants"] / variant["variant_id"]
        publication = read_json(root / "candidate.json")
        evaluation = build_evaluation_publication(
            base_manifest=base_manifest,
            candidate_publication=publication,
            entry_capture=entry,
            exit_capture=exit_capture,
            published_at=event["scheduled_at"].isoformat(),
        )
        write_once(root / "evaluation.json", evaluation)


def _run_action(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    event: dict[str, Any],
    output_root: Path,
    base_root: Path,
) -> None:
    action = event["action"]
    if action == "attach_evidence":
        _attach_evidence(variant_manifest, base_manifest, event, output_root, base_root)
    elif action == "publish_variants":
        _publish_variants(
            variant_manifest, base_manifest, event, output_root, base_root
        )
    elif action in {"capture_entry", "capture_exit"}:
        _capture_equity(variant_manifest, base_manifest, event, output_root)
    elif action == "publish_receipts":
        _publish_receipts(variant_manifest, base_manifest, event, output_root)
    else:
        raise ValueError(f"unknown variant action: {action}")


def _dependency_failure(
    event: dict[str, Any], state: dict[str, Any]
) -> tuple[str, str] | None:
    slot_id = event["slot"]["slot_id"]
    for action in DEPENDENCIES.get(event["action"], ()):
        event_id = f"{slot_id}:{action}"
        receipt = state["events"].get(event_id)
        if receipt is None or receipt.get("status") != "completed":
            return event_id, receipt.get("status", "missing") if receipt else "missing"
    return None


def _event_receipt(
    event: dict[str, Any], observed_at: datetime, status: str, error: str | None = None
) -> dict[str, Any]:
    payload = {
        "schema": EVENT_SCHEMA,
        "event_id": event["event_id"],
        "slot_id": event["slot"]["slot_id"],
        "symbol": event["slot"]["symbol"],
        "action": event["action"],
        "scheduled_at": event["scheduled_at"].isoformat(),
        "observed_at": observed_at.isoformat(),
        "lateness_seconds": round(
            (observed_at - event["scheduled_at"]).total_seconds(), 6
        ),
        "status": status,
        "error": error,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "aggregate_score_computed": False,
    }
    return payload


def _record_event(
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    event: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    safe_name = event["event_id"].replace(":", "__") + ".json"
    write_once(output_root / "events" / safe_name, receipt)
    state["events"][event["event_id"]] = receipt
    state["updated_at"] = receipt["observed_at"]
    write_state(state_path, state)


def run_worker(
    variant_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    now_fn=utc_now,
    sleep_fn=time.sleep,
    once: bool = False,
) -> dict[str, Any]:
    validate_variant_manifest(variant_manifest, base_manifest, verify_sources=True)
    base_root = Path(variant_manifest["base_manifest"]["input_root"])
    schedule = event_schedule(variant_manifest, base_manifest)
    state_path = output_root / "worker_state.json"
    locked_hash = variant_manifest_sha256(variant_manifest)
    if state_path.exists():
        state = read_json(state_path)
        if state.get("variant_manifest_sha256") != locked_hash:
            raise ValueError("worker state belongs to a different variant manifest")
    else:
        state = {
            "schema": STATE_SCHEMA,
            "run_id": variant_manifest["run_id"],
            "variant_manifest_sha256": locked_hash,
            "base_manifest_sha256": manifest_sha256(base_manifest),
            "started_at": now_fn().isoformat(),
            "events": {},
            "paper_only": True,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "aggregate_score_computed": False,
        }
    state["pid"] = os.getpid()
    state["updated_at"] = now_fn().isoformat()
    write_state(state_path, state)
    last_heartbeat = now_fn()
    while True:
        now = now_fn()
        if (now - last_heartbeat).total_seconds() >= HEARTBEAT_SECONDS:
            state["heartbeat_at"] = now.isoformat()
            state["updated_at"] = now.isoformat()
            write_state(state_path, state)
            last_heartbeat = now
        due = [
            item
            for item in schedule
            if item["event_id"] not in state["events"] and item["scheduled_at"] <= now
        ]
        for event in due:
            observed = now_fn()
            dependency = _dependency_failure(event, state)
            if dependency:
                event_id, status = dependency
                receipt = _event_receipt(
                    event, observed, "blocked", f"dependency {event_id} is {status}"
                )
            elif (observed - event["scheduled_at"]).total_seconds() > event[
                "tolerance_seconds"
            ]:
                receipt = _event_receipt(
                    event,
                    observed,
                    "missed",
                    "event exceeded locked lateness tolerance",
                )
            else:
                try:
                    _run_action(
                        variant_manifest,
                        base_manifest,
                        event,
                        output_root,
                        base_root,
                    )
                    receipt = _event_receipt(event, now_fn(), "completed")
                except Exception as exc:
                    receipt = _event_receipt(
                        event, now_fn(), "failed", f"{type(exc).__name__}: {exc}"
                    )
            _record_event(output_root, state_path, state, event, receipt)
        if once:
            return state
        final_time = schedule[-1]["scheduled_at"] + timedelta(
            seconds=schedule[-1]["tolerance_seconds"]
        )
        if now > final_time:
            return state
        future = [
            item["scheduled_at"]
            for item in schedule
            if item["event_id"] not in state["events"] and item["scheduled_at"] > now
        ]
        wait = (
            min(POLL_SECONDS, max(0.01, (min(future) - now).total_seconds()))
            if future
            else POLL_SECONDS
        )
        sleep_fn(wait)


def status(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state_path = output_root / "worker_state.json"
    if not state_path.exists():
        return {"state": "not_started", "output_root": str(output_root)}
    state = read_json(state_path)
    counts: dict[str, int] = {}
    for event in state.get("events", {}).values():
        event_status = event.get("status", "unknown")
        counts[event_status] = counts.get(event_status, 0) + 1
    pid = int(state.get("pid") or 0)
    alive = pid > 0 and Path(f"/proc/{pid}").exists()
    return {
        "state": "running" if alive else "stopped",
        "pid": pid or None,
        "alive": alive,
        "event_counts": counts,
        "heartbeat_at": state.get("heartbeat_at"),
        "variant_count": len(VARIANTS),
        "paper_only": True,
        "execution_permitted": False,
        "aggregate_score_computed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "tick"))
    parser.add_argument(
        "--variant-manifest", type=Path, default=DEFAULT_VARIANT_MANIFEST
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "status":
        print(json.dumps(status(args.output_root), indent=2))
        return 0
    variant_manifest = read_json(args.variant_manifest)
    base_manifest = read_json(args.base_manifest)
    run_worker(
        variant_manifest,
        base_manifest,
        output_root=args.output_root,
        once=args.command == "tick",
    )
    print(json.dumps(status(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
