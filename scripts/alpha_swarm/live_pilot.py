#!/usr/bin/env python3
"""Bounded Yahoo/CBOE live-data worker for the paper-only Alpha Swarm pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
    validate_manifest,
)
from scripts.alpha_swarm.data_steward import build_eligibility  # noqa: E402
from scripts.alpha_swarm.hypothesis_researcher import (  # noqa: E402
    build_publication as build_hypothesis,
)
from scripts.alpha_swarm.options_expression_agent import (  # noqa: E402
    build_publication as build_expression,
)
from scripts.alpha_swarm.paper_mark_receipt_publisher import (  # noqa: E402
    MARK_SNAPSHOT_SCHEMA,
    build_publication as build_receipt,
)
from scripts.alpha_swarm.skeptic_veto_agent import build_review  # noqa: E402
from scripts.alpha_swarm.live_pilot_sources import (  # noqa: E402
    fetch_options_capture,
    fetch_price_capture,
)
from scripts.alpha_swarm.snapshot_acquirer import (  # noqa: E402
    build_option_snapshot,
    build_research_snapshot,
    payload_sha256 as capture_sha256,
)

OUTPUT_ROOT = ROOT / "outputs" / "alpha_swarm_pilot"
MANIFEST_PATH = ROOT / "outputs" / "alpha_swarm_phase1_manifest.json"
STATE_SCHEMA = "sharpedge.alpha_swarm.live_worker_state.v1"
EVENT_SCHEMA = "sharpedge.alpha_swarm.live_event_receipt.v1"
MAX_PUBLICATION_LATENESS_SECONDS = 20
MAX_PREFETCH_LATENESS_SECONDS = 90
POLL_SECONDS = 0.5
PREFETCH_RESEARCH_SECONDS = 120
PREFETCH_QUOTE_SECONDS = 30
HEARTBEAT_SECONDS = 5
DEPENDENCIES = {
    "publish_hypothesis": ("publish_eligibility",),
    "option_prefetch": ("publish_hypothesis",),
    "publish_expression_review": ("option_prefetch",),
    "entry_prefetch": ("publish_expression_review",),
    "exit_prefetch": ("publish_expression_review",),
    "publish_receipt": (
        "publish_expression_review",
        "entry_prefetch",
        "exit_prefetch",
    ),
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def artifact_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _slot_root(slot: dict[str, Any], output_root: Path) -> Path:
    return output_root / slot["session_date"] / slot["slot_id"]


def event_schedule(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    events = []
    for slot in manifest["slots"]:
        due = parse_timestamp(
            slot["eligibility_declared_at"], "eligibility_declared_at"
        )
        prediction = parse_timestamp(slot["prediction_ts"], "prediction_ts")
        expression = prediction + timedelta(minutes=1)
        entry = parse_timestamp(slot["entry_ts"], "entry_ts")
        exit_time = parse_timestamp(slot["exit_ts"], "exit_ts")
        label = parse_timestamp(slot["label_available_ts"], "label_available_ts")
        specs = (
            (due - timedelta(seconds=PREFETCH_RESEARCH_SECONDS), "research_prefetch"),
            (due, "publish_eligibility"),
            (prediction, "publish_hypothesis"),
            (expression - timedelta(seconds=PREFETCH_QUOTE_SECONDS), "option_prefetch"),
            (expression, "publish_expression_review"),
            (entry - timedelta(seconds=PREFETCH_QUOTE_SECONDS), "entry_prefetch"),
            (exit_time - timedelta(seconds=PREFETCH_QUOTE_SECONDS), "exit_prefetch"),
            (label, "publish_receipt"),
        )
        for scheduled, action in specs:
            events.append(
                {
                    "event_id": f"{slot['slot_id']}:{action}",
                    "scheduled_at": scheduled,
                    "action": action,
                    "slot": slot,
                }
            )
    return sorted(events, key=lambda event: (event["scheduled_at"], event["event_id"]))


def _phase_paths(root: Path) -> dict[str, Path]:
    return {
        "price": root / "price_capture.json",
        "research_options": root / "research_options_capture.json",
        "snapshot": root / "research_snapshot.json",
        "phase2": root / "phase2_eligibility.json",
        "phase3": root / "phase3_hypothesis.json",
        "option_capture": root / "option_capture.json",
        "option_snapshot": root / "option_snapshot.json",
        "phase4": root / "phase4_expression.json",
        "phase5": root / "phase5_review.json",
        "entry": root / "entry_quote.json",
        "exit": root / "exit_quote.json",
        "phase6": root / "phase6_receipt.json",
    }


def _quote_payload(
    capture: dict[str, Any], expression: dict[str, Any]
) -> dict[str, Any]:
    by_symbol = {item["contract_symbol"]: item for item in capture["contracts"]}
    long_symbol = expression["long_leg"]["contract_symbol"]
    short_symbol = expression["short_leg"]["contract_symbol"]
    if long_symbol not in by_symbol or short_symbol not in by_symbol:
        raise ValueError("selected expression contracts are absent from quote capture")
    return {
        "long": by_symbol[long_symbol],
        "short": by_symbol[short_symbol],
        "source": {
            "provider": capture["provider"],
            "source_sha256": capture_sha256(capture),
            "source_ref": capture["source_ref"],
        },
    }


def _run_action(
    manifest: dict[str, Any], event: dict[str, Any], output_root: Path
) -> str:
    slot = event["slot"]
    action = event["action"]
    scheduled = event["scheduled_at"]
    root = _slot_root(slot, output_root)
    paths = _phase_paths(root)
    if action == "research_prefetch":
        observed = utc_now()
        price = fetch_price_capture(
            slot["symbol"], slot["session_date"], observed_at=observed
        )
        options = fetch_options_capture(
            slot["symbol"], slot["session_date"], observed_at=utc_now()
        )
        write_once(paths["price"], price)
        write_once(paths["research_options"], options)
        snapshot = build_research_snapshot(
            manifest,
            slot_id=slot["slot_id"],
            captured_at=parse_timestamp(
                slot["eligibility_declared_at"], "eligibility_declared_at"
            ),
            price_capture=price,
            options_capture=options,
        )
        write_once(paths["snapshot"], snapshot)
    elif action == "publish_eligibility":
        snapshot = read_json(paths["snapshot"]) if paths["snapshot"].exists() else None
        artifact = build_eligibility(
            manifest, now=scheduled, slot_id=slot["slot_id"], snapshot=snapshot
        )
        write_once(paths["phase2"], artifact)
    elif action == "publish_hypothesis":
        phase2 = read_json(paths["phase2"])
        snapshot = read_json(paths["snapshot"]) if paths["snapshot"].exists() else None
        artifact = build_hypothesis(
            phase2, now=scheduled, manifest=manifest, snapshot=snapshot
        )
        write_once(paths["phase3"], artifact)
    elif action == "option_prefetch":
        phase3 = read_json(paths["phase3"])
        candidate = phase3.get("candidate") or {}
        if candidate.get("decision") in {"long", "short"}:
            capture = fetch_options_capture(
                slot["symbol"], slot["session_date"], observed_at=utc_now()
            )
            write_once(paths["option_capture"], capture)
            option_snapshot = build_option_snapshot(
                manifest,
                slot_id=slot["slot_id"],
                captured_at=parse_timestamp(slot["prediction_ts"], "prediction_ts")
                + timedelta(minutes=1),
                options_capture=capture,
            )
            write_once(paths["option_snapshot"], option_snapshot)
    elif action == "publish_expression_review":
        phase3 = read_json(paths["phase3"])
        option_snapshot = (
            read_json(paths["option_snapshot"])
            if paths["option_snapshot"].exists()
            else None
        )
        phase4 = build_expression(
            phase3, now=scheduled, manifest=manifest, option_snapshot=option_snapshot
        )
        write_once(paths["phase4"], phase4)
        phase5 = build_review(
            phase3,
            phase4,
            now=scheduled,
            manifest=manifest,
            option_snapshot=option_snapshot,
        )
        write_once(paths["phase5"], phase5)
    elif action in {"entry_prefetch", "exit_prefetch"}:
        phase5 = read_json(paths["phase5"])
        if phase5.get("state") == "paper_expression_accepted":
            phase4 = read_json(paths["phase4"])
            capture = fetch_options_capture(
                slot["symbol"], slot["session_date"], observed_at=utc_now()
            )
            quote = _quote_payload(capture, phase4["expression"])
            quote["capture"] = capture
            write_once(paths["entry" if action == "entry_prefetch" else "exit"], quote)
    elif action == "publish_receipt":
        phase3 = read_json(paths["phase3"])
        phase4 = read_json(paths["phase4"])
        phase5 = read_json(paths["phase5"])
        option_snapshot = (
            read_json(paths["option_snapshot"])
            if paths["option_snapshot"].exists()
            else None
        )
        mark_snapshot = None
        if phase5.get("state") == "paper_expression_accepted":
            entry = read_json(paths["entry"])
            exit_quote = read_json(paths["exit"])
            expression = phase4["expression"]
            mark_snapshot = {
                "schema": MARK_SNAPSHOT_SCHEMA,
                "run_id": manifest["run_id"],
                "manifest_sha256": manifest_sha256(manifest),
                "slot_id": slot["slot_id"],
                "symbol": slot["symbol"],
                "session_date": slot["session_date"],
                "candidate_sha256": payload_sha256(phase3["candidate"]),
                "expression_sha256": artifact_sha256(expression),
                "captured_at": slot["label_available_ts"],
                "entry": {
                    "observed_at": slot["entry_ts"],
                    "long_contract_symbol": expression["long_leg"]["contract_symbol"],
                    "short_contract_symbol": expression["short_leg"]["contract_symbol"],
                    "long_ask": entry["long"]["ask"],
                    "short_bid": entry["short"]["bid"],
                    "source": entry["source"],
                },
                "exit": {
                    "observed_at": slot["exit_ts"],
                    "long_contract_symbol": expression["long_leg"]["contract_symbol"],
                    "short_contract_symbol": expression["short_leg"]["contract_symbol"],
                    "long_bid": exit_quote["long"]["bid"],
                    "short_ask": exit_quote["short"]["ask"],
                    "source": exit_quote["source"],
                },
                "paper_only": True,
                "authoritative": False,
                "execution_permitted": False,
            }
        artifact = build_receipt(
            phase3,
            phase4,
            phase5,
            now=scheduled,
            manifest=manifest,
            option_snapshot=option_snapshot,
            mark_snapshot=mark_snapshot,
        )
        write_once(paths["phase6"], artifact)
    else:
        raise ValueError(f"unsupported live action: {action}")
    return "completed"


def _event_receipt(
    event: dict[str, Any], observed: datetime, lateness: float, status: str
) -> dict[str, Any]:
    return {
        "schema": EVENT_SCHEMA,
        "event_id": event["event_id"],
        "scheduled_at": event["scheduled_at"].isoformat(),
        "observed_at": observed.isoformat(),
        "lateness_seconds": round(lateness, 3),
        "status": status,
        "paper_only": True,
        "execution_permitted": False,
        "aggregate_score_computed": False,
    }


def _record_event(
    output_root: Path,
    state: dict[str, Any],
    event: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    event_id = event["event_id"]
    event_path = output_root / "events" / f"{event_id.replace(':', '__')}.json"
    write_once(event_path, receipt)
    state["events"][event_id] = receipt


def _blocked_dependency(
    event: dict[str, Any], state: dict[str, Any]
) -> tuple[str, str] | None:
    slot_id = event["slot"]["slot_id"]
    for action in DEPENDENCIES.get(event["action"], ()):
        dependency_id = f"{slot_id}:{action}"
        receipt = state["events"].get(dependency_id)
        if receipt is None or receipt.get("status") != "completed":
            status = receipt.get("status") if receipt else "missing"
            return dependency_id, status
    return None


def _sleep_until_next(
    schedule: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
    sleep_fn,
) -> None:
    future = [
        event["scheduled_at"]
        for event in schedule
        if event["event_id"] not in state["events"] and event["scheduled_at"] > now
    ]
    wait = (
        min(POLL_SECONDS, max(0.01, (min(future) - now).total_seconds()))
        if future
        else POLL_SECONDS
    )
    sleep_fn(wait)


def run_worker(
    manifest: dict[str, Any],
    *,
    output_root: Path = OUTPUT_ROOT,
    now_fn=utc_now,
    sleep_fn=time.sleep,
    once: bool = False,
) -> dict[str, Any]:
    schedule = event_schedule(manifest)
    state_path = output_root / "worker_state.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else {
            "schema": STATE_SCHEMA,
            "run_id": manifest["run_id"],
            "manifest_sha256": manifest_sha256(manifest),
            "started_at": now_fn().isoformat(),
            "events": {},
            "paper_only": True,
            "execution_permitted": False,
            "aggregate_score_computed": False,
        }
    )
    state["updated_at"] = now_fn().isoformat()
    state["pid"] = os.getpid()
    write_state(state_path, state)
    prefetch_actions = {
        "research_prefetch",
        "option_prefetch",
        "entry_prefetch",
        "exit_prefetch",
    }
    last_heartbeat = now_fn()
    while True:
        now = now_fn()
        if (now - last_heartbeat).total_seconds() >= HEARTBEAT_SECONDS:
            state["heartbeat_at"] = now.isoformat()
            state["updated_at"] = now.isoformat()
            write_state(state_path, state)
            last_heartbeat = now
        due = [
            event
            for event in schedule
            if event["event_id"] not in state["events"] and event["scheduled_at"] <= now
        ]
        runnable = []
        for event in due:
            blocked = _blocked_dependency(event, state)
            if blocked:
                dependency_id, dependency_status = blocked
                receipt = _event_receipt(event, now, 0.0, "blocked")
                receipt["error"] = (
                    f"dependency {dependency_id} is {dependency_status}; no downstream action"
                )
                _record_event(output_root, state, event, receipt)
                continue
            lateness = (now - event["scheduled_at"]).total_seconds()
            tolerance = (
                MAX_PREFETCH_LATENESS_SECONDS
                if event["action"] in prefetch_actions
                else MAX_PUBLICATION_LATENESS_SECONDS
            )
            if lateness <= tolerance:
                runnable.append(event)
                continue
            receipt = _event_receipt(event, now, lateness, "missed")
            receipt["error"] = "event exceeded locked lateness tolerance"
            _record_event(output_root, state, event, receipt)
        if runnable:
            with ThreadPoolExecutor(max_workers=min(6, len(runnable))) as executor:
                futures = {
                    executor.submit(_run_action, manifest, event, output_root): event
                    for event in runnable
                }
                for future in as_completed(futures):
                    event = futures[future]
                    observed = now_fn()
                    lateness = (observed - event["scheduled_at"]).total_seconds()
                    try:
                        status = future.result()
                        receipt = _event_receipt(event, observed, lateness, status)
                    except Exception as exc:
                        receipt = _event_receipt(event, observed, lateness, "failed")
                        receipt["error"] = f"{type(exc).__name__}: {exc}"
                    _record_event(output_root, state, event, receipt)
        if due:
            state["updated_at"] = now_fn().isoformat()
            write_state(state_path, state)
        final_time = schedule[-1]["scheduled_at"] + timedelta(
            seconds=MAX_PUBLICATION_LATENESS_SECONDS
        )
        if now > final_time or once:
            return state
        _sleep_until_next(schedule, state, now_fn(), sleep_fn)


def status(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    state_path = output_root / "worker_state.json"
    if not state_path.exists():
        return {"state": "not_started", "output_root": str(output_root)}
    state = read_json(state_path)
    counts: dict[str, int] = {}
    for event in state.get("events", {}).values():
        key = event.get("status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    pid = int(state.get("pid") or 0)
    alive = pid > 0 and Path(f"/proc/{pid}").exists()
    return {
        "state": "running" if alive else "stopped",
        "pid": pid or None,
        "alive": alive,
        "event_counts": counts,
        "updated_at": state.get("updated_at"),
        "paper_only": True,
        "execution_permitted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "tick"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "status":
        print(json.dumps(status(args.output_root), indent=2))
        return 0
    manifest = read_json(args.manifest)
    run_worker(manifest, output_root=args.output_root, once=args.command == "tick")
    print(json.dumps(status(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
