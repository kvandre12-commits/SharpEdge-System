#!/usr/bin/env python3
"""Veto-only deterministic replay firewall for Phase 4 option expressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import canonical_json, parse_timestamp  # noqa: E402
from scripts.alpha_swarm.hypothesis_researcher import (  # noqa: E402
    PUBLICATION_SCHEMA as PHASE3_SCHEMA,
)
from scripts.alpha_swarm.options_expression_agent import (  # noqa: E402
    PUBLICATION_SCHEMA as PHASE4_SCHEMA,
    build_publication as replay_phase4,
)

PUBLICATION_SCHEMA = "sharpedge.alpha_swarm.skeptic_review.v1"
FORBIDDEN_FIELDS = frozenset(
    {
        "utility",
        "performance",
        "score",
        "alpha_score",
        "rank",
        "ret_1d",
        "return_prediction_to_exit",
        "broker",
        "route",
        "order_id",
    }
)


def _artifact_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_forbidden(value: Any, *, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_FIELDS:
                failures.append(f"forbidden field {path}.{key}")
            failures.extend(_walk_forbidden(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden(child, path=f"{path}[{index}]"))
    return failures


def _base(
    phase3: dict[str, Any], phase4: dict[str, Any], now: datetime
) -> dict[str, Any]:
    return {
        "schema": PUBLICATION_SCHEMA,
        "run_id": phase4.get("run_id"),
        "manifest_sha256": phase4.get("manifest_sha256"),
        "evaluator_source_sha256": phase4.get("evaluator_source_sha256"),
        "slot_id": phase4.get("slot_id"),
        "session_date": phase4.get("session_date"),
        "symbol": phase4.get("symbol"),
        "reviewed_at": now.isoformat(),
        "phase3_publication_sha256": _artifact_sha256(phase3),
        "phase4_publication_sha256": _artifact_sha256(phase4),
        "skeptic_source_sha256": _source_sha256(Path(__file__)),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "can_modify_expression": False,
    }


def _phase4_failures(
    phase3: dict[str, Any], phase4: dict[str, Any], now: datetime
) -> list[str]:
    failures: list[str] = []
    if phase3.get("schema") != PHASE3_SCHEMA:
        failures.append(f"Phase 3 schema must be {PHASE3_SCHEMA}")
    if phase4.get("schema") != PHASE4_SCHEMA:
        failures.append(f"Phase 4 schema must be {PHASE4_SCHEMA}")
    for name, payload in (("Phase 3", phase3), ("Phase 4", phase4)):
        if payload.get("paper_only") is not True:
            failures.append(f"{name} paper_only must be true")
        if payload.get("authoritative") is not False:
            failures.append(f"{name} authoritative must be false")
        if payload.get("execution_permitted") is not False:
            failures.append(f"{name} execution_permitted must be false")
    if phase4.get("broker_action_allowed") is not False:
        failures.append("Phase 4 broker_action_allowed must be false")
    if phase4.get("phase3_publication_sha256") != _artifact_sha256(phase3):
        failures.append("Phase 4 does not reference the exact Phase 3 publication")
    expected_phase4_source = _source_sha256(
        Path(__file__).with_name("options_expression_agent.py")
    )
    if phase4.get("agent_source_sha256") != expected_phase4_source:
        failures.append("Phase 4 agent source SHA256 does not match current source")
    for field in (
        "run_id",
        "manifest_sha256",
        "evaluator_source_sha256",
        "slot_id",
        "session_date",
        "symbol",
    ):
        if phase4.get(field) != phase3.get(field):
            failures.append(f"Phase 4 {field} does not match Phase 3")
    failures.extend(_walk_forbidden(phase3, path="phase3"))
    failures.extend(_walk_forbidden(phase4, path="phase4"))
    try:
        expression_at = parse_timestamp(phase4.get("expression_at"), "expression_at")
        if now != expression_at:
            failures.append("skeptic review must occur at exact Phase 4 expression_at")
    except ValueError as exc:
        failures.append(str(exc))
    return sorted(set(failures))


def _veto(
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    now: datetime,
    reasons: list[str],
) -> dict[str, Any]:
    review = {
        **_base(phase3, phase4, now),
        "state": "vetoed",
        "verdict": "veto",
        "evaluator_accounting": "zero_utility_rejection",
        "reasons": sorted(set(reasons)),
        "accepted_expression_sha256": None,
    }
    review["review_id"] = _artifact_sha256(review)
    return review


def build_review(
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    *,
    now: datetime,
    manifest: dict[str, Any] | None = None,
    option_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = _phase4_failures(phase3, phase4, now)
    if failures:
        return _veto(phase3, phase4, now, failures)

    state = phase4.get("state")
    no_snapshot_states = {"upstream_not_ready", "upstream_data_rejected", "abstained"}
    if state in no_snapshot_states and option_snapshot is not None:
        return _veto(
            phase3,
            phase4,
            now,
            [f"{state} must not consume an option snapshot"],
        )
    try:
        replay = replay_phase4(
            phase3,
            now=now,
            manifest=manifest,
            option_snapshot=option_snapshot,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _veto(phase3, phase4, now, [f"deterministic replay failed: {exc}"])
    if canonical_json(replay) != canonical_json(phase4):
        return _veto(
            phase3,
            phase4,
            now,
            ["Phase 4 publication does not match deterministic replay"],
        )

    mapping = {
        "upstream_not_ready": ("upstream_not_ready", None, "none"),
        "upstream_data_rejected": (
            "vetoed",
            "veto",
            "zero_utility_rejection",
        ),
        "abstained": ("abstained", None, "stand_down"),
        "no_valid_expression": (
            "vetoed",
            "veto",
            "zero_utility_rejection",
        ),
        "expression_published": (
            "paper_expression_accepted",
            "accept",
            "candidate_accepted_for_paper_mark",
        ),
    }
    if state not in mapping:
        return _veto(phase3, phase4, now, [f"unsupported Phase 4 state: {state}"])
    review_state, verdict, accounting = mapping[state]
    expression = phase4.get("expression")
    accepted_hash = _artifact_sha256(expression) if verdict == "accept" else None
    review = {
        **_base(phase3, phase4, now),
        "state": review_state,
        "verdict": verdict,
        "evaluator_accounting": accounting,
        "reasons": [],
        "accepted_expression_sha256": accepted_hash,
    }
    review["review_id"] = _artifact_sha256(review)
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-publication", required=True, type=Path)
    parser.add_argument("--phase4-publication", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--option-snapshot", type=Path)
    parser.add_argument("--now")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit(
            "Skeptic/Veto Agent is artifact-only; --no-network is required"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only artifact: {args.output}")
    phase3 = json.loads(args.phase3_publication.read_text(encoding="utf-8"))
    phase4 = json.loads(args.phase4_publication.read_text(encoding="utf-8"))
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    )
    snapshot = (
        json.loads(args.option_snapshot.read_text(encoding="utf-8"))
        if args.option_snapshot
        else None
    )
    now = parse_timestamp(args.now or phase4.get("expression_at"), "now")
    review = build_review(
        phase3,
        phase4,
        now=now,
        manifest=manifest,
        option_snapshot=snapshot,
    )
    args.output.write_text(canonical_json(review) + "\n", encoding="utf-8")
    print(json.dumps({"state": review["state"], "verdict": review["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
