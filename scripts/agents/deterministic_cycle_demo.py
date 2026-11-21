#!/usr/bin/env python3
"""Run one deterministic protective cycle with an authority-free learning step."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.compounding_agent import build_candidate
from scripts.agents.disagreement_adjudicator import (
    POST_FILL,
    PRE_FILL,
    adjudicate,
    build_outcome_receipt,
)
from scripts.agents.receipt_utils import canonical_json, payload_sha256

SCHEMA = "sharpedge.deterministic_cycle_demo.v1"
FIXTURE_TIME = "2026-11-19T15:00:00+00:00"


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_id"] = payload_sha256(result)
    return result


def frozen_fixture() -> dict[str, Any]:
    """Return synthetic point-in-time evidence for a reproducible demo."""
    return {
        "schema": "sharpedge.deterministic_cycle_fixture.v1",
        "as_of": FIXTURE_TIME,
        "symbol": "SPY",
        "synthetic": True,
        "observer_available": False,
        "position": {
            "verified": True,
            "contract": "SPY-DEMO-CALL",
            "quantity": 1,
            "entry_debit_dollars": 125.0,
            "latest_return_pct": -5.0,
            "entry_thesis_id": "demo-thesis-001",
        },
        "settled_outcome": {
            "actual_return_pct": -5.0,
            "counterfactual_return_pct": -20.0,
            "resolution_minutes": 3.5,
        },
    }


def run_cycle(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run independently recoverable components and return one canonical commit."""
    evidence = fixture or frozen_fixture()
    evidence_hash = payload_sha256(evidence)
    cycle_id = f"demo-{evidence_hash[:16]}"

    observer = _receipt(
        {
            "schema": "sharpedge.cycle_component.v1",
            "cycle_id": cycle_id,
            "component": "observer",
            "status": "failed",
            "fresh": False,
            "decision": "abstain",
            "hard_blocks": ["observer_fixture_unavailable"],
            "thesis_status": "unknown",
            "error_code": "OBSERVER_UNAVAILABLE",
        }
    )
    helios = _receipt(
        {
            "schema": "sharpedge.cycle_component.v1",
            "cycle_id": cycle_id,
            "component": "helios",
            "status": "abstained",
            "decision": "abstain",
            "reasons": ["observer evidence is unavailable"],
        }
    )
    steward = _receipt(
        {
            "schema": "sharpedge.cycle_component.v1",
            "cycle_id": cycle_id,
            "component": "position_steward",
            "status": "completed",
            "verified_position": evidence["position"]["verified"],
            "decision": "exit_review",
            "reasons": [
                "protect verified position while entry-analysis lane is degraded",
                "current observer evidence is unavailable",
            ],
            "position_ref": payload_sha256(evidence["position"]),
        }
    )

    pre_fill = adjudicate(phase=PRE_FILL, observer=observer, helios=helios)
    post_fill = adjudicate(
        phase=POST_FILL,
        observer=observer,
        helios=helios,
        owner=steward,
    )
    outcome = build_outcome_receipt(post_fill, **evidence["settled_outcome"])

    cycle_core = {
        "schema": SCHEMA,
        "cycle_id": cycle_id,
        "parent_cycle_id": None,
        "as_of": evidence["as_of"],
        "fixture": {"synthetic": True, "sha256": evidence_hash},
        "components": {
            "observer": observer,
            "helios": helios,
            "position_steward": steward,
        },
        "adjudication": {"pre_fill": pre_fill, "post_fill": post_fill},
        "team_result": {
            "status": "PROTECTIVE_ONLY",
            "new_entry_eligible": False,
            "existing_position_reviewed": True,
            "resolution": post_fill["resolution"],
        },
        "outcome": outcome,
        "authority": {
            "authoritative": False,
            "paper_only": True,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "approval_decision_unchanged": True,
        },
    }
    cycle_receipt_id = payload_sha256(cycle_core)
    compounding = build_candidate(
        cycle_receipt_id=cycle_receipt_id,
        adjudication=post_fill,
        outcome=outcome,
    )
    manifest = {
        **cycle_core,
        "compounding_agent": compounding,
        "commit": {
            "status": "COMMITTED",
            "atomic_record": True,
            "cycle_receipt_id": cycle_receipt_id,
        },
    }
    manifest["receipt_id"] = payload_sha256(manifest)
    return manifest


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish the canonical projection with one same-filesystem rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json(payload) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/deterministic_cycle_demo.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_cycle()
    write_atomic(args.output, payload)
    summary = {
        "cycle_id": payload["cycle_id"],
        "team_status": payload["team_result"]["status"],
        "resolution": payload["team_result"]["resolution"],
        "compounding": payload["compounding_agent"]["disposition"],
        "receipt_id": payload["receipt_id"],
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
