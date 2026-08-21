#!/usr/bin/env python3
"""Deterministic paper-only hypothesis producer for eligible locked slots."""

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

from scripts.alpha_swarm.contracts import (  # noqa: E402
    CANDIDATE_SCHEMA,
    EXPLICIT_OUTCOME_FIELD,
    FORBIDDEN_OUTCOME_FIELDS,
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
    slots_by_id,
    validate_candidate,
)
from scripts.alpha_swarm.data_steward import ELIGIBILITY_SCHEMA, SNAPSHOT_SCHEMA  # noqa: E402

PUBLICATION_SCHEMA = "sharpedge.alpha_swarm.hypothesis_publication.v1"
RULE_ID = "vwap_momentum_volume_v1"
RULE_VERSION = "1.0.0"
RULE_FEATURES = ("vs_vwap_pct", "momentum_15m_pct", "volume_ratio")
DIRECTIONAL_RISK_CAP_DOLLARS = 100.0
VWAP_THRESHOLD = 0.05
MOMENTUM_THRESHOLD = 0.05
VOLUME_THRESHOLD = 1.2
FORBIDDEN_RESEARCH_FIELDS = frozenset(
    {
        *FORBIDDEN_OUTCOME_FIELDS,
        EXPLICIT_OUTCOME_FIELD,
        "vehicle",
        "contract",
        "strike",
        "expiry",
        "quantity",
        "utility",
        "performance",
        "rank",
        "confidence",
        "alpha_score",
        "score",
    }
)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _artifact_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _forbidden_feature_names(snapshot: dict[str, Any]) -> list[str]:
    features = snapshot.get("features") or {}
    return sorted(
        name
        for name in (str(key).strip().lower() for key in features)
        if name in FORBIDDEN_RESEARCH_FIELDS
    )


def _decision(features: dict[str, Any]) -> tuple[str, dict[str, float]]:
    values = {name: _safe_float(features.get(name), name) for name in RULE_FEATURES}
    if (
        values["vs_vwap_pct"] >= VWAP_THRESHOLD
        and values["momentum_15m_pct"] >= MOMENTUM_THRESHOLD
        and values["volume_ratio"] >= VOLUME_THRESHOLD
    ):
        decision = "long"
    elif (
        values["vs_vwap_pct"] <= -VWAP_THRESHOLD
        and values["momentum_15m_pct"] <= -MOMENTUM_THRESHOLD
        and values["volume_ratio"] >= VOLUME_THRESHOLD
    ):
        decision = "short"
    else:
        decision = "stand_down"
    return decision, values


def _base_publication(steward: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "schema": PUBLICATION_SCHEMA,
        "run_id": steward.get("run_id"),
        "manifest_sha256": steward.get("manifest_sha256"),
        "evaluator_source_sha256": steward.get("evaluator_source_sha256"),
        "data_steward_artifact_sha256": _artifact_sha256(steward),
        "researcher_source_sha256": _source_sha256(),
        "slot_id": steward.get("slot_id"),
        "session_date": steward.get("session_date"),
        "symbol": steward.get("symbol"),
        "published_at": now.isoformat(),
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "option_selection_allowed": False,
    }


def _validate_steward_identity(steward: dict[str, Any]) -> None:
    if steward.get("schema") != ELIGIBILITY_SCHEMA:
        raise ValueError(f"Data Steward schema must be {ELIGIBILITY_SCHEMA}")
    if steward.get("paper_only") is not True:
        raise ValueError("Data Steward artifact must remain paper_only")
    if steward.get("authoritative") is not False:
        raise ValueError("Data Steward artifact must remain non-authoritative")
    if steward.get("execution_permitted") is not False:
        raise ValueError("Data Steward artifact must remain non-executable")
    if steward.get("directional_output_allowed") is not False:
        raise ValueError("Data Steward artifact must remain direction-blind")


def _not_ready_publication(
    steward: dict[str, Any], now: datetime
) -> dict[str, Any] | None:
    state = steward.get("state")
    if state == "not_due":
        return {
            **_base_publication(steward, now),
            "state": "not_ready",
            "reason": "Data Steward eligibility window has not opened",
            "candidate": None,
        }
    if state == "ineligible":
        return {
            **_base_publication(steward, now),
            "state": "data_rejected",
            "reason": "Data Steward rejected the point-in-time evidence",
            "candidate": None,
        }
    if state != "eligible":
        raise ValueError("Data Steward state must be not_due, ineligible, or eligible")
    return None


def _validate_eligible_inputs(
    manifest: dict[str, Any],
    steward: dict[str, Any],
    snapshot: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    slots = slots_by_id(manifest)
    slot_id = str(steward.get("slot_id") or "")
    if slot_id not in slots:
        raise ValueError("Data Steward slot is outside the locked manifest")
    slot = slots[slot_id]
    locked_hash = manifest_sha256(manifest)
    expected = {
        "run_id": manifest["run_id"],
        "manifest_sha256": locked_hash,
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "symbol": slot["symbol"],
        "session_date": slot["session_date"],
        "prediction_ts": slot["prediction_ts"],
    }
    for field, value in expected.items():
        if steward.get(field) != value:
            raise ValueError(f"Data Steward {field} does not match locked manifest")
    if steward.get("eligible") is not True:
        raise ValueError("eligible Data Steward state must set eligible=true")
    if steward.get("evaluator_accounting") != "candidate_allowed":
        raise ValueError("eligible Data Steward state must allow a candidate")
    prediction = parse_timestamp(slot["prediction_ts"], "prediction_ts")
    if now != prediction:
        raise ValueError(
            "researcher publication must occur at exact locked prediction_ts"
        )

    evidence = steward.get("snapshot_evidence") or {}
    snapshot_hash = _artifact_sha256(snapshot)
    if snapshot_hash != evidence.get("snapshot_sha256"):
        raise ValueError("snapshot SHA256 does not match Data Steward evidence")
    for field in ("run_id", "manifest_sha256", "slot_id", "symbol", "session_date"):
        expected_value = steward.get(field)
        if snapshot.get(field) != expected_value:
            raise ValueError(f"snapshot {field} does not match Data Steward artifact")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("authoritative") is not False
    ):
        raise ValueError("snapshot must remain paper-only and non-authoritative")
    if snapshot.get("execution_permitted") is not False:
        raise ValueError("snapshot must remain non-executable")
    feature_available = parse_timestamp(
        snapshot.get("feature_available_ts"), "feature_available_ts"
    )
    if feature_available > prediction:
        raise ValueError("feature_available_ts cannot be after prediction_ts")
    if snapshot.get("feature_available_ts") != evidence.get("feature_available_ts"):
        raise ValueError("feature_available_ts does not match Data Steward evidence")
    features = snapshot.get("features") or {}
    if sorted(str(name) for name in features) != evidence.get("feature_names"):
        raise ValueError("snapshot feature names do not match Data Steward evidence")
    forbidden = _forbidden_feature_names(snapshot)
    if forbidden:
        raise ValueError(f"snapshot contains forbidden research fields: {forbidden}")
    missing = [name for name in RULE_FEATURES if name not in features]
    if missing:
        raise ValueError(f"snapshot is missing required rule features: {missing}")
    if list(snapshot.get("source_refs") or []) != list(
        evidence.get("source_refs") or []
    ):
        raise ValueError("snapshot source_refs do not match Data Steward evidence")
    return slot


def build_publication(
    steward: dict[str, Any],
    *,
    now: datetime,
    manifest: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_steward_identity(steward)
    early = _not_ready_publication(steward, now)
    if early is not None:
        return early
    if manifest is None or snapshot is None:
        raise ValueError(
            "eligible Data Steward input requires manifest and exact snapshot"
        )
    slot = _validate_eligible_inputs(manifest, steward, snapshot, now)
    decision, values = _decision(snapshot["features"])
    risk_cap = 0.0 if decision == "stand_down" else DIRECTIONAL_RISK_CAP_DOLLARS
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "prediction_ts": slot["prediction_ts"],
        "published_at": slot["prediction_ts"],
        "feature_available_ts": snapshot["feature_available_ts"],
        "decision": decision,
        "risk_cap_dollars": risk_cap,
        "outcome_field": EXPLICIT_OUTCOME_FIELD,
        "feature_names": list(RULE_FEATURES),
        "feature_values": values,
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "variant_index": 1,
        "variant_count": 1,
        "source_refs": [
            f"data-steward-sha256://{_artifact_sha256(steward)}",
            f"snapshot-sha256://{_artifact_sha256(snapshot)}",
            *list(snapshot.get("source_refs") or []),
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }
    candidate["candidate_id"] = payload_sha256(candidate)
    validate_candidate(manifest, candidate)
    return {
        **_base_publication(steward, now),
        "state": "candidate_published",
        "reason": "fixed v1 hypothesis rule evaluated eligible point-in-time evidence",
        "candidate": candidate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-steward-artifact", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--now")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit(
            "Hypothesis Researcher is artifact-only; --no-network is required"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only artifact: {args.output}")
    steward = json.loads(args.data_steward_artifact.read_text(encoding="utf-8"))
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    )
    snapshot = (
        json.loads(args.snapshot.read_text(encoding="utf-8")) if args.snapshot else None
    )
    now_value = args.now or steward.get("declared_at")
    now = parse_timestamp(now_value, "now")
    artifact = build_publication(
        steward,
        now=now,
        manifest=manifest,
        snapshot=snapshot,
    )
    args.output.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    print(json.dumps({"state": artifact["state"], "slot_id": artifact["slot_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
