#!/usr/bin/env python3
"""Lock and validate an additive paper-only Alpha Swarm variant family."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.alpha_swarm.contracts import (
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    validate_manifest,
)
from scripts.alpha_swarm.variant_rules import VARIANTS, rules_fingerprint

SCHEMA = "sharpedge.alpha_swarm.variant_manifest.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MANIFEST = ROOT / "outputs" / "alpha_swarm_pltr_manifest.json"
DEFAULT_BASE_INPUT_ROOT = ROOT / "outputs" / "alpha_swarm_pltr_pilot"
DEFAULT_OUTPUT = ROOT / "outputs" / "alpha_swarm_pltr_variant_manifest.json"
SOURCE_NAMES = (
    "variant_manifest.py",
    "variant_rules.py",
    "variant_equity.py",
    "variant_live_pilot.py",
)
EXPECTED_SCHEDULE = {
    "shared_evidence_attach_seconds_after_eligibility": 60,
    "candidate_publication_lateness_seconds": 20,
    "entry_capture_seconds_after_locked_time": 90,
    "exit_capture_seconds_after_locked_time": 90,
    "capture_lateness_seconds": 90,
    "receipt_publication_lateness_seconds": 20,
    "catch_up_allowed": False,
}


class VariantManifestError(ValueError):
    """Raised when the additive variant contract is invalid."""


def source_paths() -> list[Path]:
    package = Path(__file__).resolve().parent
    return [package / name for name in SOURCE_NAMES]


def _source_hashes() -> dict[str, str]:
    missing = [path.name for path in source_paths() if not path.is_file()]
    if missing:
        raise VariantManifestError(f"variant source files are missing: {missing}")
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in source_paths()}


def variant_manifest_sha256(payload: dict[str, Any]) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _first_evidence_time(base_manifest: dict[str, Any]) -> datetime:
    eligibility = min(
        parse_timestamp(slot["eligibility_declared_at"], "eligibility_declared_at")
        for slot in base_manifest["slots"]
    )
    return eligibility - timedelta(seconds=120)


def build_variant_manifest(
    *,
    run_id: str,
    locked_at: datetime,
    base_manifest: dict[str, Any],
    base_manifest_path: Path,
    base_input_root: Path,
) -> dict[str, Any]:
    validate_manifest(base_manifest)
    payload = {
        "schema": SCHEMA,
        "run_id": run_id,
        "locked_at": locked_at.isoformat(),
        "base_manifest": {
            "run_id": base_manifest["run_id"],
            "manifest_path": str(base_manifest_path),
            "manifest_sha256": manifest_sha256(base_manifest),
            "input_root": str(base_input_root),
            "universe": list(base_manifest["universe"]),
            "slot_count": len(base_manifest["slots"]),
        },
        "rules_fingerprint": rules_fingerprint(),
        "variants": list(VARIANTS),
        "variant_source_sha256": _source_hashes(),
        "paper_vehicle": "equity",
        "paper_notional_dollars": 100.0,
        "schedule": dict(EXPECTED_SCHEDULE),
        "governance": {
            "all_variants_counted": True,
            "aggregate_score_hidden_during_pilot": True,
            "adaptive_tuning_allowed": False,
            "winner_selection_allowed": False,
            "base_artifact_mutation_allowed": False,
        },
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_access_allowed": False,
        "order_actions_allowed": False,
        "aggregate_score_computed": False,
    }
    validate_variant_manifest(payload, base_manifest, verify_sources=True)
    return payload


def validate_variant_manifest(
    payload: dict[str, Any],
    base_manifest: dict[str, Any],
    *,
    verify_sources: bool,
) -> None:
    validate_manifest(base_manifest)
    if payload.get("schema") != SCHEMA:
        raise VariantManifestError(f"variant manifest schema must be {SCHEMA}")
    if not str(payload.get("run_id") or "").strip():
        raise VariantManifestError("variant run_id is required")
    locked_at = parse_timestamp(payload.get("locked_at"), "locked_at")
    if locked_at > _first_evidence_time(base_manifest):
        raise VariantManifestError(
            "variant family was locked after first evidence acquisition"
        )
    base = payload.get("base_manifest") or {}
    expected_base = {
        "run_id": base_manifest["run_id"],
        "manifest_sha256": manifest_sha256(base_manifest),
        "universe": list(base_manifest["universe"]),
        "slot_count": len(base_manifest["slots"]),
    }
    for field, expected in expected_base.items():
        if base.get(field) != expected:
            raise VariantManifestError(f"base manifest {field} does not match")
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise VariantManifestError("variants must be a non-empty list")
    count = len(variants)
    ids = [item.get("variant_id") for item in variants]
    indexes = [item.get("variant_index") for item in variants]
    if len(ids) != len(set(ids)) or any(not str(item or "").strip() for item in ids):
        raise VariantManifestError("variant IDs must be non-empty and unique")
    if sorted(indexes) != list(range(1, count + 1)):
        raise VariantManifestError(
            "variant indexes must completely cover 1..variant_count"
        )
    if any(item.get("variant_count") != count for item in variants):
        raise VariantManifestError("every variant must disclose the full variant_count")
    if (
        variants != list(VARIANTS)
        or payload.get("rules_fingerprint") != rules_fingerprint()
    ):
        raise VariantManifestError(
            "variant family differs from the frozen source registry"
        )
    if payload.get("schedule") != EXPECTED_SCHEDULE:
        raise VariantManifestError(
            "variant event schedule differs from the frozen contract"
        )
    safety = {
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_access_allowed": False,
        "order_actions_allowed": False,
        "aggregate_score_computed": False,
    }
    for field, expected in safety.items():
        if payload.get(field) is not expected:
            raise VariantManifestError(f"{field} must be {expected!r}")
    governance = payload.get("governance") or {}
    expected_governance = {
        "all_variants_counted": True,
        "aggregate_score_hidden_during_pilot": True,
        "adaptive_tuning_allowed": False,
        "winner_selection_allowed": False,
        "base_artifact_mutation_allowed": False,
    }
    for field, expected in expected_governance.items():
        if governance.get(field) is not expected:
            raise VariantManifestError(f"governance.{field} must be {expected!r}")
    if verify_sources:
        if payload.get("variant_source_sha256") != _source_hashes():
            raise VariantManifestError("variant source changed after manifest lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--base-input-root", type=Path, default=DEFAULT_BASE_INPUT_ROOT)
    parser.add_argument("--locked-at", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    payload = build_variant_manifest(
        run_id=args.run_id,
        locked_at=parse_timestamp(args.locked_at, "locked_at"),
        base_manifest=base_manifest,
        base_manifest_path=args.base_manifest,
        base_input_root=args.base_input_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "variant_manifest_sha256": variant_manifest_sha256(payload),
                "variant_count": len(payload["variants"]),
                "base_slot_count": payload["base_manifest"]["slot_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
