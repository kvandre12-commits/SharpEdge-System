#!/usr/bin/env python3
"""Direction-blind point-in-time evidence gate for locked alpha-swarm slots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.contracts import (  # noqa: E402
    FORBIDDEN_OUTCOME_FIELDS,
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    slots_by_id,
    validate_manifest,
)

SNAPSHOT_SCHEMA = "sharpedge.alpha_swarm.point_in_time_snapshot.v1"
ELIGIBILITY_SCHEMA = "sharpedge.alpha_swarm.data_eligibility.v1"
FORBIDDEN_FIELD_NAMES = frozenset({"long", "short", "bias", "score"})
MIN_BAR_COUNT = 10
MIN_OPTION_CONTRACT_COUNT = 100
MAX_PRICE_AGE_MINUTES = 20.0
MAX_OPTIONS_AGE_MINUTES = 45.0
MAX_SPOT_DIVERGENCE_PCT = 2.0
NY = ZoneInfo("America/New_York")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _minutes_between(older: datetime, newer: datetime) -> float:
    return (newer - older).total_seconds() / 60.0


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _walk_forbidden_fields(value: Any, *, path: str = "snapshot") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in FORBIDDEN_FIELD_NAMES
                or normalized in FORBIDDEN_OUTCOME_FIELDS
            ):
                failures.append(f"forbidden field {path}.{key}")
            failures.extend(_walk_forbidden_fields(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_walk_forbidden_fields(child, path=f"{path}[{index}]"))
    return failures


def _parse_or_record(
    value: Any,
    field: str,
    failures: list[str],
) -> datetime | None:
    try:
        return parse_timestamp(value, field)
    except ValueError as exc:
        failures.append(str(exc))
        return None


def _source_failures(
    source: dict[str, Any],
    *,
    name: str,
    captured_at: datetime,
    session_date: str,
    max_age_minutes: float,
) -> tuple[list[str], datetime | None]:
    failures: list[str] = []
    if not str(source.get("provider") or "").strip():
        failures.append(f"{name}.provider is required")
    source_hash = str(source.get("source_sha256") or "")
    if not SHA256_RE.fullmatch(source_hash):
        failures.append(f"{name}.source_sha256 must be a lowercase SHA-256 digest")
    latest = _parse_or_record(
        source.get("latest_data_ts"), f"{name}.latest_data_ts", failures
    )
    if latest is not None:
        if latest > captured_at:
            failures.append(f"{name}.latest_data_ts cannot be after captured_at")
        age = _minutes_between(latest, captured_at)
        if age > max_age_minutes:
            failures.append(f"{name} is stale at {age:.1f} minutes")
        if latest.astimezone(NY).date().isoformat() != session_date:
            failures.append(f"{name} latest data is outside the locked session")
    return failures, latest


def _assess_snapshot(
    manifest: dict[str, Any],
    slot: dict[str, Any],
    snapshot: dict[str, Any],
    declared_at: datetime,
) -> tuple[list[str], dict[str, Any]]:
    failures = _walk_forbidden_fields(snapshot)
    locked_hash = manifest_sha256(manifest)
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        failures.append(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    if snapshot.get("run_id") != manifest.get("run_id"):
        failures.append("snapshot run_id does not match manifest")
    if snapshot.get("manifest_sha256") != locked_hash:
        failures.append("snapshot manifest_sha256 does not match locked manifest")
    if snapshot.get("slot_id") != slot.get("slot_id"):
        failures.append("snapshot slot_id does not match locked slot")
    if snapshot.get("symbol") != slot.get("symbol"):
        failures.append("snapshot symbol does not match locked slot")
    if snapshot.get("session_date") != slot.get("session_date"):
        failures.append("snapshot session_date does not match locked slot")
    if snapshot.get("paper_only") is not True:
        failures.append("snapshot paper_only must be true")
    if snapshot.get("authoritative") is not False:
        failures.append("snapshot authoritative must be false")
    if snapshot.get("execution_permitted") is not False:
        failures.append("snapshot execution_permitted must be false")

    captured_at = _parse_or_record(snapshot.get("captured_at"), "captured_at", failures)
    feature_available = _parse_or_record(
        snapshot.get("feature_available_ts"), "feature_available_ts", failures
    )
    if captured_at is not None:
        if captured_at > declared_at:
            failures.append("captured_at cannot be after eligibility declaration")
        if feature_available is not None and feature_available > captured_at:
            failures.append("feature_available_ts cannot be after captured_at")

    features = snapshot.get("features")
    if not isinstance(features, dict) or not features:
        failures.append("features must be a non-empty direction-neutral object")
    if not snapshot.get("source_refs"):
        failures.append("snapshot source_refs are required")

    price = snapshot.get("price_source")
    options = snapshot.get("options_source")
    if not isinstance(price, dict):
        failures.append("price_source is required")
        price = {}
    if not isinstance(options, dict):
        failures.append("options_source is required")
        options = {}
    price_latest = None
    options_latest = None
    if captured_at is not None:
        price_errors, price_latest = _source_failures(
            price,
            name="price_source",
            captured_at=captured_at,
            session_date=str(slot["session_date"]),
            max_age_minutes=MAX_PRICE_AGE_MINUTES,
        )
        option_errors, options_latest = _source_failures(
            options,
            name="options_source",
            captured_at=captured_at,
            session_date=str(slot["session_date"]),
            max_age_minutes=MAX_OPTIONS_AGE_MINUTES,
        )
        failures.extend(price_errors)
        failures.extend(option_errors)

    bar_count = int(_safe_float(price.get("bar_count")) or 0)
    contract_count = int(_safe_float(options.get("contract_count")) or 0)
    price_spot = _safe_float(price.get("spot"))
    options_spot = _safe_float(options.get("spot"))
    if bar_count < MIN_BAR_COUNT:
        failures.append(f"price_source.bar_count must be at least {MIN_BAR_COUNT}")
    if contract_count < MIN_OPTION_CONTRACT_COUNT:
        failures.append(
            f"options_source.contract_count must be at least {MIN_OPTION_CONTRACT_COUNT}"
        )
    if price_spot is None or price_spot <= 0:
        failures.append("price_source.spot must be positive")
    if options_spot is None or options_spot <= 0:
        failures.append("options_source.spot must be positive")
    spot_divergence = None
    if price_spot and options_spot:
        spot_divergence = abs(price_spot - options_spot) / price_spot * 100.0
        if spot_divergence > MAX_SPOT_DIVERGENCE_PCT:
            failures.append(
                f"price/options spot divergence {spot_divergence:.2f}% exceeds policy"
            )

    evidence = {
        "snapshot_sha256": hashlib.sha256(
            canonical_json(snapshot).encode("utf-8")
        ).hexdigest(),
        "captured_at": snapshot.get("captured_at"),
        "feature_available_ts": snapshot.get("feature_available_ts"),
        "feature_names": sorted(str(name) for name in (features or {})),
        "source_refs": list(snapshot.get("source_refs") or []),
        "price": {
            "provider": price.get("provider"),
            "source_sha256": price.get("source_sha256"),
            "latest_data_ts": price.get("latest_data_ts"),
            "bar_count": bar_count,
            "spot": price_spot,
        },
        "options": {
            "provider": options.get("provider"),
            "source_sha256": options.get("source_sha256"),
            "latest_data_ts": options.get("latest_data_ts"),
            "contract_count": contract_count,
            "spot": options_spot,
        },
        "spot_divergence_pct": round(spot_divergence, 4)
        if spot_divergence is not None
        else None,
        "latest_source_times_present": price_latest is not None
        and options_latest is not None,
    }
    return sorted(set(failures)), evidence


def select_slot(
    manifest: dict[str, Any],
    now: datetime,
    slot_id: str | None = None,
) -> dict[str, Any]:
    slots = slots_by_id(manifest)
    if slot_id:
        if slot_id not in slots:
            raise ValueError("slot_id is not present in the locked manifest")
        return slots[slot_id]
    ordered = sorted(
        slots.values(),
        key=lambda slot: (
            parse_timestamp(slot["eligibility_declared_at"], "eligibility_declared_at"),
            str(slot["slot_id"]),
        ),
    )
    for slot in ordered:
        if parse_timestamp(slot["prediction_ts"], "prediction_ts") >= now:
            return slot
    raise ValueError("no remaining locked slot is available for this timestamp")


def build_eligibility(
    manifest: dict[str, Any],
    *,
    now: datetime,
    slot_id: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest)
    slot = select_slot(manifest, now, slot_id)
    due = parse_timestamp(slot["eligibility_declared_at"], "eligibility_declared_at")
    prediction = parse_timestamp(slot["prediction_ts"], "prediction_ts")
    base = {
        "schema": ELIGIBILITY_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "producer_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "declared_at": now.isoformat(),
        "eligibility_due_at": slot["eligibility_declared_at"],
        "prediction_ts": slot["prediction_ts"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "directional_output_allowed": False,
    }
    if now < due:
        return {
            **base,
            "state": "not_due",
            "eligible": None,
            "evaluator_accounting": "none",
            "reasons": ["eligibility declaration window has not opened"],
        }
    if now > prediction:
        return {
            **base,
            "state": "ineligible",
            "eligible": False,
            "evaluator_accounting": "zero_utility_rejection",
            "reasons": [
                "eligibility declaration missed the locked prediction deadline"
            ],
        }
    if snapshot is None:
        return {
            **base,
            "state": "ineligible",
            "eligible": False,
            "evaluator_accounting": "zero_utility_rejection",
            "reasons": [
                "point-in-time snapshot is missing after eligibility became due"
            ],
        }

    failures, evidence = _assess_snapshot(manifest, slot, snapshot, now)
    if failures:
        return {
            **base,
            "state": "ineligible",
            "eligible": False,
            "evaluator_accounting": "zero_utility_rejection",
            "reasons": failures,
            "snapshot_evidence": evidence,
        }
    return {
        **base,
        "state": "eligible",
        "eligible": True,
        "evaluator_accounting": "candidate_allowed",
        "reasons": [],
        "snapshot_evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--slot-id")
    parser.add_argument("--now", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-network", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_network:
        raise SystemExit(
            "Data Steward compilation is snapshot-only; --no-network is required"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite append-only artifact: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    snapshot = (
        json.loads(args.snapshot.read_text(encoding="utf-8")) if args.snapshot else None
    )
    now = parse_timestamp(args.now, "now")
    artifact = build_eligibility(
        manifest,
        now=now,
        slot_id=args.slot_id,
        snapshot=snapshot,
    )
    args.output.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "state": artifact["state"],
                "slot_id": artifact["slot_id"],
                "evaluator_accounting": artifact["evaluator_accounting"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
