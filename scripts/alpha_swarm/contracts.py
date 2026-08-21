from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

RUN_MANIFEST_SCHEMA = "sharpedge.alpha_swarm.run_manifest.v1"
CANDIDATE_SCHEMA = "sharpedge.alpha_swarm.candidate.v1"
PAPER_MARK_SCHEMA = "sharpedge.alpha_swarm.paper_mark.v1"
RECEIPT_SCHEMA = "sharpedge.alpha_swarm.evaluation_receipt.v1"
SCORE_SCHEMA = "sharpedge.alpha_swarm.score.v1"
EXPLICIT_OUTCOME_FIELD = "return_prediction_to_exit"
FORBIDDEN_OUTCOME_FIELDS = frozenset({"ret_1d"})
DECISIONS = frozenset({"long", "short", "stand_down"})
VEHICLES = frozenset({"equity", "debit_spread"})


class ContractError(ValueError):
    """Raised when an alpha-swarm artifact violates the locked contract."""


def parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty ISO timestamp")
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractError(f"{field} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return payload_sha256(manifest)


def source_bundle_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    resolved = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if not resolved:
        raise ContractError("at least one evaluator source path is required")
    for path in resolved:
        if not path.is_file():
            raise ContractError(f"evaluator source file is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    if payload.get(field) is not expected:
        raise ContractError(f"{field} must be {expected}")


def _require_non_empty_list(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a non-empty list")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ContractError(f"manifest schema must be {RUN_MANIFEST_SCHEMA}")
    if not str(manifest.get("run_id") or "").strip():
        raise ContractError("run_id is required")
    locked_at = parse_timestamp(manifest.get("locked_at"), "locked_at")
    _require_bool(manifest, "paper_only", True)
    _require_bool(manifest, "authoritative", False)
    _require_bool(manifest, "execution_permitted", False)

    source_hash = str(manifest.get("evaluator_source_sha256") or "")
    if len(source_hash) != 64:
        raise ContractError("evaluator_source_sha256 must be a SHA-256 hex digest")

    universe = _require_non_empty_list(manifest, "universe")
    if len(universe) != len(set(universe)):
        raise ContractError("universe symbols must be unique")
    if any(symbol != str(symbol).upper() for symbol in universe):
        raise ContractError("universe symbols must be uppercase")

    label = manifest.get("label_contract") or {}
    if label.get("outcome_field") != EXPLICIT_OUTCOME_FIELD:
        raise ContractError(
            f"outcome_field must be explicit: {EXPLICIT_OUTCOME_FIELD}; ret_1d is forbidden"
        )
    forbidden = set(label.get("forbidden_outcome_fields") or [])
    if not FORBIDDEN_OUTCOME_FIELDS.issubset(forbidden):
        raise ContractError("label contract must explicitly forbid ret_1d")

    metric = manifest.get("metric") or {}
    if metric.get("name") != "lower_confidence_net_utility_per_eligible_slot":
        raise ContractError("locked metric name is invalid")
    quantile = float(metric.get("lower_quantile", -1))
    if not 0 < quantile < 0.5:
        raise ContractError("lower_quantile must be between 0 and 0.5")
    if int(metric.get("bootstrap_iterations") or 0) < 100:
        raise ContractError("bootstrap_iterations must be at least 100")

    slots = _require_non_empty_list(manifest, "slots")
    slot_ids: set[str] = set()
    for slot in slots:
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id or slot_id in slot_ids:
            raise ContractError("slot_id values must be non-empty and unique")
        slot_ids.add(slot_id)
        if slot.get("symbol") not in universe:
            raise ContractError(
                f"slot {slot_id} uses a symbol outside the locked universe"
            )
        declared = parse_timestamp(
            slot.get("eligibility_declared_at"), "eligibility_declared_at"
        )
        prediction = parse_timestamp(slot.get("prediction_ts"), "prediction_ts")
        entry = parse_timestamp(slot.get("entry_ts"), "entry_ts")
        exit_ts = parse_timestamp(slot.get("exit_ts"), "exit_ts")
        label_available = parse_timestamp(
            slot.get("label_available_ts"), "label_available_ts"
        )
        if locked_at > declared:
            raise ContractError(f"slot {slot_id} was declared before the manifest lock")
        if not declared <= prediction < entry < exit_ts <= label_available:
            raise ContractError(f"slot {slot_id} has invalid point-in-time ordering")
        if not isinstance(slot.get("eligible"), bool):
            raise ContractError(f"slot {slot_id} must declare eligible as a boolean")
        if (
            not slot["eligible"]
            and not str(slot.get("unavailable_reason") or "").strip()
        ):
            raise ContractError(
                f"ineligible slot {slot_id} needs an unavailable_reason"
            )


def slots_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_manifest(manifest)
    return {str(slot["slot_id"]): slot for slot in manifest["slots"]}


def validate_candidate(
    manifest: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    slots = slots_by_id(manifest)
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ContractError(f"candidate schema must be {CANDIDATE_SCHEMA}")
    if candidate.get("run_id") != manifest.get("run_id"):
        raise ContractError("candidate run_id does not match manifest")
    if candidate.get("manifest_sha256") != manifest_sha256(manifest):
        raise ContractError("candidate manifest_sha256 does not match locked manifest")
    _require_bool(candidate, "paper_only", True)
    _require_bool(candidate, "authoritative", False)
    _require_bool(candidate, "execution_permitted", False)

    slot_id = str(candidate.get("slot_id") or "")
    if slot_id not in slots:
        raise ContractError("candidate slot_id is not in the locked manifest")
    slot = slots[slot_id]
    if not slot["eligible"]:
        raise ContractError("candidate cannot target a predeclared ineligible slot")
    if candidate.get("symbol") != slot.get("symbol"):
        raise ContractError("candidate symbol does not match its slot")
    prediction_ts = parse_timestamp(candidate.get("prediction_ts"), "prediction_ts")
    if prediction_ts != parse_timestamp(
        slot.get("prediction_ts"), "slot.prediction_ts"
    ):
        raise ContractError(
            "candidate prediction_ts must equal the locked slot timestamp"
        )
    if parse_timestamp(candidate.get("published_at"), "published_at") != prediction_ts:
        raise ContractError(
            "candidate published_at must equal the locked prediction_ts"
        )
    if (
        parse_timestamp(candidate.get("feature_available_ts"), "feature_available_ts")
        > prediction_ts
    ):
        raise ContractError("feature_available_ts cannot be after prediction_ts")

    decision = str(candidate.get("decision") or "")
    if decision not in DECISIONS:
        raise ContractError(f"decision must be one of {sorted(DECISIONS)}")
    risk_cap = candidate.get("risk_cap_dollars")
    if decision == "stand_down":
        if risk_cap not in (None, 0, 0.0):
            raise ContractError("stand_down risk_cap_dollars must be zero or null")
    else:
        try:
            if float(risk_cap) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "directional candidates require a positive risk_cap_dollars"
            ) from exc
    if candidate.get("outcome_field") != EXPLICIT_OUTCOME_FIELD:
        raise ContractError(f"candidate outcome_field must be {EXPLICIT_OUTCOME_FIELD}")
    feature_names = {str(name) for name in candidate.get("feature_names") or []}
    if FORBIDDEN_OUTCOME_FIELDS & feature_names:
        raise ContractError(
            "ret_1d is ambiguous and forbidden as a candidate feature/outcome"
        )
    if EXPLICIT_OUTCOME_FIELD in feature_names:
        raise ContractError("the forward outcome cannot be used as an input feature")

    if not str(candidate.get("rule_id") or "").strip():
        raise ContractError("rule_id is required")
    if not str(candidate.get("rule_version") or "").strip():
        raise ContractError("rule_version is required")
    variant_index = int(candidate.get("variant_index") or 0)
    variant_count = int(candidate.get("variant_count") or 0)
    if variant_index < 1 or variant_count < variant_index:
        raise ContractError(
            "variant_index/count must disclose the tested search family"
        )
    if not candidate.get("source_refs"):
        raise ContractError("candidate source_refs are required")
    return slot


def validate_mark(
    manifest: dict[str, Any], candidate: dict[str, Any], mark: dict[str, Any]
) -> dict[str, Any]:
    slot = validate_candidate(manifest, candidate)
    if mark.get("schema") != PAPER_MARK_SCHEMA:
        raise ContractError(f"paper mark schema must be {PAPER_MARK_SCHEMA}")
    if mark.get("run_id") != manifest.get("run_id"):
        raise ContractError("paper mark run_id does not match manifest")
    if mark.get("manifest_sha256") != manifest_sha256(manifest):
        raise ContractError("paper mark manifest_sha256 does not match manifest")
    if mark.get("slot_id") != candidate.get("slot_id"):
        raise ContractError("paper mark slot_id does not match candidate")
    _require_bool(mark, "paper_only", True)
    _require_bool(mark, "execution_permitted", False)
    if parse_timestamp(mark.get("entry_ts"), "entry_ts") != parse_timestamp(
        slot.get("entry_ts"), "slot.entry_ts"
    ):
        raise ContractError("paper mark entry_ts must equal the locked slot entry_ts")
    if parse_timestamp(mark.get("exit_ts"), "exit_ts") != parse_timestamp(
        slot.get("exit_ts"), "slot.exit_ts"
    ):
        raise ContractError("paper mark exit_ts must equal the locked slot exit_ts")
    locked_label_time = parse_timestamp(
        slot.get("label_available_ts"), "slot.label_available_ts"
    )
    if (
        parse_timestamp(mark.get("label_available_ts"), "label_available_ts")
        < locked_label_time
    ):
        raise ContractError(
            "paper mark label timestamp precedes locked label availability"
        )
    if parse_timestamp(mark.get("published_at"), "published_at") < locked_label_time:
        raise ContractError(
            "paper mark was published before the label became available"
        )
    if mark.get("vehicle") not in VEHICLES:
        raise ContractError(f"paper mark vehicle must be one of {sorted(VEHICLES)}")
    if not mark.get("source_refs"):
        raise ContractError("paper mark source_refs are required")
    return slot


__all__ = [
    "CANDIDATE_SCHEMA",
    "ContractError",
    "EXPLICIT_OUTCOME_FIELD",
    "FORBIDDEN_OUTCOME_FIELDS",
    "PAPER_MARK_SCHEMA",
    "RECEIPT_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "SCORE_SCHEMA",
    "canonical_json",
    "manifest_sha256",
    "parse_timestamp",
    "payload_sha256",
    "slots_by_id",
    "source_bundle_sha256",
    "validate_candidate",
    "validate_manifest",
    "validate_mark",
]
