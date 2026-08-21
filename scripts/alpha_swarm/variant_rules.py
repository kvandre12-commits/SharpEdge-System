"""Frozen deterministic rule family for additive Alpha Swarm paper variants."""

from __future__ import annotations

from typing import Any

from scripts.alpha_swarm.contracts import (
    CANDIDATE_SCHEMA,
    EXPLICIT_OUTCOME_FIELD,
    canonical_json,
    manifest_sha256,
    payload_sha256,
    validate_candidate,
)

PUBLICATION_SCHEMA = "sharpedge.alpha_swarm.variant_candidate_publication.v1"
RULE_ID = "vwap_momentum_volume_variant_v1"
RULE_VERSION = "1.0.0"
FEATURES = ("vs_vwap_pct", "momentum_15m_pct", "volume_ratio")
PAPER_NOTIONAL_DOLLARS = 100.0
FORBIDDEN_FEATURES = frozenset(
    {
        "ret_1d",
        EXPLICIT_OUTCOME_FIELD,
        "score",
        "rank",
        "confidence",
        "utility",
        "pnl",
        "net_pnl_dollars",
        "entry_price",
        "exit_price",
    }
)
VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant_id": "balanced_trend_v1",
        "variant_index": 1,
        "variant_count": 4,
        "family": "aligned_trend",
        "parameters": {
            "vwap_threshold_pct": 0.05,
            "momentum_threshold_pct": 0.05,
            "minimum_volume_ratio": 1.2,
        },
    },
    {
        "variant_id": "high_confirmation_breakout_v1",
        "variant_index": 2,
        "variant_count": 4,
        "family": "aligned_trend",
        "parameters": {
            "vwap_threshold_pct": 0.1,
            "momentum_threshold_pct": 0.1,
            "minimum_volume_ratio": 1.5,
        },
    },
    {
        "variant_id": "early_momentum_v1",
        "variant_index": 3,
        "variant_count": 4,
        "family": "aligned_trend",
        "parameters": {
            "vwap_threshold_pct": 0.03,
            "momentum_threshold_pct": 0.05,
            "minimum_volume_ratio": 1.0,
        },
    },
    {
        "variant_id": "stand_down_control_v1",
        "variant_index": 4,
        "variant_count": 4,
        "family": "control",
        "parameters": {},
    },
)


def variant_by_id(variant_id: str) -> dict[str, Any]:
    for variant in VARIANTS:
        if variant["variant_id"] == variant_id:
            return variant
    raise ValueError(f"unknown variant_id: {variant_id}")


def _feature_values(features: dict[str, Any]) -> dict[str, float]:
    forbidden = FORBIDDEN_FEATURES & {str(name).lower() for name in features}
    if forbidden:
        raise ValueError(
            f"variant evidence contains forbidden features: {sorted(forbidden)}"
        )
    missing = [name for name in FEATURES if name not in features]
    if missing:
        raise ValueError(f"variant evidence is missing features: {missing}")
    values: dict[str, float] = {}
    for name in FEATURES:
        try:
            values[name] = float(features[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"variant feature {name} must be numeric") from exc
    return values


def decide(variant: dict[str, Any], features: dict[str, Any]) -> str:
    values = _feature_values(features)
    if variant.get("family") == "control":
        return "stand_down"
    if variant.get("family") != "aligned_trend":
        raise ValueError("variant family must be aligned_trend or control")
    parameters = variant["parameters"]
    vwap = float(parameters["vwap_threshold_pct"])
    momentum = float(parameters["momentum_threshold_pct"])
    volume = float(parameters["minimum_volume_ratio"])
    if (
        values["vs_vwap_pct"] >= vwap
        and values["momentum_15m_pct"] >= momentum
        and values["volume_ratio"] >= volume
    ):
        return "long"
    if (
        values["vs_vwap_pct"] <= -vwap
        and values["momentum_15m_pct"] <= -momentum
        and values["volume_ratio"] >= volume
    ):
        return "short"
    return "stand_down"


def build_publication(
    *,
    base_manifest: dict[str, Any],
    slot: dict[str, Any],
    snapshot: dict[str, Any],
    steward: dict[str, Any],
    evidence_ref: dict[str, Any],
    variant: dict[str, Any],
    variant_manifest_sha256: str,
    observed_at: str,
) -> dict[str, Any]:
    if variant != variant_by_id(str(variant.get("variant_id") or "")):
        raise ValueError("variant differs from the frozen source registry")
    evidence_checks = {
        "variant_manifest_sha256": variant_manifest_sha256,
        "base_manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "snapshot_sha256": payload_sha256(snapshot),
        "data_steward_sha256": payload_sha256(steward),
    }
    for field, expected in evidence_checks.items():
        if evidence_ref.get(field) != expected:
            raise ValueError(
                f"shared evidence {field} does not match publication input"
            )
    values = _feature_values(snapshot.get("features") or {})
    decision = decide(variant, values)
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "run_id": base_manifest["run_id"],
        "manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "prediction_ts": slot["prediction_ts"],
        "published_at": slot["prediction_ts"],
        "feature_available_ts": snapshot["feature_available_ts"],
        "decision": decision,
        "risk_cap_dollars": 0.0 if decision == "stand_down" else PAPER_NOTIONAL_DOLLARS,
        "outcome_field": EXPLICIT_OUTCOME_FIELD,
        "feature_names": list(FEATURES),
        "feature_values": values,
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "variant_id": variant["variant_id"],
        "variant_index": variant["variant_index"],
        "variant_count": variant["variant_count"],
        "source_refs": [
            f"variant-manifest-sha256://{variant_manifest_sha256}",
            f"shared-evidence-sha256://{payload_sha256(evidence_ref)}",
            f"data-steward-sha256://{payload_sha256(steward)}",
            f"snapshot-sha256://{payload_sha256(snapshot)}",
            *list(snapshot.get("source_refs") or []),
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }
    candidate["candidate_id"] = payload_sha256(candidate)
    validate_candidate(base_manifest, candidate)
    return {
        "schema": PUBLICATION_SCHEMA,
        "variant_manifest_sha256": variant_manifest_sha256,
        "variant_id": variant["variant_id"],
        "variant_index": variant["variant_index"],
        "variant_count": variant["variant_count"],
        "rule": {
            "rule_id": RULE_ID,
            "rule_version": RULE_VERSION,
            "family": variant["family"],
            "parameters": variant["parameters"],
        },
        "base_run_id": base_manifest["run_id"],
        "base_manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "session_date": slot["session_date"],
        "observed_at": observed_at,
        "shared_evidence_sha256": payload_sha256(evidence_ref),
        "candidate": candidate,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "aggregate_score_computed": False,
    }


def rules_fingerprint() -> str:
    return payload_sha256(
        {
            "rule_id": RULE_ID,
            "rule_version": RULE_VERSION,
            "features": list(FEATURES),
            "paper_notional_dollars": PAPER_NOTIONAL_DOLLARS,
            "variants": list(VARIANTS),
            "canonical": canonical_json(list(VARIANTS)),
        }
    )
