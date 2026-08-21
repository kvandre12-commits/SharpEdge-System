"""Shared equity captures and separate receipts for paper-only variants."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from scripts.alpha_swarm.contracts import (
    PAPER_MARK_SCHEMA,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
)
from scripts.alpha_swarm.evaluator import evaluate_candidate
from scripts.alpha_swarm.variant_rules import PUBLICATION_SCHEMA

SHARED_CAPTURE_SCHEMA = "sharpedge.alpha_swarm.shared_equity_capture.v1"
EVALUATION_SCHEMA = "sharpedge.alpha_swarm.variant_evaluation_publication.v1"


def select_complete_bar(
    provider_capture: dict[str, Any], *, target_ts: str
) -> dict[str, Any]:
    observed = parse_timestamp(provider_capture.get("observed_at"), "observed_at")
    target = parse_timestamp(target_ts, "target_ts")
    if provider_capture.get("provider") != "yahoo_chart_1m":
        raise ValueError("shared equity marks require yahoo_chart_1m evidence")
    candidates = []
    for index, raw in enumerate(provider_capture.get("bars") or []):
        timestamp = parse_timestamp(raw.get("timestamp"), f"bars[{index}].timestamp")
        if timestamp < target or timestamp + timedelta(minutes=1) > observed:
            continue
        try:
            bar = {
                "timestamp": timestamp.isoformat(),
                "open": float(raw["open"]),
                "high": float(raw["high"]),
                "low": float(raw["low"]),
                "close": float(raw["close"]),
                "volume": float(raw["volume"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bar {index} contains invalid equity fields") from exc
        if min(bar["open"], bar["high"], bar["low"], bar["close"]) <= 0:
            raise ValueError("equity bar prices must be positive")
        if (
            bar["low"] > bar["high"]
            or not bar["low"] <= bar["open"] <= bar["high"]
            or not bar["low"] <= bar["close"] <= bar["high"]
        ):
            raise ValueError("equity bar OHLC ordering is invalid")
        candidates.append(bar)
    if not candidates:
        raise ValueError(
            "no complete one-minute bar exists at or after the locked time"
        )
    return min(candidates, key=lambda item: item["timestamp"])


def build_shared_capture(
    *,
    base_manifest: dict[str, Any],
    slot: dict[str, Any],
    phase: str,
    provider_capture: dict[str, Any],
    variant_manifest_sha256: str,
) -> dict[str, Any]:
    if phase not in {"entry", "exit"}:
        raise ValueError("shared capture phase must be entry or exit")
    if provider_capture.get("symbol") != slot["symbol"]:
        raise ValueError("provider capture symbol does not match the locked slot")
    if provider_capture.get("session_date") != slot["session_date"]:
        raise ValueError("provider capture session does not match the locked slot")
    target_field = "entry_ts" if phase == "entry" else "exit_ts"
    selected = select_complete_bar(provider_capture, target_ts=slot[target_field])
    return {
        "schema": SHARED_CAPTURE_SCHEMA,
        "variant_manifest_sha256": variant_manifest_sha256,
        "base_run_id": base_manifest["run_id"],
        "base_manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "phase": phase,
        "locked_target_ts": slot[target_field],
        "observed_at": provider_capture["observed_at"],
        "provider_capture_sha256": payload_sha256(provider_capture),
        "provider_source_ref": provider_capture["source_ref"],
        "selection_rule": "earliest_complete_one_minute_bar_at_or_after_locked_time",
        "selected_bar": selected,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
    }


def _mark(
    *,
    base_manifest: dict[str, Any],
    slot: dict[str, Any],
    candidate: dict[str, Any],
    entry_capture: dict[str, Any],
    exit_capture: dict[str, Any],
    published_at: str,
) -> dict[str, Any] | None:
    if candidate["decision"] == "stand_down":
        return None
    if entry_capture.get("phase") != "entry" or exit_capture.get("phase") != "exit":
        raise ValueError("equity receipt requires entry and exit shared captures")
    for capture in (entry_capture, exit_capture):
        if capture.get("slot_id") != slot["slot_id"]:
            raise ValueError("shared capture slot does not match candidate")
    entry_bar = entry_capture["selected_bar"]
    exit_bar = exit_capture["selected_bar"]
    if candidate["decision"] == "long":
        entry_price = float(entry_bar["high"])
        exit_price = float(exit_bar["low"])
    else:
        entry_price = float(entry_bar["low"])
        exit_price = float(exit_bar["high"])
    quantity = float(candidate["risk_cap_dollars"]) / entry_price
    return {
        "schema": PAPER_MARK_SCHEMA,
        "run_id": base_manifest["run_id"],
        "manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "entry_ts": slot["entry_ts"],
        "exit_ts": slot["exit_ts"],
        "label_available_ts": slot["label_available_ts"],
        "published_at": published_at,
        "vehicle": "equity",
        "entry_method": base_manifest["fill_rules"]["equity_entry"],
        "exit_method": base_manifest["fill_rules"]["equity_exit"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "source_refs": [
            f"shared-entry-sha256://{payload_sha256(entry_capture)}",
            f"shared-exit-sha256://{payload_sha256(exit_capture)}",
        ],
        "paper_only": True,
        "execution_permitted": False,
    }


def build_evaluation_publication(
    *,
    base_manifest: dict[str, Any],
    candidate_publication: dict[str, Any],
    entry_capture: dict[str, Any],
    exit_capture: dict[str, Any],
    published_at: str,
) -> dict[str, Any]:
    if candidate_publication.get("schema") != PUBLICATION_SCHEMA:
        raise ValueError("variant candidate publication schema is invalid")
    candidate = candidate_publication.get("candidate") or {}
    if candidate.get("variant_id") != candidate_publication.get("variant_id"):
        raise ValueError("candidate variant identity does not match its publication")
    locked_variant_hash = candidate_publication.get("variant_manifest_sha256")
    expected_base_hash = manifest_sha256(base_manifest)
    for capture in (entry_capture, exit_capture):
        if capture.get("variant_manifest_sha256") != locked_variant_hash:
            raise ValueError("shared capture variant manifest hash does not match")
        if capture.get("base_manifest_sha256") != expected_base_hash:
            raise ValueError("shared capture base manifest hash does not match")
    slot = next(
        (
            item
            for item in base_manifest["slots"]
            if item["slot_id"] == candidate.get("slot_id")
        ),
        None,
    )
    if slot is None:
        raise ValueError("variant candidate targets an unknown base slot")
    if parse_timestamp(published_at, "published_at") < parse_timestamp(
        slot["label_available_ts"], "label_available_ts"
    ):
        raise ValueError("variant receipt cannot publish before label availability")
    mark = _mark(
        base_manifest=base_manifest,
        slot=slot,
        candidate=candidate,
        entry_capture=entry_capture,
        exit_capture=exit_capture,
        published_at=published_at,
    )
    receipt = evaluate_candidate(base_manifest, candidate, mark)
    return {
        "schema": EVALUATION_SCHEMA,
        "variant_manifest_sha256": candidate_publication["variant_manifest_sha256"],
        "variant_id": candidate_publication["variant_id"],
        "variant_index": candidate_publication["variant_index"],
        "variant_count": candidate_publication["variant_count"],
        "base_run_id": base_manifest["run_id"],
        "base_manifest_sha256": manifest_sha256(base_manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "published_at": published_at,
        "candidate_sha256": payload_sha256(candidate),
        "shared_entry_capture_sha256": payload_sha256(entry_capture),
        "shared_exit_capture_sha256": payload_sha256(exit_capture),
        "paper_mark": mark,
        "evaluation_receipt": receipt,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "aggregate_score_computed": False,
    }
