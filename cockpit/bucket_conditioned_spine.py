"""Bucket-conditioned execution spine for SharpEdge.

Day bucket defines the battlefield; the spine scores inside that battlefield.
This score is a cockpit diagnostic/advisory read, not final broker or operator
authority. Final authority lives in the approval/governance layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import execution_vector_primitives as prim
from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES
from graph_state import attach_graph_agreement

DEFAULT_PROFILE = {
    "structure_score": 0.18,
    "acceptance_score": 0.18,
    "trend_score": 0.18,
    "location_score": 0.16,
    "volume_score": 0.12,
    "time_of_day_score": 0.08,
    "dealer_gamma_score": 0.10,
}

BUCKET_PROFILES = {
    "a_plus_trend_day": {
        "structure_score": 0.21,
        "acceptance_score": 0.16,
        "trend_score": 0.23,
        "location_score": 0.10,
        "volume_score": 0.16,
        "time_of_day_score": 0.05,
        "dealer_gamma_score": 0.09,
    },
    "failed_breakdown_long_day": {
        "structure_score": 0.15,
        "acceptance_score": 0.24,
        "trend_score": 0.10,
        "location_score": 0.19,
        "volume_score": 0.16,
        "time_of_day_score": 0.06,
        "dealer_gamma_score": 0.10,
    },
    "failed_breakout_short_day": {
        "structure_score": 0.15,
        "acceptance_score": 0.24,
        "trend_score": 0.10,
        "location_score": 0.19,
        "volume_score": 0.16,
        "time_of_day_score": 0.06,
        "dealer_gamma_score": 0.10,
    },
    "range_balance_day": {
        "structure_score": 0.10,
        "acceptance_score": 0.22,
        "trend_score": 0.08,
        "location_score": 0.24,
        "volume_score": 0.14,
        "time_of_day_score": 0.08,
        "dealer_gamma_score": 0.14,
    },
    "trap_noise_day": {
        "structure_score": 0.16,
        "acceptance_score": 0.16,
        "trend_score": 0.12,
        "location_score": 0.16,
        "volume_score": 0.14,
        "time_of_day_score": 0.10,
        "dealer_gamma_score": 0.16,
    },
    "news_vol_shock_day": {
        "structure_score": 0.14,
        "acceptance_score": 0.14,
        "trend_score": 0.14,
        "location_score": 0.16,
        "volume_score": 0.14,
        "time_of_day_score": 0.12,
        "dealer_gamma_score": 0.16,
    },
    "unclassified_day": DEFAULT_PROFILE,
}

BUCKET_SCORE_OFFSETS = {
    "a_plus_trend_day": 10,
    "failed_breakdown_long_day": 12,
    "failed_breakout_short_day": 12,
    "range_balance_day": 0,
    "trap_noise_day": -12,
    "news_vol_shock_day": -20,
    "unclassified_day": -6,
}

SPINE_ADJUNCT_WEIGHTS = {
    "pressure_score": 0.07,
    "balance_context_score": 0.07,
}

REALTIME_ADJUST_ENV = "SHARPEDGE_SPINE_REALTIME_ADJUST"
REALTIME_ADJUST_PATH_ENV = "SHARPEDGE_SPINE_REALTIME_ADJUSTMENTS"
DEFAULT_REALTIME_ADJUSTMENT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "spine_realtime_adjustments.json"
)

BUCKET_DEFAULT_BIAS = {
    "failed_breakdown_long_day": "CALLS",
    "failed_breakout_short_day": "PUTS",
}

BUCKET_DISPLAY_LABELS = {
    "a_plus_trend_day": "A+ trend day",
    "failed_breakdown_long_day": "failed-breakdown / long reclaim day",
    "failed_breakout_short_day": "failed-breakout / short rejection day",
    "range_balance_day": "range / balance day",
    "trap_noise_day": "trap-noise day",
    "news_vol_shock_day": "news / vol shock day",
    "unclassified_day": "awaiting clean day type",
}


def _bucket_display_label(bucket: str) -> str:
    return BUCKET_DISPLAY_LABELS.get(bucket, bucket.replace("_", " "))


def _bucket_bias_value(label: str) -> float:
    if label == "CALLS":
        return 0.22
    if label == "PUTS":
        return -0.22
    return 0.0


def _runtime_adjustment_enabled() -> bool:
    return str(os.getenv(REALTIME_ADJUST_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_realtime_adjustment_overlay() -> dict[str, Any]:
    if not _runtime_adjustment_enabled():
        return {}
    path = Path(os.getenv(REALTIME_ADJUST_PATH_ENV) or DEFAULT_REALTIME_ADJUSTMENT_PATH)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _apply_realtime_adjustments(
    profile: dict[str, float], overlay: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not overlay or overlay.get("authority") != "diagnostic_shadow_overlay":
        return profile, []
    if not overlay.get("enabled"):
        return profile, []
    max_abs_delta = min(0.03, abs(float(overlay.get("max_abs_weight_delta") or 0.03)))
    adjusted = dict(profile)
    applied = []
    for name, row in (overlay.get("adjustments") or {}).items():
        if name not in adjusted or name not in CORE_EXECUTION_SPINE_PART_NAMES:
            continue
        raw_delta = float((row or {}).get("weight_delta") or 0.0)
        delta = max(-max_abs_delta, min(max_abs_delta, raw_delta))
        if delta == 0.0:
            continue
        before = float(adjusted[name])
        after = max(0.0, before + delta)
        adjusted[name] = after
        applied.append(
            {
                "name": name,
                "before": round(before, 3),
                "after": round(after, 3),
                "delta": round(after - before, 3),
                "reason": str((row or {}).get("reason") or "runtime shadow audit"),
            }
        )
    return adjusted, applied


def _profile_for_bucket(bucket: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    profile = dict(BUCKET_PROFILES.get(bucket, DEFAULT_PROFILE))
    profile.update(SPINE_ADJUNCT_WEIGHTS)
    return _apply_realtime_adjustments(profile, _load_realtime_adjustment_overlay())


def _weighted_score(
    parts: dict[str, Any],
    weights: dict[str, float],
    graph_state: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    rows = []
    total_weight = 0.0
    weighted = 0.0
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        part = parts.get(name)
        if part is None:
            continue
        weight = float(weights.get(name, 0.0))
        row = {
            "name": name,
            "score": int(part.score),
            "bias": prim.bias_label(part.bias),
            "reason": part.reason,
            "weight": round(weight, 3),
            "contribution": round(float(part.score) * weight, 3),
        }
        rows.append(attach_graph_agreement(row, part, graph_state))
        weighted += float(part.score) * weight
        total_weight += weight
    normalized = weighted / max(total_weight, 1e-9)
    return prim.clamp(normalized), rows


def _weighted_bias(
    parts: dict[str, Any], weights: dict[str, float], bucket_bias: str
) -> float:
    total = 0.0
    total_weight = 0.0
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        part = parts.get(name)
        if part is None:
            continue
        weight = float(weights.get(name, 0.0))
        total += float(part.bias) * weight * (float(part.score) / 100.0)
        total_weight += weight
    bias_value = total / max(total_weight, 1e-9)
    if bucket_bias == "NEUTRAL":
        return bias_value
    if bucket_bias == "CALLS":
        return max(bias_value, _bucket_bias_value(bucket_bias))
    return min(bias_value, _bucket_bias_value(bucket_bias))


def _rank_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda row: (row["score"], row["weight"]), reverse=True)
    return ranked[:3], list(reversed(ranked[-3:]))


def _recommended_action(bucket: str, gate: str, bias: str) -> str:
    if bucket in {"news_vol_shock_day", "trap_noise_day"}:
        return "stand_down"
    if gate == "BLOCK":
        return "stand_down"
    if bucket == "range_balance_day" and gate == "CAUTION":
        return "watch_edges"
    if bias == "CALLS":
        return "candidate_calls"
    if bias == "PUTS":
        return "candidate_puts"
    return "watch_only"


def _diagnostic_posture(action: str) -> str:
    labels = {
        "stand_down": "stand_down_context_only",
        "watch_edges": "watch_edges_context_only",
        "watch_only": "watch_only_context",
        "candidate_calls": "calls_context_only",
        "candidate_puts": "puts_context_only",
    }
    return labels.get(action, f"{action}_context_only")


def build_bucket_conditioned_spine(
    parts: dict[str, Any],
    market_day: dict[str, Any],
    graph_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bucket = str(market_day.get("bucket") or "unclassified_day")
    weights, realtime_adjustments = _profile_for_bucket(bucket)
    base_score, rows = _weighted_score(parts, weights, graph_state)
    offset = int(BUCKET_SCORE_OFFSETS.get(bucket, 0))
    score = prim.clamp(base_score + offset)
    bucket_bias = str(market_day.get("bias") or "NEUTRAL")
    if bucket_bias == "NEUTRAL":
        bucket_bias = BUCKET_DEFAULT_BIAS.get(bucket, bucket_bias)
    bias_value = _weighted_bias(parts, weights, bucket_bias)
    bias = prim.bias_label(bias_value)
    gate = prim.gate_label(score)
    best, worst = _rank_rows(rows)
    posture = str(market_day.get("risk_posture") or "")
    bucket_label = _bucket_display_label(bucket)
    action = _recommended_action(bucket, gate, bias)
    reason = (
        f"{bucket_label} conditions the core spine; posture={posture or 'n/a'}; "
        f"base {base_score} with bucket offset {offset:+d}."
    )
    return {
        "schema": "sharpedge.bucket_conditioned_spine.v1",
        "bucket": bucket,
        "score": score,
        "base_score": base_score,
        "bucket_offset": offset,
        "gate": gate,
        "bias": bias,
        "bias_strength": round(abs(bias_value), 3),
        "recommended_action": action,
        "recommended_action_status": "deprecated_compatibility_alias",
        "diagnostic_posture": _diagnostic_posture(action),
        "advisory_only": True,
        "authority_role": "diagnostic_advisory",
        "allowed_playbooks": list(market_day.get("allowed_playbooks") or []),
        "risk_posture": posture,
        "reason": reason,
        "graph_state": graph_state or {},
        "features": rows,
        "realtime_adjustments": {
            "enabled": bool(realtime_adjustments),
            "applied": realtime_adjustments,
            "authority": "diagnostic_shadow_overlay",
        },
        "best": best,
        "worst": worst,
    }


__all__ = ["build_bucket_conditioned_spine"]
