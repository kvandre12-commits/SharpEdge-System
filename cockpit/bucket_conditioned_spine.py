"""Bucket-conditioned execution spine for SharpEdge.

This is the primary execution authority lane.
Day bucket defines the battlefield; the spine scores inside that battlefield.
No live-trigger wait caps. No post-score governance clamps. No token-policy math.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim
from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES

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
    "a_plus_trend_day": 6,
    "failed_breakdown_long_day": 8,
    "failed_breakout_short_day": 8,
    "range_balance_day": 0,
    "trap_noise_day": -12,
    "news_vol_shock_day": -20,
    "unclassified_day": -6,
}

BUCKET_DEFAULT_BIAS = {
    "failed_breakdown_long_day": "CALLS",
    "failed_breakout_short_day": "PUTS",
}


def _bucket_bias_value(label: str) -> float:
    if label == "CALLS":
        return 0.22
    if label == "PUTS":
        return -0.22
    return 0.0


def _profile_for_bucket(bucket: str) -> dict[str, float]:
    return dict(BUCKET_PROFILES.get(bucket, DEFAULT_PROFILE))


def _weighted_score(
    parts: dict[str, Any], weights: dict[str, float]
) -> tuple[int, list[dict[str, Any]]]:
    rows = []
    total_weight = 0.0
    weighted = 0.0
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        part = parts[name]
        weight = float(weights.get(name, 0.0))
        rows.append(
            {
                "name": name,
                "score": int(part.score),
                "bias": prim.bias_label(part.bias),
                "reason": part.reason,
                "weight": round(weight, 3),
                "contribution": round(float(part.score) * weight, 3),
            }
        )
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
        part = parts[name]
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


def build_bucket_conditioned_spine(
    parts: dict[str, Any],
    market_day: dict[str, Any],
) -> dict[str, Any]:
    bucket = str(market_day.get("bucket") or "unclassified_day")
    weights = _profile_for_bucket(bucket)
    base_score, rows = _weighted_score(parts, weights)
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
    reason = (
        f"{bucket} conditions the core spine; posture={posture or 'n/a'}; "
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
        "recommended_action": _recommended_action(bucket, gate, bias),
        "allowed_playbooks": list(market_day.get("allowed_playbooks") or []),
        "risk_posture": posture,
        "reason": reason,
        "features": rows,
        "best": best,
        "worst": worst,
    }


__all__ = ["build_bucket_conditioned_spine"]
