"""Ace-style authority adapter for SharpEdge cockpit permissions.

This keeps the SharpEdge cockpit shell, bucket context, carry maps, and event
receipts while trimming the permission contract down to the core execution
spine.
"""

from __future__ import annotations

from typing import Any

from bucket_conditioned_spine import build_bucket_conditioned_spine
from day_bucket import classify_day_bucket
from execution_card_builder import build_trade_permission_card
from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES
from execution_vector_engine import ExecutionVectorEngine
from execution_vector_weights import DEFAULT_BASE_WEIGHTS

ACE_AUTHORITY_ENGINE = "ace"
LEGACY_AUTHORITY_ENGINE = "legacy"


def _core_parts(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        name: parts[name]
        for name in CORE_EXECUTION_SPINE_PART_NAMES
        if name in parts and parts[name] is not None
    }


def _core_score_weights() -> dict[str, float]:
    return {
        name: float(DEFAULT_BASE_WEIGHTS.get(name, 0.0))
        for name in CORE_EXECUTION_SPINE_PART_NAMES
    }


def _signed_bias_value(spine: dict[str, Any]) -> float:
    strength = float(spine.get("bias_strength") or 0.0)
    bias = str(spine.get("bias") or "NEUTRAL")
    if bias == "CALLS":
        return strength
    if bias == "PUTS":
        return -strength
    return 0.0


def build_ace_authority_card(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    pa: dict[str, Any],
    levels: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
    op: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
    magnitude: dict[str, Any] | None = None,
    volatility_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a SharpEdge permission card with Ace-style authority.

    The day bucket is still classified with the richer local context so the
    market-type lens survives, but final permission authority is reduced to the
    core spine only.
    """
    engine = ExecutionVectorEngine()
    full_parts = engine.build_parts(
        bars,
        pa,
        levels,
        setups=setups,
        op=op,
        gp=gp,
        magnitude=magnitude,
        volatility_structure=volatility_structure,
    )
    core_parts = _core_parts(full_parts)
    market_day = classify_day_bucket(
        full_parts,
        engine.pa,
        engine.op,
        engine.gp,
        engine.setups,
    )
    spine = build_bucket_conditioned_spine(core_parts, market_day)
    card = build_trade_permission_card(
        parts=core_parts,
        setups=engine.setups,
        pa=engine.pa,
        raw_permission=int(spine.get("base_score") or spine.get("score") or 0),
        permission=int(spine.get("score") or 0),
        bias_value=_signed_bias_value(spine),
        grammar={"authority_engine": ACE_AUTHORITY_ENGINE, "mode": "core_spine_only"},
        market_day=market_day,
        bucket_conditioned_spine=spine,
        score_weights=_core_score_weights(),
        op=engine.op,
        gp=engine.gp,
        corroboration_parts=full_parts,
        structure_state=engine.structure_state,
        acceptance_state=engine.acceptance_state,
        location_state=engine.location_state,
        dealer_state=engine.dealer_state,
        volume_state=engine.volume_state,
        trend_state=engine.trend_state,
        time_state=engine.time_state,
    )
    card["authority_engine"] = ACE_AUTHORITY_ENGINE
    card["authority_mode"] = "core_spine_only"
    card["authority_summary"] = {
        "bucket": market_day.get("bucket"),
        "gate": spine.get("gate"),
        "score": spine.get("score"),
        "bias": spine.get("bias"),
        "recommended_action": spine.get("recommended_action"),
    }
    card["legacy_engine_retained_for_context_only"] = {
        "uses_day_bucket_context": True,
        "uses_event_receipts": True,
        "uses_weekly_monthly_context": True,
        "uses_carry_maps": True,
    }
    return card


def is_ace_authority_engine(name: str | None) -> bool:
    return str(name or "").strip().lower() == ACE_AUTHORITY_ENGINE


def normalize_authority_engine(name: str | None) -> str:
    if is_ace_authority_engine(name):
        return ACE_AUTHORITY_ENGINE
    return LEGACY_AUTHORITY_ENGINE


__all__ = [
    "ACE_AUTHORITY_ENGINE",
    "LEGACY_AUTHORITY_ENGINE",
    "build_ace_authority_card",
    "is_ace_authority_engine",
    "normalize_authority_engine",
]
