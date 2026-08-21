"""Assemble the final SharpEdge trade permission card.

Card builder may derive packet/display fields from upstream contracts. It may
not infer market state, bucket identity, thesis, readiness, or permission.
"""

from __future__ import annotations

from typing import Any

from authority_self_audit import build_authority_self_audit
from setup_conviction import build_setup_conviction, sync_setup_evidence_fields
import execution_vector_primitives as prim
from execution_expansion_potential import build_execution_expansion_potential
from execution_hierarchy import (
    CORE_EXECUTION_SPINE_PART_NAMES,
    build_execution_hierarchy,
)
from execution_vector_interactions import build_execution_vector_interactions
from spine_phase_model import annotate_spine_score_phases


def _resolve_display_bias(
    bias_value: float, bucket_conditioned_spine: dict[str, Any]
) -> tuple[str, float]:
    if bucket_conditioned_spine:
        return (
            str(bucket_conditioned_spine.get("bias") or "NEUTRAL"),
            round(float(bucket_conditioned_spine.get("bias_strength") or 0.0), 3),
        )
    return prim.bias_label(bias_value), round(abs(bias_value), 3)


def build_execution_flow(
    market_day: dict[str, Any],
    bucket_conditioned_spine: dict[str, Any],
    permission: int,
    bias_label: str,
) -> dict[str, Any]:
    return {
        "market_day_score": market_day.get("score"),
        "day_bucket": market_day.get("bucket"),
        "allowed_playbooks": market_day.get("allowed_playbooks", []),
        "bucket_conditioned_spine": {
            "gate": bucket_conditioned_spine.get("gate"),
            "score": bucket_conditioned_spine.get("score"),
            "bias": bucket_conditioned_spine.get("bias"),
            "diagnostic_posture": bucket_conditioned_spine.get("diagnostic_posture"),
            "advisory_only": bucket_conditioned_spine.get("advisory_only", True),
            "authority_role": bucket_conditioned_spine.get("authority_role"),
            "reason": bucket_conditioned_spine.get("reason"),
        },
        "execution_permission": {
            "gate": prim.gate_label(permission),
            "score": permission,
            "bias": bias_label,
        },
    }


def _clean_reasons(items: list[Any] | None, limit: int = 3) -> list[str]:
    clean = [str(item).strip() for item in (items or []) if str(item).strip()]
    return clean[:limit]


def _voice_packet(
    *,
    voice_id: str,
    label: str,
    stance: str,
    summary: str,
    score: Any = None,
    bias: Any = None,
    source: str = "",
) -> dict[str, Any]:
    return {
        "voice_id": voice_id,
        "label": label,
        "stance": stance,
        "summary": summary,
        "score": score,
        "bias": bias,
        "source": source,
    }


def _candidate_thesis(setup_conviction: dict[str, Any]) -> dict[str, Any]:
    fresh = setup_conviction.get("fresh_setup_evidence") or {}
    persisted = setup_conviction.get("persisted_setup_thesis") or {}
    setup_tag = (
        fresh.get("setup_tag") or persisted.get("setup_tag") or "No active setup thesis"
    )
    source = fresh.get("source") or persisted.get("source") or "setup_conviction"
    confidence = int(setup_conviction.get("setup_conviction_score") or 0)
    if fresh.get("setup_tag"):
        status = str(fresh.get("status") or "fresh_setup")
        reason = (
            f"{fresh.get('setup_tag')} via {status}; "
            f"actionable={bool(fresh.get('actionable'))}"
        )
        return {
            "label": fresh.get("setup_tag"),
            "source": source,
            "bias": setup_conviction.get("bias") or "NEUTRAL",
            "confidence": confidence,
            "reason": reason,
        }
    if persisted.get("active"):
        reason = f"carried thesis from lifecycle; status={persisted.get('event_status') or 'unknown'}"
        return {
            "label": setup_tag,
            "source": persisted.get("source") or source,
            "bias": setup_conviction.get("bias") or "NEUTRAL",
            "confidence": confidence,
            "reason": reason,
        }
    return {
        "label": setup_tag,
        "source": source,
        "bias": setup_conviction.get("bias") or "NEUTRAL",
        "confidence": confidence,
        "reason": str(setup_conviction.get("reason") or "no active setup card"),
    }


def _trap_voice_packet(setup_conviction: dict[str, Any]) -> dict[str, Any]:
    trap = setup_conviction.get("live_trap_corroboration") or {}
    trap_score = int(trap.get("trap_score") or 0)
    rejection_score = int(trap.get("rejection_score") or 0)
    if rejection_score >= trap_score:
        score = rejection_score
        bias = trap.get("rejection_bias") or "NEUTRAL"
        summary = trap.get("rejection_reason") or "no obvious rejection/trap"
        stance = "rejection"
    else:
        score = trap_score
        bias = trap.get("trap_bias") or "NEUTRAL"
        summary = trap.get("trap_reason") or "no failed-break trap detected"
        stance = "trap"
    return _voice_packet(
        voice_id="trap_corroboration",
        label="Trap / rejection corroboration",
        stance=stance,
        summary=str(summary),
        score=score,
        bias=bias,
        source=str(trap.get("source") or "execution_vectors"),
    )


def _expansion_voice_packet(expansion_potential: dict[str, Any]) -> dict[str, Any]:
    summary = expansion_potential.get("summary") or {}
    surface = expansion_potential.get("surface") or {}
    score = surface.get("score")
    bias = surface.get("bias") or "NEUTRAL"
    stance = str(summary.get("state") or "mixed")
    text = str(summary.get("note") or "expansion state unavailable")
    return _voice_packet(
        voice_id="expansion_fuel",
        label="Participation / expansion fuel",
        stance=stance,
        summary=text,
        score=score,
        bias=bias,
        source="execution_expansion_potential",
    )


def _build_authority_adjudication(
    *,
    setup_conviction: dict[str, Any],
    market_day: dict[str, Any],
    bucket_conditioned_spine: dict[str, Any],
    expansion_potential: dict[str, Any],
    reasons: dict[str, list[str]],
    grammar: dict[str, Any],
    permission: int,
    bias_label: str,
    authority_self_audit: dict[str, Any],
) -> dict[str, Any]:
    candidate = _candidate_thesis(setup_conviction)
    because = _clean_reasons(
        [bucket_conditioned_spine.get("reason"), *reasons.get("supporting", [])],
        limit=3,
    )
    despite = _clean_reasons(reasons.get("warnings"), limit=3)
    chosen_action = str(
        bucket_conditioned_spine.get("diagnostic_posture") or "watch_only_context"
    )
    gate = str(bucket_conditioned_spine.get("gate") or prim.gate_label(permission))
    bucket = str(market_day.get("bucket") or "unclassified_day")
    trap_voice = _trap_voice_packet(setup_conviction)
    voices = [
        _voice_packet(
            voice_id="setup_identity",
            label="Setup identity",
            stance=str(candidate.get("label") or "none"),
            summary=str(candidate.get("reason") or "no active setup card"),
            score=candidate.get("confidence"),
            bias=candidate.get("bias") or "NEUTRAL",
            source=str(candidate.get("source") or "setup_conviction"),
        ),
        trap_voice,
        _voice_packet(
            voice_id="bucket_context",
            label="Bucket context governor",
            stance=bucket,
            summary=str(market_day.get("reason") or "bucket context unavailable"),
            score=market_day.get("score"),
            bias=market_day.get("bias") or "NEUTRAL",
            source="market_day",
        ),
        _expansion_voice_packet(expansion_potential),
    ]
    overridden = [
        voice["label"]
        for voice in voices
        if voice.get("bias") not in {None, "", "NEUTRAL", bias_label}
    ]
    final_authority = str(
        authority_self_audit.get("final_authority_source")
        or "approval_decision_plus_operator"
    )
    score_role = str(
        authority_self_audit.get("score_spine_role") or "diagnostic_advisory"
    )
    summary = (
        f"Context may be occurring: {candidate.get('label') or 'no active setup thesis'}. "
        f"Cockpit read posture: {chosen_action} ({gate} / {bias_label}) because "
        f"{'; '.join(because[:2]) or 'authority reasons are still sparse'}. "
        f"Score-spine role={score_role}; final authority={final_authority}."
    )
    cockpit_read = {
        "gate": gate,
        "action": chosen_action,
        "bias": bias_label,
        "bucket": bucket,
        "score": int(bucket_conditioned_spine.get("score") or permission),
        "authority_engine": str(grammar.get("authority_engine") or "legacy"),
        "score_spine_role": score_role,
        "final_authority_source": final_authority,
        "advisory_only": True,
        "authority_role": "diagnostic_advisory",
    }
    return {
        "schema": "sharpedge.authority_adjudication.v1",
        "this_may_be_occurring": candidate,
        "cockpit_read": cockpit_read,
        "we_are_doing_this": cockpit_read,
        "because": because,
        "despite": despite,
        "competing_voices": voices,
        "overridden_voices": overridden,
        "summary": summary,
    }


def build_trade_permission_card(
    *,
    parts: dict[str, Any],
    setups: list[dict[str, Any]],
    pa: dict[str, Any],
    raw_permission: int,
    permission: int,
    bias_value: float,
    grammar: dict[str, Any],
    market_day: dict[str, Any],
    bucket_conditioned_spine: dict[str, Any],
    score_weights: dict[str, float],
    op: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
    corroboration_parts: dict[str, Any] | None = None,
    structure_state: dict[str, Any] | None = None,
    acceptance_state: dict[str, Any] | None = None,
    location_state: dict[str, Any] | None = None,
    dealer_state: dict[str, Any] | None = None,
    volume_state: dict[str, Any] | None = None,
    trend_state: dict[str, Any] | None = None,
    time_state: dict[str, Any] | None = None,
    graph_state: dict[str, Any] | None = None,
    line_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bias_label, bias_strength = _resolve_display_bias(
        bias_value, bucket_conditioned_spine
    )
    reasons = prim.reasons_from_parts(parts)
    scores = annotate_spine_score_phases(
        prim.serialize_parts(parts),
        pa,
        op=op or {},
        gp=gp or {},
        market_day=market_day,
        setups=setups,
    )
    spine_phase_model = {
        name: {
            "phase": item.get("phase"),
            "phase_reason": item.get("phase_reason"),
        }
        for name, item in scores.items()
        if name in CORE_EXECUTION_SPINE_PART_NAMES and item.get("phase")
    }
    expansion_potential = build_execution_expansion_potential(
        scores,
        pa=pa,
        gp=gp or {},
    )
    vector_interactions = build_execution_vector_interactions(
        scores,
        pa=pa,
        gp=gp or {},
    )
    setup_conviction = build_setup_conviction(
        setups,
        corroboration_parts=corroboration_parts or parts,
    )
    authority_self_audit = build_authority_self_audit(
        authority_engine=str(grammar.get("authority_engine") or "legacy"),
        authority_mode=str(grammar.get("mode") or "full_stack"),
        bucket_conditioned_spine=bucket_conditioned_spine,
        raw_permission=raw_permission,
        permission=permission,
    )
    card = {
        "schema": "sharpedge.trade_permission.v1",
        "raw_vector_permission_score": raw_permission,
        "trade_permission_score": permission,
        "execution_permission_score": permission,
        "trade_gate": prim.gate_label(permission),
        "bias": bias_label,
        "bias_strength": bias_strength,
        "setup_conviction": setup_conviction,
        "scores": scores,
        "structure_state": structure_state or {},
        "acceptance_state": acceptance_state or {},
        "location_state": location_state or {},
        "dealer_state": dealer_state or {},
        "volume_state": volume_state or {},
        "trend_state": trend_state or {},
        "time_state": time_state or {},
        "graph_state": graph_state or {},
        "line_authority": line_authority or {},
        "market_day": market_day,
        "execution_flow": build_execution_flow(
            market_day,
            bucket_conditioned_spine,
            permission,
            bias_label,
        ),
        "execution_hierarchy": build_execution_hierarchy(
            parts,
            score_weights,
            graph_state,
        ),
        "execution_expansion_potential": expansion_potential,
        "execution_vector_interactions": vector_interactions,
        "bucket_conditioned_spine": bucket_conditioned_spine,
        "authority_self_audit": authority_self_audit,
        "spine_phase_model": spine_phase_model,
        "execution_grammar": grammar,
        "authority_engine": str(grammar.get("authority_engine") or "legacy"),
        "authority_mode": str(grammar.get("mode") or "full_stack"),
        "balance_confluence": pa.get("balance_confluence") or {},
        "balance_disagreement": pa.get("balance_disagreement") or {},
        "dominant_balance_flip": pa.get("dominant_balance_flip") or {},
        "supporting_reasons": reasons["supporting"],
        "warning_reasons": reasons["warnings"],
        "authority_adjudication": _build_authority_adjudication(
            setup_conviction=setup_conviction,
            market_day=market_day,
            bucket_conditioned_spine=bucket_conditioned_spine,
            expansion_potential=expansion_potential,
            reasons=reasons,
            grammar=grammar,
            permission=permission,
            bias_label=bias_label,
            authority_self_audit=authority_self_audit,
        ),
    }
    return sync_setup_evidence_fields(card)


__all__ = ["build_execution_flow", "build_trade_permission_card"]
