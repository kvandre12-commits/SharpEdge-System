"""Deterministic authority-safe adjudication for Observer/Helios/Owner disagreement."""

from __future__ import annotations

from typing import Any

from scripts.alpha_swarm.contracts import payload_sha256

SCHEMA = "sharpedge.agent_disagreement.v1"
PRE_FILL = "pre_fill"
POST_FILL = "post_fill"


def _decision(packet: dict[str, Any], default: str = "abstain") -> str:
    return str(packet.get("decision") or default).lower()


def _base(
    phase: str,
    observer: dict[str, Any],
    helios: dict[str, Any],
    owner: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "phase": phase,
        "inputs": {
            "observer": observer,
            "helios": helios,
            "owner": owner,
        },
        "authority": {
            "authoritative": False,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "approval_decision_unchanged": True,
        },
    }


def _finish(payload: dict[str, Any], **fields: Any) -> dict[str, Any]:
    payload.update(fields)
    payload["receipt_id"] = payload_sha256(payload)
    return payload


def adjudicate(
    *,
    phase: str,
    observer: dict[str, Any],
    helios: dict[str, Any],
    owner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve disagreement without escalating execution authority."""
    owner = owner or {}
    if phase not in {PRE_FILL, POST_FILL}:
        raise ValueError("phase must be pre_fill or post_fill")
    payload = _base(phase, observer, helios, owner)
    observer_fresh = observer.get("fresh") is True
    observer_hard_blocks = sorted(set(observer.get("hard_blocks") or []))
    observer_decision = _decision(observer)
    helios_decision = _decision(helios)
    owner_decision = _decision(owner, default="none")
    thesis_status = str(observer.get("thesis_status") or "unknown").lower()

    if phase == PRE_FILL and not observer_fresh:
        return _finish(
            payload,
            disagreement=True,
            disagreement_type="observation_unavailable",
            resolution="ABSTAIN_REFRESH_REQUIRED",
            effect="authority_reduced",
            reasons=["observer evidence is missing or stale"],
        )

    if phase == PRE_FILL:
        if observer_hard_blocks or observer_decision == "reject":
            return _finish(
                payload,
                disagreement=helios_decision == "approve",
                disagreement_type="observer_vetoes_entry",
                resolution="STAND_DOWN",
                effect="entry_authority_reduced",
                reasons=observer_hard_blocks or ["observer rejected entry"],
            )
        if helios_decision == "approve" and observer_decision == "approve":
            return _finish(
                payload,
                disagreement=False,
                disagreement_type="none",
                resolution="ENTRY_PROPOSAL_ELIGIBLE",
                effect="no_authority_change",
                reasons=["observer evidence and Helios proposal align"],
            )
        if helios_decision == "reject":
            return _finish(
                payload,
                disagreement=observer_decision == "approve",
                disagreement_type="helios_declines_observed_setup",
                resolution="STAND_DOWN",
                effect="entry_authority_reduced",
                reasons=list(helios.get("reasons") or ["Helios rejected entry"]),
            )
        return _finish(
            payload,
            disagreement=observer_decision != helios_decision,
            disagreement_type="incomplete_entry_consensus",
            resolution="ABSTAIN_REVIEW_REQUIRED",
            effect="authority_reduced",
            reasons=["entry evidence is incomplete or conflicting"],
        )

    verified_position = owner.get("verified_position") is True
    if not verified_position:
        return _finish(
            payload,
            disagreement=True,
            disagreement_type="ownership_unverified",
            resolution="ABSTAIN_POSITION_REFRESH_REQUIRED",
            effect="authority_reduced",
            reasons=["post-fill adjudication requires a verified owned position"],
        )
    if owner_decision in {"exit", "exit_review", "exit_proposed"}:
        return _finish(
            payload,
            disagreement=helios_decision == "hold" or not observer_fresh,
            disagreement_type=(
                "owner_protects_without_observer"
                if not observer_fresh
                else "owner_exits_despite_helios"
                if helios_decision == "hold"
                else "owner_exit"
            ),
            resolution="OWNER_EXIT_REVIEW",
            effect="exit_tightened",
            reasons=list(owner.get("reasons") or ["position owner requested exit review"]),
        )
    if not observer_fresh:
        return _finish(
            payload,
            disagreement=True,
            disagreement_type="observation_unavailable",
            resolution="ABSTAIN_REFRESH_REQUIRED",
            effect="authority_reduced",
            reasons=["observer evidence is missing or stale"],
        )
    if observer_hard_blocks or thesis_status == "broken":
        reasons = observer_hard_blocks or ["entry thesis is broken"]
        return _finish(
            payload,
            disagreement=helios_decision == "hold" or owner_decision == "hold",
            disagreement_type="facts_break_thesis",
            resolution="OWNER_EXIT_REVIEW",
            effect="exit_tightened",
            reasons=reasons,
        )
    if owner_decision == "hold" and helios_decision == "hold":
        return _finish(
            payload,
            disagreement=False,
            disagreement_type="none",
            resolution="OWNER_HOLD",
            effect="no_authority_change",
            reasons=["owner rules and current thesis context align"],
        )
    return _finish(
        payload,
        disagreement=True,
        disagreement_type="post_fill_policy_conflict",
        resolution="OWNER_REVIEW_REQUIRED",
        effect="no_automatic_action",
        reasons=["non-safety disagreement requires explicit owner review"],
    )


def build_outcome_receipt(
    adjudication: dict[str, Any],
    *,
    actual_return_pct: float,
    counterfactual_return_pct: float,
    resolution_minutes: float,
) -> dict[str, Any]:
    """Measure whether disagreement improved outcome; never infer causality."""
    actual = round(float(actual_return_pct), 4)
    counterfactual = round(float(counterfactual_return_pct), 4)
    payload = {
        "schema": "sharpedge.agent_disagreement_outcome.v1",
        "adjudication_receipt_id": adjudication.get("receipt_id"),
        "resolution": adjudication.get("resolution"),
        "disagreement": bool(adjudication.get("disagreement")),
        "actual_return_pct": actual,
        "counterfactual_return_pct": counterfactual,
        "observed_delta_pct": round(actual - counterfactual, 4),
        "resolution_minutes": round(float(resolution_minutes), 2),
        "interpretation": (
            "observed_better_than_counterfactual"
            if actual > counterfactual
            else "observed_worse_than_counterfactual"
            if actual < counterfactual
            else "observed_equal_to_counterfactual"
        ),
        "causality_claimed": False,
        "authority": "historical_measurement_only",
    }
    payload["receipt_id"] = payload_sha256(payload)
    return payload


__all__ = ["POST_FILL", "PRE_FILL", "adjudicate", "build_outcome_receipt"]
