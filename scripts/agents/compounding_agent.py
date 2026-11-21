"""Deterministic, authority-free lessons from settled team outcomes."""

from __future__ import annotations

from typing import Any

from scripts.agents.receipt_utils import payload_sha256

SCHEMA = "sharpedge.compounding_candidate.v1"
SUMMARY_SCHEMA = "sharpedge.compounding_evidence_summary.v1"


def build_candidate(
    *,
    cycle_receipt_id: str,
    adjudication: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    """Build a next-cycle lesson candidate without modifying current authority."""
    observed_delta = round(float(outcome.get("observed_delta_pct", 0.0)), 4)
    if observed_delta > 0:
        disposition = "retain_for_evaluation"
        lesson = "preserve independent position protection when entry evidence fails"
    elif observed_delta < 0:
        disposition = "review_for_regression"
        lesson = "review protective escalation; observed counterfactual performed better"
    else:
        disposition = "insufficient_evidence"
        lesson = "collect more settled outcomes before proposing a policy preference"

    payload = {
        "schema": SCHEMA,
        "cycle_receipt_id": cycle_receipt_id,
        "adjudication_receipt_id": adjudication.get("receipt_id"),
        "outcome_receipt_id": outcome.get("receipt_id"),
        "observed_delta_pct": observed_delta,
        "disposition": disposition,
        "candidate_lesson": lesson,
        "evidence_count": 1,
        "promotion_eligible": False,
        "applied_to_current_cycle": False,
        "causality_claimed": False,
        "authority": {
            "authoritative": False,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "may_modify_approval_decision": False,
            "may_modify_policy": False,
        },
    }
    payload["receipt_id"] = payload_sha256(payload)
    return payload


def summarize_candidates(
    candidates: list[dict[str, Any]], *, minimum_evidence: int = 3
) -> dict[str, Any]:
    """Aggregate distinct settled evidence without granting policy authority."""
    if minimum_evidence < 2:
        raise ValueError("minimum_evidence must be at least 2")
    if not candidates:
        raise ValueError("at least one candidate is required")

    cycle_ids = [str(candidate.get("cycle_receipt_id") or "") for candidate in candidates]
    if any(not cycle_id for cycle_id in cycle_ids):
        raise ValueError("every candidate requires a cycle_receipt_id")
    if len(set(cycle_ids)) != len(cycle_ids):
        raise ValueError("duplicate cycle evidence cannot compound confidence")

    lessons = {str(candidate.get("candidate_lesson") or "") for candidate in candidates}
    dispositions = {str(candidate.get("disposition") or "") for candidate in candidates}
    consistent = len(lessons) == 1 and len(dispositions) == 1
    evidence_count = len(candidates)
    review_eligible = consistent and evidence_count >= minimum_evidence
    observed_deltas = [float(candidate.get("observed_delta_pct", 0.0)) for candidate in candidates]

    payload = {
        "schema": SUMMARY_SCHEMA,
        "source_candidate_receipts": sorted(
            str(candidate.get("receipt_id") or "") for candidate in candidates
        ),
        "source_cycle_receipts": sorted(cycle_ids),
        "evidence_count": evidence_count,
        "minimum_evidence": minimum_evidence,
        "consistent": consistent,
        "candidate_lesson": next(iter(lessons)) if consistent else None,
        "mean_observed_delta_pct": round(sum(observed_deltas) / evidence_count, 4),
        "review_eligible": review_eligible,
        "promotion_eligible": False,
        "promotion_status": "human_review_required" if review_eligible else "collect_more_evidence",
        "policy_modified": False,
        "causality_claimed": False,
        "authority": {
            "authoritative": False,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "may_modify_approval_decision": False,
            "may_modify_policy": False,
            "may_self_promote": False,
        },
    }
    payload["receipt_id"] = payload_sha256(payload)
    return payload


__all__ = ["SCHEMA", "SUMMARY_SCHEMA", "build_candidate", "summarize_candidates"]
