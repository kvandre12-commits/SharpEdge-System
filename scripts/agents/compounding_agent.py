"""Deterministic, authority-free lessons from settled team outcomes."""

from __future__ import annotations

from typing import Any

from scripts.agents.receipt_utils import payload_sha256

SCHEMA = "sharpedge.compounding_candidate.v1"


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


__all__ = ["SCHEMA", "build_candidate"]
