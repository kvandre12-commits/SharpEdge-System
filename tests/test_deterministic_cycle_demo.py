from __future__ import annotations

import pytest

from scripts.agents.compounding_agent import build_candidate, summarize_candidates
from scripts.agents.deterministic_cycle_demo import run_cycle, write_atomic


def test_one_cycle_is_deterministic_and_content_addressed():
    first = run_cycle()
    second = run_cycle()

    assert first == second
    assert first["receipt_id"] == second["receipt_id"]
    assert first["commit"]["atomic_record"] is True


def test_failed_observer_does_not_block_position_protection():
    cycle = run_cycle()
    components = cycle["components"]

    assert components["observer"]["status"] == "failed"
    assert components["helios"]["status"] == "abstained"
    assert components["position_steward"]["status"] == "completed"
    assert cycle["adjudication"]["pre_fill"]["resolution"] == "ABSTAIN_REFRESH_REQUIRED"
    assert cycle["adjudication"]["post_fill"]["resolution"] == "OWNER_EXIT_REVIEW"
    assert cycle["team_result"] == {
        "status": "PROTECTIVE_ONLY",
        "new_entry_eligible": False,
        "existing_position_reviewed": True,
        "resolution": "OWNER_EXIT_REVIEW",
    }


def test_compounding_agent_learns_only_for_future_evaluation():
    candidate = run_cycle()["compounding_agent"]

    assert candidate["disposition"] == "retain_for_evaluation"
    assert candidate["observed_delta_pct"] == 15.0
    assert candidate["promotion_eligible"] is False
    assert candidate["applied_to_current_cycle"] is False
    assert candidate["causality_claimed"] is False
    assert candidate["authority"] == {
        "authoritative": False,
        "execution_permitted": False,
        "broker_action_allowed": False,
        "may_modify_approval_decision": False,
        "may_modify_policy": False,
    }


def _candidate(cycle_number: int, observed_delta: float = 15.0):
    return build_candidate(
        cycle_receipt_id=f"cycle-receipt-{cycle_number}",
        adjudication={"receipt_id": f"adjudication-{cycle_number}"},
        outcome={
            "receipt_id": f"outcome-{cycle_number}",
            "observed_delta_pct": observed_delta,
        },
    )


def test_compounding_summary_requires_distinct_consistent_settled_cycles():
    summary = summarize_candidates([_candidate(1), _candidate(2), _candidate(3)])

    assert summary["evidence_count"] == 3
    assert summary["consistent"] is True
    assert summary["review_eligible"] is True
    assert summary["promotion_eligible"] is False
    assert summary["promotion_status"] == "human_review_required"
    assert summary["policy_modified"] is False
    assert summary["causality_claimed"] is False
    assert summary["authority"]["may_self_promote"] is False
    assert summary["authority"]["may_modify_policy"] is False


def test_duplicate_cycle_cannot_manufacture_compounding_confidence():
    candidate = _candidate(1)

    with pytest.raises(ValueError, match="duplicate cycle evidence"):
        summarize_candidates([candidate, candidate, candidate])


def test_inconsistent_evidence_stays_below_review_gate():
    summary = summarize_candidates([_candidate(1), _candidate(2, -4.0), _candidate(3)])

    assert summary["consistent"] is False
    assert summary["review_eligible"] is False
    assert summary["promotion_eligible"] is False
    assert summary["promotion_status"] == "collect_more_evidence"
    assert summary["candidate_lesson"] is None


def test_atomic_projection_is_byte_identical_on_repeat(tmp_path):
    output = tmp_path / "cycle.json"
    cycle = run_cycle()

    write_atomic(output, cycle)
    first = output.read_bytes()
    write_atomic(output, run_cycle())

    assert output.read_bytes() == first
    assert not list(tmp_path.glob(".cycle.json.*"))
