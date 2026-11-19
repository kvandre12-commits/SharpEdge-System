from __future__ import annotations

from scripts.agents.disagreement_adjudicator import (
    POST_FILL,
    PRE_FILL,
    adjudicate,
    build_outcome_receipt,
)


def _observer(**overrides):
    return {
        "fresh": True,
        "decision": "approve",
        "hard_blocks": [],
        "thesis_status": "valid",
        **overrides,
    }


def _helios(**overrides):
    return {"decision": "approve", "reasons": [], **overrides}


def _owner(**overrides):
    return {
        "verified_position": True,
        "decision": "hold",
        "reasons": [],
        **overrides,
    }


def test_aligned_prefill_evidence_may_only_create_entry_proposal():
    receipt = adjudicate(
        phase=PRE_FILL,
        observer=_observer(),
        helios=_helios(),
    )

    assert receipt["resolution"] == "ENTRY_PROPOSAL_ELIGIBLE"
    assert receipt["disagreement"] is False
    assert receipt["authority"]["execution_permitted"] is False
    assert receipt["authority"]["approval_decision_unchanged"] is True


def test_observer_hard_fact_veto_beats_helios_entry_approval():
    receipt = adjudicate(
        phase=PRE_FILL,
        observer=_observer(decision="reject", hard_blocks=["stale_option_quote"]),
        helios=_helios(decision="approve"),
    )

    assert receipt["resolution"] == "STAND_DOWN"
    assert receipt["disagreement"] is True
    assert receipt["effect"] == "entry_authority_reduced"


def test_stale_observer_forces_abstention_not_majority_vote():
    receipt = adjudicate(
        phase=PRE_FILL,
        observer=_observer(fresh=False),
        helios=_helios(decision="approve"),
    )

    assert receipt["resolution"] == "ABSTAIN_REFRESH_REQUIRED"
    assert receipt["effect"] == "authority_reduced"


def test_owner_exit_review_wins_over_helios_hold_after_fill():
    receipt = adjudicate(
        phase=POST_FILL,
        observer=_observer(),
        helios=_helios(decision="hold"),
        owner=_owner(decision="exit_review", reasons=["theta acceleration"]),
    )

    assert receipt["resolution"] == "OWNER_EXIT_REVIEW"
    assert receipt["disagreement_type"] == "owner_exits_despite_helios"
    assert receipt["effect"] == "exit_tightened"


def test_owner_protection_survives_observer_failure_after_fill():
    receipt = adjudicate(
        phase=POST_FILL,
        observer=_observer(fresh=False, decision="abstain"),
        helios=_helios(decision="abstain"),
        owner=_owner(decision="exit_review", reasons=["protect verified position"]),
    )

    assert receipt["resolution"] == "OWNER_EXIT_REVIEW"
    assert receipt["disagreement_type"] == "owner_protects_without_observer"
    assert receipt["effect"] == "exit_tightened"
    assert receipt["authority"]["broker_action_allowed"] is False


def test_broken_thesis_accelerates_review_even_when_helios_and_owner_hold():
    receipt = adjudicate(
        phase=POST_FILL,
        observer=_observer(thesis_status="broken"),
        helios=_helios(decision="hold"),
        owner=_owner(decision="hold"),
    )

    assert receipt["resolution"] == "OWNER_EXIT_REVIEW"
    assert receipt["disagreement_type"] == "facts_break_thesis"


def test_verified_position_is_required_for_postfill_authority():
    receipt = adjudicate(
        phase=POST_FILL,
        observer=_observer(),
        helios=_helios(decision="hold"),
        owner=_owner(verified_position=False),
    )

    assert receipt["resolution"] == "ABSTAIN_POSITION_REFRESH_REQUIRED"


def test_non_safety_disagreement_requires_review_without_automatic_action():
    receipt = adjudicate(
        phase=POST_FILL,
        observer=_observer(),
        helios=_helios(decision="exit_review"),
        owner=_owner(decision="hold"),
    )

    assert receipt["resolution"] == "OWNER_REVIEW_REQUIRED"
    assert receipt["effect"] == "no_automatic_action"
    assert receipt["authority"]["broker_action_allowed"] is False


def test_outcome_receipt_measures_delta_without_claiming_causality():
    adjudication = adjudicate(
        phase=POST_FILL,
        observer=_observer(thesis_status="broken"),
        helios=_helios(decision="hold"),
        owner=_owner(decision="hold"),
    )

    outcome = build_outcome_receipt(
        adjudication,
        actual_return_pct=-5,
        counterfactual_return_pct=-20,
        resolution_minutes=3.5,
    )

    assert outcome["observed_delta_pct"] == 15
    assert outcome["interpretation"] == "observed_better_than_counterfactual"
    assert outcome["causality_claimed"] is False
    assert outcome["authority"] == "historical_measurement_only"


def test_same_inputs_produce_same_receipt_id():
    left = adjudicate(phase=PRE_FILL, observer=_observer(), helios=_helios())
    right = adjudicate(phase=PRE_FILL, observer=_observer(), helios=_helios())
    assert left["receipt_id"] == right["receipt_id"]
