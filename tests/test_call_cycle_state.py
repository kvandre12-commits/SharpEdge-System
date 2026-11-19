from __future__ import annotations

import pytest

from scripts.agents.call_cycle_state import (
    CycleTransitionError,
    initial_state,
    transition,
)
from scripts.alpha_swarm.contracts import payload_sha256


def _thesis():
    thesis = {
        "schema": "sharpedge.helios_entry_thesis.v1",
        "captured_at": "2026-11-19T17:05:00+00:00",
        "symbol": "SPY",
        "trigger": {"triggered": True},
        "nearest_refill_target": 739.95,
        "strategy": "long_otm_call_wallet_ladder",
        "seller_policy_at_entry": {"profit_management_starts_pct": 20},
        "source_hashes": {"lead": "a" * 64},
        "owner_usage": {
            "context_only": True,
            "may_tighten_exit": True,
            "may_loosen_risk_or_expiry_exit": False,
            "may_authorize_add_or_rebuy": False,
        },
    }
    thesis["thesis_id"] = payload_sha256(thesis)
    return thesis


def _proposed():
    return transition(
        initial_state("SPY"),
        {
            "type": "lead_approved",
            "trigger_id": "trigger-1",
            "lead_id": "lead-1",
            "max_debit_dollars": 240,
            "ladder": [{"contract_symbol": "SPY_CALL", "quantity": 2}],
            "entry_thesis_snapshot": _thesis(),
        },
    )


def _open():
    return transition(
        _proposed(),
        {
            "type": "verified_fill",
            "verified": True,
            "receipt_id": "fill-1",
            "filled_debit_dollars": 235,
            "contracts": [{"contract_symbol": "SPY_CALL", "quantity": 2}],
        },
    )


def test_full_buyer_to_seller_to_recycled_wallet_handoff():
    proposed = _proposed()
    assert proposed["state"] == "ENTRY_PROPOSED"
    assert proposed["reserved_dollars"] == 240
    assert proposed["entry_thesis_snapshot"]["thesis_id"] == _thesis()["thesis_id"]

    opened = _open()
    assert opened["state"] == "OPEN_POSITION"
    assert opened["position"]["fill_receipt_id"] == "fill-1"

    managing = transition(
        opened, {"type": "position_mark", "return_pct": 20, "thesis_status": "valid"}
    )
    assert managing["state"] == "SELLER_MANAGING"
    assert managing["position"]["management_band"] == "protect_and_review"

    tightened = transition(
        managing, {"type": "position_mark", "return_pct": 35, "thesis_status": "valid"}
    )
    assert tightened["position"]["management_band"] == "tighten_exit_review"

    exit_pending = transition(
        tightened, {"type": "exit_proposed", "reason": "seller_take_profit"}
    )
    assert exit_pending["state"] == "EXIT_PENDING_CONFIRM"

    closed = transition(
        exit_pending,
        {
            "type": "verified_close",
            "verified": True,
            "receipt_id": "close-1",
            "proceeds_dollars": 320,
        },
    )
    assert closed["state"] == "CLOSED"
    assert closed["realized_pnl_dollars"] == 85
    assert closed["reserved_dollars"] == 0

    cooldown = transition(closed, {"type": "begin_cooldown"})
    recycled = transition(cooldown, {"type": "trigger_reset"})
    assert recycled["state"] == "IDLE"
    assert recycled["reserved_dollars"] == 0
    assert recycled["previous_trigger_id"] == "trigger-1"


def test_same_trigger_cannot_rebuy_while_position_is_open():
    with pytest.raises(CycleTransitionError, match="IDLE"):
        transition(
            _open(),
            {
                "type": "lead_approved",
                "trigger_id": "trigger-1",
                "max_debit_dollars": 100,
                "entry_thesis_snapshot": _thesis(),
            },
        )


def test_unverified_or_over_budget_fill_is_rejected():
    with pytest.raises(CycleTransitionError, match="explicitly verified"):
        transition(
            _proposed(),
            {
                "type": "verified_fill",
                "verified": False,
                "filled_debit_dollars": 200,
                "contracts": [{"contract_symbol": "SPY_CALL"}],
            },
        )
    with pytest.raises(CycleTransitionError, match="exceeds reserved"):
        transition(
            _proposed(),
            {
                "type": "verified_fill",
                "verified": True,
                "filled_debit_dollars": 250,
                "contracts": [{"contract_symbol": "SPY_CALL"}],
            },
        )


def test_broken_thesis_can_tighten_exit_but_valid_thesis_cannot_loosen_rules():
    opened = _open()
    valid = transition(
        opened, {"type": "position_mark", "return_pct": 10, "thesis_status": "valid"}
    )
    broken = transition(
        opened, {"type": "position_mark", "return_pct": 10, "thesis_status": "broken"}
    )

    assert valid["state"] == "OPEN_POSITION"
    assert broken["state"] == "SELLER_MANAGING"
    assert broken["position"]["management_band"] == "thesis_broken_exit_review"


def test_tampered_entry_thesis_is_rejected():
    thesis = _thesis()
    thesis["nearest_refill_target"] = 999
    with pytest.raises(CycleTransitionError, match="hash"):
        transition(
            initial_state("SPY"),
            {
                "type": "lead_approved",
                "trigger_id": "trigger-1",
                "max_debit_dollars": 100,
                "entry_thesis_snapshot": thesis,
            },
        )


def test_one_contract_is_not_sold_until_seller_exit_proposal_and_verified_close():
    managing = transition(
        _open(), {"type": "position_mark", "return_pct": 55, "thesis_status": "valid"}
    )
    assert managing["state"] == "SELLER_MANAGING"
    assert managing["position"]["management_band"] == "strong_take_profit_review"
    assert managing["broker_action_allowed"] is False
    with pytest.raises(CycleTransitionError, match="EXIT_PENDING_CONFIRM"):
        transition(
            managing,
            {
                "type": "verified_close",
                "verified": True,
                "proceeds_dollars": 350,
            },
        )
