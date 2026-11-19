from __future__ import annotations

import pytest

from scripts.agents.call_cycle_state import (
    CycleTransitionError,
    initial_state,
    transition,
)


def _proposed():
    return transition(
        initial_state("SPY"),
        {
            "type": "lead_approved",
            "trigger_id": "trigger-1",
            "lead_id": "lead-1",
            "max_debit_dollars": 240,
            "ladder": [{"contract_symbol": "SPY_CALL", "quantity": 2}],
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

    opened = _open()
    assert opened["state"] == "OPEN_POSITION"
    assert opened["position"]["fill_receipt_id"] == "fill-1"

    managing = transition(opened, {"type": "position_mark", "return_pct": 20})
    assert managing["state"] == "SELLER_MANAGING"
    assert managing["position"]["management_band"] == "protect_and_review"

    tightened = transition(managing, {"type": "position_mark", "return_pct": 35})
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


def test_one_contract_is_not_sold_until_seller_exit_proposal_and_verified_close():
    managing = transition(_open(), {"type": "position_mark", "return_pct": 55})
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
