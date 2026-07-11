from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from decision_receipts import build_decision_receipt  # noqa: E402
from edge_token_manager import build_edge_token_position  # noqa: E402


def _permission(bias: str) -> dict:
    return {
        "trade_permission_score": 72,
        "execution_permission_score": 72,
        "trade_gate": "CAUTION",
        "bias": bias,
        "setup_conviction": {
            "setup_gate": "ACTIONABLE",
            "setup_conviction_score": 84,
            "bias": bias,
            "setup_tag": "FAILED BREAKDOWN" if bias == "CALLS" else "FAILED BREAKOUT",
        },
    }


def test_edge_token_manager_waits_for_confirmation_then_enters_holds_and_closes():
    candidate = build_decision_receipt(
        "2026-06-25T10:12:00",
        "SPY",
        733.0,
        _permission("CALLS"),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
    )
    pending = build_edge_token_position(
        candidate["ts"], _permission("CALLS"), candidate
    )

    confirmed = build_decision_receipt(
        "2026-06-25T10:15:00",
        "SPY",
        733.3,
        _permission("CALLS"),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
        previous_receipt=candidate,
    )
    enter = build_edge_token_position(
        confirmed["ts"], _permission("CALLS"), confirmed, pending
    )

    still_confirmed = build_decision_receipt(
        "2026-06-25T10:18:00",
        "SPY",
        733.4,
        _permission("CALLS"),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
                "trigger_price": 732.4,
                "bars_ago": 1,
            }
        ],
        previous_receipt=confirmed,
    )
    hold = build_edge_token_position(
        still_confirmed["ts"], _permission("CALLS"), still_confirmed, enter
    )

    expired = build_decision_receipt(
        "2026-06-25T10:41:00",
        "SPY",
        733.8,
        _permission("NEUTRAL"),
        {"label": "Magnet", "price": 734.0},
        [],
        previous_receipt=still_confirmed,
    )
    close = build_edge_token_position(
        expired["ts"], _permission("NEUTRAL"), expired, hold
    )

    assert pending["suggested_action"] == "stand_down"
    assert pending["position_state"] == "flat"
    assert pending["contracts_delta"] == 0
    assert pending["token_status"] == "pending_confirmation"
    assert pending["pending_token"]["status"] == "candidate"

    assert enter["suggested_action"] == "enter_call"
    assert enter["position_state"] == "open"
    assert enter["contracts_delta"] == 1
    assert enter["current_token"]["status"] == "confirmed"

    assert hold["suggested_action"] == "hold"
    assert hold["contracts_held"] == 1
    assert hold["current_token"]["status"] == "confirmed"

    assert close["suggested_action"] == "close_position"
    assert close["position_state"] == "flat"
    assert close["contracts_delta"] == -1
    assert close["closing_token"]["clear_reason"] == "expired"


def test_edge_token_manager_ignores_directional_context_cards():
    permission = _permission("CALLS")
    permission["setup_conviction"] = {
        "setup_gate": "CONTEXT",
        "setup_conviction_score": 52,
        "bias": "CALLS",
        "setup_tag": "STICKY DAY (calm/chop)",
    }
    receipt = build_decision_receipt(
        "2026-06-25T10:12:00",
        "SPY",
        733.0,
        permission,
        {"label": "Magnet", "price": 734.0},
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - CALLS below magnet / PUTS above magnet",
                "kind": "info",
                "detail": "positive gamma context only",
            }
        ],
    )

    state = build_edge_token_position(receipt["ts"], permission, receipt)

    assert state["suggested_action"] == "stand_down"
    assert state["token_status"] == "none"
    assert state["current_token"] is None
    assert state["pending_token"] is None
    assert state["action_reason"] == "no active directional edge token"


def test_edge_token_manager_closes_before_reentry_when_direction_replaces_old_one():
    calls_candidate = build_decision_receipt(
        "2026-06-25T10:12:00",
        "SPY",
        733.0,
        _permission("CALLS"),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
            }
        ],
    )
    calls_pending = build_edge_token_position(
        calls_candidate["ts"],
        _permission("CALLS"),
        calls_candidate,
    )
    calls_confirmed = build_decision_receipt(
        "2026-06-25T10:15:00",
        "SPY",
        733.2,
        _permission("CALLS"),
        {"label": "VWAP", "price": 734.5},
        [
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "level_name": "ORL",
                "level_price": 732.9,
            }
        ],
        previous_receipt=calls_candidate,
    )
    open_calls = build_edge_token_position(
        calls_confirmed["ts"],
        _permission("CALLS"),
        calls_confirmed,
        calls_pending,
    )

    puts_receipt = build_decision_receipt(
        "2026-06-25T10:18:00",
        "SPY",
        732.1,
        _permission("PUTS"),
        {"label": "VWAP", "price": 731.5},
        [
            {
                "tag": "FAILED BREAKOUT",
                "bias": "PUTS (bearish)",
                "kind": "bad",
                "level_name": "ORH",
                "level_price": 733.4,
            }
        ],
        previous_receipt=calls_confirmed,
    )
    reset = build_edge_token_position(
        puts_receipt["ts"],
        _permission("PUTS"),
        puts_receipt,
        open_calls,
    )

    assert reset["suggested_action"] == "close_position"
    assert reset["recommended_actions"] == ["close_position"]
    assert reset["position_state"] == "flat"
    assert reset["contracts_delta"] == -1
    assert reset["token_status"] == "reset_required"
    assert reset["pending_token"]["side"] == "PUTS"
    assert reset["closing_token"]["token_id"] == open_calls["current_token"]["token_id"]

    puts_confirmed = build_decision_receipt(
        "2026-06-25T10:21:00",
        "SPY",
        731.9,
        _permission("PUTS"),
        {"label": "VWAP", "price": 731.5},
        [
            {
                "tag": "FAILED BREAKOUT",
                "bias": "PUTS (bearish)",
                "kind": "bad",
                "level_name": "ORH",
                "level_price": 733.4,
            }
        ],
        previous_receipt=puts_receipt,
    )
    enter_put = build_edge_token_position(
        puts_confirmed["ts"],
        _permission("PUTS"),
        puts_confirmed,
        reset,
    )

    assert enter_put["suggested_action"] == "enter_put"
    assert enter_put["position_state"] == "open"
    assert enter_put["contracts_delta"] == 1
    assert enter_put["current_token"]["side"] == "PUTS"
