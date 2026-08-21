from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import dashboard_runtime as runtime  # noqa: E402
import make_operator_surface as operator_surface  # noqa: E402


def test_resolve_execution_context_falls_back_when_bridge_missing():
    with patch.dict(
        sys.modules, {"sharpedge_robinhood_bridge.analytics_context": None}
    ):
        ctx = runtime.resolve_execution_context(symbol="SPY")

    assert ctx.available is False
    assert "unavailable" in ctx.note.lower()


def test_resolve_decision_falls_back_to_signal_only_mode_when_bridge_missing():
    signal = {"setup_bias": "Post-selloff coil near breakdown trigger."}
    ctx = runtime.unavailable_context("bridge missing")

    with patch.dict(sys.modules, {"sharpedge_robinhood_bridge.trade_intent": None}):
        decision = runtime.resolve_decision(signal, ctx)

    assert decision["action"] == "stand_down"
    assert decision["intent"] is None
    assert "coil" in decision["reason"].lower()


def test_operator_surface_render_uses_local_operator_artifacts(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "workflow_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "symbol": "SPY",
                "lifecycle_stage": "review_required",
                "operator_action": "review_setup",
                "readiness": "ready_for_review",
                "approval_decision": "operator_confirm_required",
                "blockers": ["manual_review_required"],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "execution_plan.json").write_text(
        json.dumps(
            {
                "steps": [
                    "Review the failed-breakdown reclaim.",
                    "Check connector response.",
                ],
                "blocking_reasons": ["manual_review_required"],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "approval_decision.json").write_text(
        json.dumps(
            {
                "decision": "operator_confirm_required",
                "trade_allowed": False,
                "broker_order_allowed": False,
                "required_human_action": "review",
            }
        ),
        encoding="utf-8",
    )
    (outputs / "operator_brief.json").write_text(
        json.dumps(
            {
                "headline": "Review the reclaim before acting.",
                "operator_action": "review_setup",
                "summary": {
                    "broker_integration_status": "ready",
                    "monitoring_mode": "mcp_quote_monitoring",
                    "risk_state": "STANDARD",
                },
                "risk": {
                    "blocking_reasons": ["manual_review_required"],
                    "risk_flags": ["late_session", "connector_pending"],
                },
                "options_liquidity_read": {
                    "available": True,
                    "plain_english": "Calls are winning right now even though the thesis was cautious.",
                    "liquidity_spot": "2026-06-27 CALL 501 mid 1.20 bid/ask 1.10/1.30 vol 900 OI 2100 (near-money-flow).",
                    "flow_balance": "Crosswired tape; CALLs are leading quote-weighted focus-line pressure by 2.10x against the stated bias.",
                    "bias_alignment": "crosswired",
                    "quote_quality_context": "Both sides have usable enough quotes for the flow comparison to matter.",
                    "put_pressure_score": 1200.0,
                    "call_pressure_score": 2520.0,
                    "put_pressure_pct": 32,
                    "call_pressure_pct": 68,
                    "dominant_side": "call",
                    "put_flow": [
                        "2026-06-27 PUT 499 mid 0.90 bid/ask 0.85/0.95 vol 500 OI 1700 (near-money-flow)."
                    ],
                    "call_flow": [
                        "2026-06-27 CALL 501 mid 1.20 bid/ask 1.10/1.30 vol 900 OI 2100 (near-money-flow)."
                    ],
                    "put_side_summary": "PUT side quality usable; strongest visible lines are 2026-06-27 PUT 499 mid 0.90 bid/ask 0.85/0.95 vol 500 OI 1700 (near-money-flow).",
                    "call_side_summary": "CALL side quality usable; strongest visible lines are 2026-06-27 CALL 501 mid 1.20 bid/ask 1.10/1.30 vol 900 OI 2100 (near-money-flow).",
                    "watch_next": [
                        "Watch 501 acceptance before trusting the call lead."
                    ],
                },
                "next_steps": ["Review reclaim", "Wait for connector feedback"],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "operator_session_review.json").write_text(
        json.dumps({"latest_headline": "Review the reclaim before acting."}),
        encoding="utf-8",
    )
    (outputs / "operator_watchlist.json").write_text(
        json.dumps(
            {
                "active_count": 2,
                "items": [
                    {
                        "headline": "21 DTE ATM CALLS watch for failed breakdown",
                        "setup_type": "atm_options_thesis",
                        "option_side": "CALLS",
                        "dte_target": 21,
                        "status": "ready_for_review",
                        "spot": 501.25,
                        "atm_strike": 501.0,
                        "dealer_state_hint": "EXPANSION",
                    }
                ],
                "omitted_candidates": [
                    {
                        "headline": "1 DTE ATM CALLS execution watch",
                        "setup_type": "atm_options_execution",
                        "option_side": "CALLS",
                        "dte_target": 1,
                        "status": "removed",
                        "spot": 501.25,
                        "atm_strike": 501.0,
                        "dealer_state_hint": "EXPANSION",
                        "invalidation_reason": "permission_score_trend_weakening_-9",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "signal.json").write_text(
        json.dumps(
            {
                "ts": "2026-06-27T16:05:00",
                "spot": 501.25,
                "vs_vwap": 0.12,
                "mom15": 0.45,
                "gamma_regime": "negative",
                "entry_setup_tag": "FAILED BREAKDOWN",
                "entry_setup_bias": "CALLS (bullish)",
                "edge_token_position": {
                    "suggested_action": "hold",
                    "position_state": "open",
                    "contracts_held": 1,
                    "action_reason": "edge token is still active; keep the single-contract position on.",
                    "recommended_actions": ["hold"],
                    "current_token": {
                        "event_type": "FAILED BREAKDOWN",
                        "side": "CALLS",
                        "status": "confirmed",
                        "level_name": "ORL",
                        "level_price": 500.9,
                    },
                },
                "trade_permission": {
                    "trade_gate": "PERMIT",
                    "execution_permission_score": 88,
                    "setup_conviction": {
                        "setup_gate": "ACTIONABLE",
                        "setup_conviction_score": 91,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "robinhood_beta_execution.json").write_text(
        json.dumps(
            {
                "beta_stage": "position_hold",
                "edge_token_position": {"contracts_held": 1},
                "order_preview": {
                    "token_action": "hold",
                    "position_intent": "hold",
                    "strategy_family": "call_debit_spread",
                    "draft_allowed": False,
                    "recommended_actions": ["hold"],
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "operator_journal_append.jsonl").write_text(
        json.dumps(
            {
                "headline": "Review the reclaim before acting.",
                "created_ts": "2026-06-27T16:00:00+00:00",
                "watchlist_status": "ready_for_review",
                "risk_state": "STANDARD",
                "blocking_reasons": ["manual_review_required"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        patch.object(operator_surface, "OUTPUTS_DIR", outputs),
        patch.object(
            operator_surface,
            "_run_git",
            side_effect=[
                [" M cockpit/make_cockpit.py"],
                ["abc123 Add operator surface"],
            ],
        ),
    ):
        html = operator_surface.render()

    assert "SharpEdge Operator Surface" in html
    assert "Review the reclaim before acting." in html
    assert "manual_review_required" in html
    assert "first-class surfaces: cockpit.html • operator_surface.html" in html
    assert "artifact freshness" in html
    assert "live cockpit snapshot" in html
    assert "execution state" in html
    assert "position_hold" in html
    assert "token action" in html
    assert "hold" in html
    assert "FAILED BREAKDOWN" in html
    assert "latest operator journal" in html
    assert "options liquidity" in html
    assert "Put side" in html
    assert "Call side" in html
    assert "Pressure split" in html
    assert "lead: call" in html
    assert "2026-06-27 PUT 499" in html
    assert "2026-06-27 CALL 501" in html
    assert "21 DTE ATM CALLS watch for failed breakdown" in html
    assert "permission_score_trend_weakening_-9" in html
    assert "connector_pending" in html
