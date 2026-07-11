from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import robinhood_beta_execution as beta


class RobinhoodBetaExecutionTests(unittest.TestCase):
    def test_build_payload_prepares_approval_queue_when_bridge_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(Path(tmp), trade_allowed=True)
            with self._patched_paths(outputs):
                payload = beta.build_payload()

        self.assertEqual(payload["beta_stage"], "approval_queue_ready")
        self.assertTrue(payload["beta_capabilities"]["create_order_draft"])
        self.assertFalse(payload["beta_capabilities"]["submit_order"])
        self.assertEqual(
            payload["order_preview"]["strategy_family"], "put_debit_spread"
        )
        self.assertEqual(payload["order_preview"]["token_action"], "enter_put")
        self.assertEqual(payload["order_preview"]["position_intent"], "entry")
        self.assertTrue(payload["approval_required"])

    def test_build_payload_falls_back_to_artifact_only_when_bridge_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp), trade_allowed=False, bridge_available=False
            )
            with self._patched_paths(outputs):
                payload = beta.build_payload()
                text = beta.render_text(payload)

        self.assertEqual(payload["beta_stage"], "artifact_only")
        self.assertFalse(payload["beta_capabilities"]["create_order_draft"])
        self.assertIn(
            "artifact_only_order_preview",
            payload["robinhood_beta_handoff"]["permitted_actions"],
        )
        self.assertIn("ROBINHOOD BETA EXECUTION HANDOFF", text)

    def test_order_preview_caps_risk_to_beta_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp), trade_allowed=True, contract_risk=0.40
            )
            with self._patched_paths(outputs):
                with patch.object(beta, "BETA_MAX_RISK_CAP_PCT", 0.25):
                    payload = beta.build_payload()

        self.assertEqual(
            payload["order_preview"]["risk_limits"]["source_contract_risk_pct"], 0.4
        )
        self.assertEqual(
            payload["order_preview"]["risk_limits"]["max_capital_risk_pct"], 0.25
        )

    def test_build_payload_holds_without_new_draft_when_token_stays_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp), trade_allowed=True, token_action="hold"
            )
            with self._patched_paths(outputs):
                payload = beta.build_payload()

        self.assertEqual(payload["beta_stage"], "position_hold")
        self.assertFalse(payload["beta_capabilities"]["create_order_draft"])
        self.assertEqual(payload["order_preview"]["position_intent"], "hold")
        self.assertEqual(payload["order_preview"]["token_action"], "hold")

    def test_build_payload_prepares_close_review_when_token_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp),
                trade_allowed=False,
                token_action="close_position",
            )
            with self._patched_paths(outputs):
                payload = beta.build_payload()

        self.assertEqual(payload["beta_stage"], "close_review_ready")
        self.assertTrue(payload["beta_capabilities"]["create_order_draft"])
        self.assertEqual(payload["order_preview"]["position_intent"], "close")
        self.assertEqual(
            payload["order_preview"]["strategy_family"], "close_existing_position"
        )

    def test_build_payload_marks_pressure_point_entry_review_ready_when_blocked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp),
                trade_allowed=False,
                token_action="enter_put",
            )
            with self._patched_paths(outputs):
                payload = beta.build_payload()

        self.assertEqual(payload["beta_stage"], "pressure_point_review_ready")
        self.assertFalse(payload["beta_capabilities"]["create_order_draft"])
        self.assertFalse(payload["order_preview"]["draft_allowed"])
        self.assertTrue(payload["order_preview"]["draft_review_allowed"])
        self.assertEqual(
            payload["order_preview"]["setup_type"], "pressure_point_edge_token"
        )
        self.assertEqual(payload["order_preview"]["token_action"], "enter_put")
        self.assertEqual(payload["trade_allowed"], False)

    def test_build_payload_prepares_rotation_review_when_token_flips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(
                Path(tmp),
                trade_allowed=True,
                token_action="flip_to_put",
            )
            with self._patched_paths(outputs):
                payload = beta.build_payload()

        self.assertEqual(payload["beta_stage"], "rotation_queue_ready")
        self.assertTrue(payload["beta_capabilities"]["create_order_draft"])
        self.assertEqual(payload["order_preview"]["position_intent"], "rotate")
        self.assertEqual(
            payload["order_preview"]["recommended_actions"],
            ["close_position", "enter_put"],
        )

    def _patched_paths(self, outputs: Path):
        # Patch EVERY path build_payload reads, or leftover real artifacts in
        # ./outputs leak in and break test isolation (the canonical-object
        # resolvers prefer an existing file over the contract fallback).
        return patch.multiple(
            beta,
            OUTDIR=outputs,
            MONITOR_JSON=outputs / "robinhood_fvg_monitor.json",
            CONTRACT_JSON=outputs / "agent_v1_decision.json",
            BRIEF_JSON=outputs / "operator_brief.json",
            DASHBOARD_JSON=outputs / "morning_open_dashboard.json",
            SIGNAL_JSON=outputs / "signal.json",
            WORKFLOW_STATE_JSON=outputs / "workflow_state.json",
            EXECUTION_PLAN_JSON=outputs / "execution_plan.json",
            APPROVAL_DECISION_JSON=outputs / "approval_decision.json",
            BETA_JSON=outputs / "robinhood_beta_execution_source.json",
            OUT_JSON=outputs / "robinhood_beta_execution.json",
            OUT_TXT=outputs / "robinhood_beta_execution.txt",
        )

    def _seed_ready_outputs(
        self,
        root: Path,
        trade_allowed: bool,
        bridge_available: bool = True,
        contract_risk: float = 0.2,
        token_action: str = "enter_put",
    ) -> Path:
        outputs = root / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "robinhood_fvg_monitor.json").write_text(
            json.dumps(
                {
                    "options_context": {"dte_min": 0, "dte_max": 1},
                    "robinhood_mcp_handoff": {
                        "bridge_status": {
                            "available": bridge_available,
                            "status": "ready" if bridge_available else "disabled",
                            "server": "robinhood-trading",
                            "agent": "code-puppy",
                            "fallback_mode": (
                                "mcp_quote_monitoring"
                                if bridge_available
                                else "artifact_only_manual_review"
                            ),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        current_token = {
            "token_id": "failed_breakout:orh:502.00",
            "event_type": "FAILED BREAKOUT",
            "side": "PUTS",
            "status": "confirmed",
            "observation_count": 2,
            "level_name": "ORH",
            "level_price": 502.0,
        }
        closing_token = {
            "token_id": "failed_breakdown:orl:500.00",
            "event_type": "FAILED BREAKDOWN",
            "side": "CALLS",
            "status": "expired",
            "level_name": "ORL",
            "level_price": 500.0,
            "clear_reason": "expired",
        }
        signal_edge_token = {
            "schema": "sharpedge.edge_token_position.v1",
            "ts": "2026-06-25T11:31:00",
            "position_state": "open"
            if token_action in {"enter_put", "hold", "flip_to_put"}
            else "flat",
            "contracts_held": 1
            if token_action in {"enter_put", "hold", "flip_to_put"}
            else 0,
            "suggested_action": token_action,
            "recommended_actions": (
                ["close_position", "enter_put"]
                if token_action == "flip_to_put"
                else [token_action]
            ),
            "current_token": current_token
            if token_action != "close_position"
            else None,
            "closing_token": closing_token
            if token_action in {"close_position", "flip_to_put"}
            else None,
        }
        (outputs / "signal.json").write_text(
            json.dumps({"edge_token_position": signal_edge_token}),
            encoding="utf-8",
        )
        (outputs / "agent_v1_decision.json").write_text(
            json.dumps(
                {
                    "symbol": "SPY",
                    "decision": "operator_confirm_required"
                    if trade_allowed
                    else "hold",
                    "trade_allowed": trade_allowed,
                    "broker_integration_status": "ready"
                    if bridge_available
                    else "disabled",
                    "monitoring_mode": (
                        "approval_queue_shadow_draft"
                        if bridge_available
                        else "artifact_only_manual_review"
                    ),
                    "max_capital_risk_pct": contract_risk,
                    "blocking_reasons": [] if trade_allowed else ["controller_hold"],
                    "risk_flags": [] if trade_allowed else ["monitor_blocks_trade"],
                }
            ),
            encoding="utf-8",
        )
        (outputs / "operator_brief.json").write_text(
            json.dumps(
                {
                    "symbol": "SPY",
                    "headline": "Review bearish gap-fill setup.",
                    "operator_action": "review_trade_plan"
                    if trade_allowed
                    else "stand_down",
                    "focus": {
                        "option_side_watch": "puts_or_put_spreads",
                        "spot": 501.0,
                        "atm_strike": 501.0,
                        "gap_fill_level": "500",
                    },
                }
            ),
            encoding="utf-8",
        )
        (outputs / "morning_open_dashboard.json").write_text(
            json.dumps({"readiness": "review" if trade_allowed else "blocked"}),
            encoding="utf-8",
        )
        return outputs


if __name__ == "__main__":
    unittest.main()
