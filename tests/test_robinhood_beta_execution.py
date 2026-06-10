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
        self.assertEqual(payload["order_preview"]["strategy_family"], "put_debit_spread")
        self.assertTrue(payload["approval_required"])

    def test_build_payload_falls_back_to_artifact_only_when_bridge_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(Path(tmp), trade_allowed=False, bridge_available=False)
            with self._patched_paths(outputs):
                payload = beta.build_payload()
                text = beta.render_text(payload)

        self.assertEqual(payload["beta_stage"], "artifact_only")
        self.assertFalse(payload["beta_capabilities"]["create_order_draft"])
        self.assertIn("artifact_only_order_preview", payload["robinhood_beta_handoff"]["permitted_actions"])
        self.assertIn("ROBINHOOD BETA EXECUTION HANDOFF", text)

    def test_order_preview_caps_risk_to_beta_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_ready_outputs(Path(tmp), trade_allowed=True, contract_risk=0.40)
            with self._patched_paths(outputs):
                with patch.object(beta, "BETA_MAX_RISK_CAP_PCT", 0.25):
                    payload = beta.build_payload()

        self.assertEqual(payload["order_preview"]["risk_limits"]["source_contract_risk_pct"], 0.4)
        self.assertEqual(payload["order_preview"]["risk_limits"]["max_capital_risk_pct"], 0.25)

    def _patched_paths(self, outputs: Path):
        return patch.multiple(
            beta,
            OUTDIR=outputs,
            MONITOR_JSON=outputs / "robinhood_fvg_monitor.json",
            CONTRACT_JSON=outputs / "agent_v1_decision.json",
            BRIEF_JSON=outputs / "operator_brief.json",
            DASHBOARD_JSON=outputs / "morning_open_dashboard.json",
            OUT_JSON=outputs / "robinhood_beta_execution.json",
            OUT_TXT=outputs / "robinhood_beta_execution.txt",
        )

    def _seed_ready_outputs(
        self,
        root: Path,
        trade_allowed: bool,
        bridge_available: bool = True,
        contract_risk: float = 0.2,
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
        (outputs / "agent_v1_decision.json").write_text(
            json.dumps(
                {
                    "symbol": "SPY",
                    "decision": "operator_confirm_required" if trade_allowed else "hold",
                    "trade_allowed": trade_allowed,
                    "broker_integration_status": "ready" if bridge_available else "disabled",
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
                    "operator_action": "review_trade_plan" if trade_allowed else "stand_down",
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
