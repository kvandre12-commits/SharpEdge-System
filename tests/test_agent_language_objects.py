from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import agent_language_objects as lang


class AgentLanguageObjectsTests(unittest.TestCase):
    def test_build_objects_normalizes_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self._seed_outputs(Path(tmp))
            with self._patched_paths(outputs):
                objects = lang.build_objects()
                workflow = objects["workflow_state"]
                plan = objects["execution_plan"]
                approval = objects["approval_decision"]
                journal = objects["journal"]

        run_ids = {
            workflow["run_id"],
            plan["run_id"],
            approval["run_id"],
            journal["run_id"],
        }
        self.assertEqual(len(run_ids), 1)
        self.assertEqual(workflow["state"]["lifecycle_stage"], "approval_pending_operator")
        self.assertEqual(plan["intended_action"], "review_trade_plan")
        self.assertIn("confirm_order", plan["prerequisites"]["required_approvals"])
        self.assertTrue(approval["trade_allowed"])
        self.assertEqual(approval["authority"], "authoritative_permission_gate")
        self.assertEqual(journal["latest_entry"]["entry_id"], "entry-123")
        self.assertIn("Observed journal winners cluster", journal["lessons"][0])

    def test_build_objects_handles_missing_inputs_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            outputs.mkdir(parents=True)
            with self._patched_paths(outputs):
                objects = lang.build_objects()

        self.assertEqual(objects["workflow_state"]["state"]["lifecycle_stage"], "blocked")
        self.assertEqual(objects["approval_decision"]["decision"], "missing")
        self.assertEqual(objects["journal"]["entries_total"], 0)
        self.assertFalse(objects["workflow_state"]["historical_context"]["available"])

    def _patched_paths(self, outputs: Path):
        return patch.multiple(
            lang,
            OUTDIR=outputs,
            CONTROLLER_JSON=outputs / "agent_controller_decision.json",
            MONITOR_JSON=outputs / "robinhood_fvg_monitor.json",
            CONTRACT_JSON=outputs / "agent_v1_decision.json",
            BRIEF_JSON=outputs / "operator_brief.json",
            DASHBOARD_JSON=outputs / "morning_open_dashboard.json",
            BETA_JSON=outputs / "robinhood_beta_execution.json",
            SESSION_REVIEW_JSON=outputs / "operator_session_review.json",
            TRADE_HINTS_JSON=outputs / "trade_journal_hints.json",
            JOURNAL_JSONL=outputs / "operator_journal_append.jsonl",
            WORKFLOW_STATE_JSON=outputs / "workflow_state.json",
            WORKFLOW_STATE_TXT=outputs / "workflow_state.txt",
            EXECUTION_PLAN_JSON=outputs / "execution_plan.json",
            EXECUTION_PLAN_TXT=outputs / "execution_plan.txt",
            APPROVAL_DECISION_JSON=outputs / "approval_decision.json",
            APPROVAL_DECISION_TXT=outputs / "approval_decision.txt",
            JOURNAL_JSON=outputs / "journal.json",
            JOURNAL_TXT=outputs / "journal.txt",
        )

    def _seed_outputs(self, root: Path) -> Path:
        outputs = root / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "agent_controller_decision.json").write_text(
            json.dumps(
                {
                    "decision": "post",
                    "confidence": 0.84,
                    "symbol": "SPY",
                    "ts_utc": "2026-06-11T22:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        (outputs / "robinhood_fvg_monitor.json").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T21:59:00+00:00",
                    "decision": "watch",
                }
            ),
            encoding="utf-8",
        )
        (outputs / "agent_v1_decision.json").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T22:01:00+00:00",
                    "symbol": "SPY",
                    "decision": "operator_confirm_required",
                    "trade_allowed": True,
                    "broker_order_allowed": False,
                    "required_human_action": "confirm_order",
                    "broker_integration_status": "ready",
                    "monitoring_mode": "approval_queue_shadow_draft",
                    "risk_state": "STANDARD",
                    "max_capital_risk_pct": 0.25,
                    "blocking_reasons": [],
                    "risk_flags": [],
                    "freshness": {"stale_inputs": []},
                    "confidence_evidence_quality": 0.84,
                    "confidence_trade_edge": 0.72,
                    "source_decisions": {"controller_decision": "post", "monitor_decision": "watch"},
                    "permissions": {"place_order": False, "read_market_data": True},
                }
            ),
            encoding="utf-8",
        )
        (outputs / "operator_brief.json").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T22:02:00+00:00",
                    "symbol": "SPY",
                    "operator_action": "review_trade_plan",
                    "headline": "Review trade plan.",
                    "focus": {
                        "gap_direction": "DOWN",
                        "gap_fill_level": "500",
                        "gap_session_date": "2026-06-11",
                        "fill_bias": "bullish_gap_fill_watch",
                        "option_side_watch": "calls_or_call_spreads",
                        "spot": 501.0,
                        "atm_strike": 501.0,
                        "dealer_state_hint": "DEFENSIVE",
                    },
                    "historical_hints": {
                        "available": True,
                        "top_pattern_summary": "Observed journal winners cluster around VWAP_FADE + REJECT.",
                        "low_sample": True,
                    },
                    "next_steps": ["Review thesis.", "Confirm risk."],
                }
            ),
            encoding="utf-8",
        )
        (outputs / "morning_open_dashboard.json").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T22:03:00+00:00",
                    "readiness": "review",
                }
            ),
            encoding="utf-8",
        )
        (outputs / "robinhood_beta_execution.json").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T22:04:00+00:00",
                    "beta_stage": "approval_queue_ready",
                    "required_approvals": ["operator_trade_confirmation", "risk_budget_review"],
                    "order_preview": {"strategy_family": "call_debit_spread"},
                    "beta_capabilities": {"create_order_draft": True},
                    "robinhood_beta_handoff": {"fallback_mode": "approval_queue_shadow_draft"},
                }
            ),
            encoding="utf-8",
        )
        (outputs / "operator_session_review.json").write_text(
            json.dumps(
                {
                    "journal_entries_reviewed": 1,
                    "top_blockers": [],
                    "top_risk_flags": [],
                }
            ),
            encoding="utf-8",
        )
        (outputs / "trade_journal_hints.json").write_text(
            json.dumps(
                {
                    "top_patterns": [
                        {
                            "condition": {"setup": "VWAP_FADE", "vwap_behavior": "REJECT"},
                            "confidence_label": "LOW_SAMPLE",
                        }
                    ],
                    "actionable_hints": [
                        {
                            "summary": "Observed journal winners cluster around VWAP_FADE + REJECT.",
                        }
                    ],
                    "metric_collection_priorities": ["MFE", "MAE"],
                    "research_hypotheses": ["optimal strike distance"],
                    "usage_constraints": ["never override approval_decision"],
                }
            ),
            encoding="utf-8",
        )
        (outputs / "operator_journal_append.jsonl").write_text(
            json.dumps(
                {
                    "created_ts": "2026-06-11T22:05:00+00:00",
                    "entry_id": "entry-123",
                    "operator_action": "review_trade_plan",
                    "watchlist_status": "ready_for_review",
                    "headline": "Review trade plan.",
                    "blocking_reasons": [],
                    "risk_flags": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return outputs


if __name__ == "__main__":
    unittest.main()
