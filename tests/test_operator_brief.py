from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import operator_brief as brief


class OperatorBriefTests(unittest.TestCase):
    def test_build_brief_stand_down_with_disabled_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            health = outputs / "health"
            health.mkdir(parents=True)

            (outputs / "agent_controller_decision.json").write_text(
                json.dumps({"decision": "hold", "confidence": 1.0}),
                encoding="utf-8",
            )
            (outputs / "robinhood_fvg_monitor.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:00:00+00:00",
                        "decision": "no_trade",
                        "latest_gap_event": {
                            "session_date": "2026-06-10",
                            "gap_direction": "DOWN",
                            "gap_fill_level": "737.05",
                        },
                        "directional_hypothesis": {
                            "fill_bias": "bullish_gap_fill_watch",
                            "option_side_watch": "calls_or_call_spreads",
                        },
                        "options_context": {
                            "spot": 742.72,
                            "atm_strike": 743.0,
                            "dealer_state_hint": "DEFENSIVE",
                        },
                        "risk_context": {"sample_n": 1, "deployment_state": "PROBE"},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "agent_v1_decision.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:01:00+00:00",
                        "decision": "hold",
                        "risk_state": "PROBE",
                        "broker_integration_status": "disabled",
                        "monitoring_mode": "artifact_only_manual_review",
                        "blocking_reasons": ["monitor_no_trade", "sample_n_below_30"],
                        "risk_flags": ["broker_integration_unavailable", "monitor_blocks_trade"],
                        "freshness": {"stale_inputs": []},
                        "confidence_evidence_quality": 1.0,
                        "confidence_trade_edge": 0.0,
                        "max_capital_risk_pct": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            (health / "warnings.log").write_text("warn a\n", encoding="utf-8")

            with self._patched_paths(outputs):
                payload = brief.build_brief()
                watchlist = brief.build_watchlist()

        self.assertEqual(payload["operator_action"], "stand_down")
        self.assertEqual(payload["summary"]["broker_integration_status"], "disabled")
        self.assertIn("Broker integration is not live", payload["next_steps"][1])
        self.assertEqual(payload["focus"]["option_side_watch"], "calls_or_call_spreads")
        self.assertEqual(watchlist["active_count"], 0)
        self.assertEqual(watchlist["items"][0]["status"], "blocked")

    def test_build_brief_review_trade_plan_when_contract_requires_operator_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir(parents=True)

            (outputs / "agent_controller_decision.json").write_text(
                json.dumps({"decision": "post", "confidence": 0.8, "ts_utc": "2026-06-10T20:02:00+00:00"}),
                encoding="utf-8",
            )
            (outputs / "robinhood_fvg_monitor.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:00:00+00:00",
                        "decision": "watch",
                        "latest_gap_event": {
                            "session_date": "2026-06-10",
                            "gap_direction": "UP",
                            "gap_fill_level": "500",
                        },
                        "directional_hypothesis": {
                            "fill_bias": "bearish_gap_fill_watch",
                            "option_side_watch": "puts_or_put_spreads",
                        },
                        "options_context": {"spot": 501.0, "atm_strike": 501.0},
                        "risk_context": {"sample_n": 50, "deployment_state": "STANDARD"},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "agent_v1_decision.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:01:00+00:00",
                        "decision": "operator_confirm_required",
                        "risk_state": "STANDARD",
                        "broker_integration_status": "ready",
                        "monitoring_mode": "mcp_quote_monitoring",
                        "blocking_reasons": [],
                        "risk_flags": [],
                        "freshness": {"stale_inputs": []},
                        "confidence_evidence_quality": 0.8,
                        "confidence_trade_edge": 0.72,
                        "max_capital_risk_pct": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(outputs):
                payload = brief.build_brief()
                watchlist = brief.build_watchlist()
                text = brief.render_text(payload)

        self.assertEqual(payload["operator_action"], "review_trade_plan")
        self.assertIn("manual confirmation still required", payload["headline"])
        self.assertIn("Review the puts_or_put_spreads thesis", payload["next_steps"][0])
        self.assertIn("SHARPEDGE OPERATOR BRIEF", text)
        self.assertEqual(watchlist["active_count"], 1)
        self.assertEqual(watchlist["items"][0]["status"], "ready_for_review")

    def test_append_journal_entry_is_idempotent_for_same_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir(parents=True)

            (outputs / "agent_controller_decision.json").write_text(
                json.dumps({"decision": "post", "confidence": 0.8, "ts_utc": "2026-06-10T20:02:00+00:00"}),
                encoding="utf-8",
            )
            (outputs / "robinhood_fvg_monitor.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:00:00+00:00",
                        "decision": "watch",
                        "latest_gap_event": {
                            "session_date": "2026-06-10",
                            "gap_direction": "UP",
                            "gap_fill_level": "500",
                        },
                        "directional_hypothesis": {
                            "fill_bias": "bearish_gap_fill_watch",
                            "option_side_watch": "puts_or_put_spreads",
                        },
                        "options_context": {"spot": 501.0, "atm_strike": 501.0},
                        "risk_context": {"sample_n": 50, "deployment_state": "STANDARD"},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "agent_v1_decision.json").write_text(
                json.dumps(
                    {
                        "created_ts": "2026-06-10T20:01:00+00:00",
                        "decision": "operator_confirm_required",
                        "risk_state": "STANDARD",
                        "broker_integration_status": "ready",
                        "monitoring_mode": "mcp_quote_monitoring",
                        "blocking_reasons": [],
                        "risk_flags": [],
                        "freshness": {"stale_inputs": []},
                        "confidence_evidence_quality": 0.8,
                        "confidence_trade_edge": 0.72,
                        "max_capital_risk_pct": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(outputs):
                entry = brief.build_journal_entry()
                first = brief.append_journal_entry(entry)
                second = brief.append_journal_entry(entry)
                lines = (outputs / "operator_journal_append.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["entry_id"], entry["entry_id"])

    def _patched_paths(self, outputs: Path):
        return patch.multiple(
            brief,
            OUTDIR=outputs,
            CONTROLLER_JSON=outputs / "agent_controller_decision.json",
            MONITOR_JSON=outputs / "robinhood_fvg_monitor.json",
            AGENT_V1_JSON=outputs / "agent_v1_decision.json",
            HEALTH_WARNINGS=outputs / "health" / "warnings.log",
            OUT_JSON=outputs / "operator_brief.json",
            OUT_TXT=outputs / "operator_brief.txt",
            OUT_WATCHLIST_JSON=outputs / "operator_watchlist.json",
            OUT_JOURNAL_JSONL=outputs / "operator_journal_append.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
