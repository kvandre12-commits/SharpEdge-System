from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import morning_open_dashboard as dashboard


class MorningOpenDashboardTests(unittest.TestCase):
    def test_build_dashboard_flags_blocked_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir(parents=True)
            (outputs / "operator_brief.json").write_text(
                json.dumps(
                    {
                        "symbol": "SPY",
                        "operator_action": "stand_down",
                        "headline": "Stand down.",
                        "focus": {
                            "gap_direction": "DOWN",
                            "gap_fill_level": "737.05",
                            "option_side_watch": "calls",
                            "spot": 742.7,
                            "atm_strike": 743.0,
                            "dealer_state_hint": "DEFENSIVE",
                        },
                        "summary": {"broker_integration_status": "disabled"},
                        "next_steps": ["Step one"],
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "operator_watchlist.json").write_text(
                json.dumps(
                    {
                        "active_count": 0,
                        "items": [{"status": "blocked", "priority": "low", "headline": "Blocked."}],
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "agent_v1_decision.json").write_text(
                json.dumps(
                    {
                        "decision": "hold",
                        "trade_allowed": False,
                        "permissions": {"read_market_data": True, "monitor_quotes": False},
                        "blocking_reasons": ["controller_hold"],
                        "risk_flags": ["low_sample"],
                        "freshness": {"stale_inputs": [{"input": "signal"}]},
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "operator_session_review.json").write_text(
                json.dumps(
                    {
                        "journal_entries_reviewed": 3,
                        "latest_entry": {"operator_action": "stand_down"},
                        "top_blockers": [{"value": "controller_hold", "count": 2}],
                    }
                ),
                encoding="utf-8",
            )

            with patch.multiple(
                dashboard,
                OUTDIR=outputs,
                BRIEF_JSON=outputs / "operator_brief.json",
                WATCHLIST_JSON=outputs / "operator_watchlist.json",
                CONTRACT_JSON=outputs / "agent_v1_decision.json",
                SESSION_REVIEW_JSON=outputs / "operator_session_review.json",
                OUT_JSON=outputs / "morning_open_dashboard.json",
                OUT_TXT=outputs / "morning_open_dashboard.txt",
            ):
                payload = dashboard.build_dashboard()
                text = dashboard.render_text(payload)

        self.assertEqual(payload["readiness"], "blocked")
        self.assertEqual(payload["watchlist_snapshot"]["top_item_status"], "blocked")
        self.assertFalse(payload["checklist"][0]["ok"])
        self.assertIn("SHARPEDGE MORNING OPEN DASHBOARD", text)


if __name__ == "__main__":
    unittest.main()
