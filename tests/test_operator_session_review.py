from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import operator_session_review as review


class OperatorSessionReviewTests(unittest.TestCase):
    def test_build_review_summarizes_recent_journal_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir(parents=True)
            (outputs / "operator_journal_append.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "created_ts": "2026-06-10T20:00:00+00:00",
                                "symbol": "SPY",
                                "operator_action": "stand_down",
                                "watchlist_status": "blocked",
                                "headline": "Stand down.",
                                "blocking_reasons": ["sample_n_below_30"],
                                "risk_flags": ["low_sample"],
                                "broker_integration_status": "disabled",
                            }
                        ),
                        json.dumps(
                            {
                                "created_ts": "2026-06-11T20:00:00+00:00",
                                "symbol": "SPY",
                                "operator_action": "monitor_only",
                                "watchlist_status": "monitor_only",
                                "headline": "Monitor setup.",
                                "blocking_reasons": ["stale_or_missing_inputs"],
                                "risk_flags": ["freshness_gate_failed"],
                                "broker_integration_status": "ready",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (outputs / "operator_watchlist.json").write_text(
                json.dumps({"active_count": 1, "items": [{"status": "monitor_only"}]}),
                encoding="utf-8",
            )

            with patch.multiple(
                review,
                OUTDIR=outputs,
                JOURNAL_JSONL=outputs / "operator_journal_append.jsonl",
                WATCHLIST_JSON=outputs / "operator_watchlist.json",
                OUT_JSON=outputs / "operator_session_review.json",
                OUT_TXT=outputs / "operator_session_review.txt",
                LOOKBACK_ENTRIES=20,
            ):
                payload = review.build_review()
                text = review.render_text(payload)

        self.assertEqual(payload["journal_entries_total"], 2)
        self.assertEqual(payload["journal_entries_reviewed"], 2)
        self.assertEqual(payload["current_watchlist_active_count"], 1)
        self.assertEqual(payload["latest_entry"]["operator_action"], "monitor_only")
        self.assertEqual(payload["top_blockers"][0]["value"], "sample_n_below_30")
        self.assertIn("SHARPEDGE OPERATOR SESSION REVIEW", text)


if __name__ == "__main__":
    unittest.main()
