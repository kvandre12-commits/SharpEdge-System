from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agents import trade_journal_hints as hints


class TradeJournalHintsTests(unittest.TestCase):
    def test_build_payload_extracts_patterns_and_note_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "trades.db"
            notes_dir = root / "notes"
            notes_dir.mkdir()
            self._seed_db(db_path)
            (notes_dir / "2026-05-29_put_reclaim_capture.md").write_text(
                "\n".join(
                    [
                        "# Sample Note",
                        "Future Metrics To Track:",
                        "- MFE",
                        "- MAE",
                        "- Hold duration",
                        "",
                        "## Recursive Learning Hypothesis",
                        "- optimal strike distance",
                        "- gamma responsiveness",
                        "",
                        "Category: Execution Attribution",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.multiple(
                hints,
                DB_PATH=db_path,
                NOTES_DIR=notes_dir,
                MIN_PATTERN_SAMPLE_N=5,
                TOP_PATTERN_LIMIT=3,
            ):
                payload = hints.build_payload()
                text = hints.render_text(payload)

        self.assertEqual(payload["sample_state"]["total_trades"], 3)
        self.assertTrue(payload["sample_state"]["low_sample"])
        self.assertEqual(payload["top_patterns"][0]["condition"]["setup"], "VWAP_FADE")
        self.assertEqual(payload["top_patterns"][0]["condition"]["vwap_behavior"], "REJECT")
        self.assertEqual(payload["top_patterns"][0]["preferred_exit_reason"], "TARGET")
        self.assertEqual(payload["actionable_hints"][0]["hint_type"], "pattern_repeat_watch")
        self.assertIn("Hold duration", payload["metric_collection_priorities"])
        self.assertIn("optimal strike distance", payload["research_hypotheses"])
        self.assertIn("SHARPEDGE TRADE JOURNAL HINTS", text)

    def test_build_payload_handles_missing_db_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / "notes"
            notes_dir.mkdir()

            with patch.multiple(
                hints,
                DB_PATH=root / "missing.db",
                NOTES_DIR=notes_dir,
                MIN_PATTERN_SAMPLE_N=5,
                TOP_PATTERN_LIMIT=3,
            ):
                payload = hints.build_payload()

        self.assertEqual(payload["sample_state"]["total_trades"], 0)
        self.assertEqual(payload["top_patterns"], [])
        self.assertFalse(payload["symbols"])
        self.assertEqual(payload["field_coverage"], {})

    def _seed_db(self, db_path: Path) -> None:
        con = sqlite3.connect(db_path)
        try:
            con.executescript(
                """
                CREATE TABLE trades (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  trade_date TEXT NOT NULL,
                  symbol TEXT NOT NULL,
                  asset TEXT NOT NULL DEFAULT 'OPTION',
                  direction TEXT NOT NULL,
                  option_type TEXT,
                  strike REAL,
                  expiry TEXT,
                  qty INTEGER NOT NULL DEFAULT 1,
                  entry_time TEXT,
                  exit_time TEXT,
                  entry_price REAL,
                  exit_price REAL,
                  fees REAL DEFAULT 0.0,
                  pnl REAL,
                  pnl_r REAL,
                  setup TEXT,
                  vwap_behavior TEXT,
                  invalidation_rule TEXT DEFAULT 'RECLAIM_AND_HOLD_ABOVE_VWAP',
                  time_stop_min INTEGER DEFAULT 15,
                  risk_dollars REAL,
                  exit_reason TEXT,
                  notes TEXT
                );
                """
            )
            con.executemany(
                """
                INSERT INTO trades(
                  trade_date, symbol, asset, direction, option_type, strike, expiry, qty,
                  entry_time, exit_time, entry_price, exit_price, fees, pnl, pnl_r,
                  setup, vwap_behavior, time_stop_min, risk_dollars, exit_reason, notes
                ) VALUES (?, ?, 'OPTION', 'LONG', 'PUT', ?, ?, 1, ?, ?, ?, ?, 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "2026-02-17",
                        "SPY",
                        679.0,
                        "2026-02-17",
                        "11:57",
                        "12:46",
                        0.57,
                        1.11,
                        54.0,
                        0.947,
                        "VWAP_FADE",
                        "REJECT",
                        60,
                        57.0,
                        "TARGET",
                        "good",
                    ),
                    (
                        "2026-02-17",
                        "SPY",
                        675.0,
                        "2026-02-18",
                        "12:30",
                        "12:46",
                        1.13,
                        1.65,
                        52.0,
                        0.765,
                        "VWAP_FADE",
                        "REJECT",
                        15,
                        68.0,
                        "TARGET",
                        "good",
                    ),
                    (
                        "2026-02-17",
                        "SPY",
                        673.0,
                        "2026-02-17",
                        "10:23",
                        "10:34",
                        0.68,
                        1.09,
                        41.0,
                        0.603,
                        "VWAP_FADE",
                        "REJECT",
                        15,
                        68.0,
                        "TARGET",
                        "good",
                    ),
                ],
            )
            con.commit()
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
