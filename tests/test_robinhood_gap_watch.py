from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import scripts.build_robinhood_gap_watch as gap_watch


class RobinhoodGapWatchTests(unittest.TestCase):
    def test_classify_unfilled_down_gap_as_watch(self) -> None:
        bars = [
            self._bar("2026-11-17", 100, 102, 99, 100),
            self._bar("2026-11-18", 98, 99, 96, 97),
        ]

        result = gap_watch.classify_gap(
            "NVDA", bars, datetime(2026, 11, 18, tzinfo=timezone.utc).date()
        )

        self.assertEqual(result["decision"], "watch")
        self.assertEqual(result["gap_direction"], "DOWN")
        self.assertEqual(result["gap_fill_level"], 100.0)
        self.assertFalse(result["fill_completed"])

    def test_classify_filled_up_gap_as_no_trade(self) -> None:
        bars = [
            self._bar("2026-11-17", 100, 102, 99, 100),
            self._bar("2026-11-18", 102, 103, 99.5, 101),
        ]

        result = gap_watch.classify_gap(
            "SPY", bars, datetime(2026, 11, 18, tzinfo=timezone.utc).date()
        )

        self.assertEqual(result["decision"], "no_trade")
        self.assertTrue(result["fill_completed"])
        self.assertIn("gap_already_filled", result["reasons"])

    def test_refill_stack_promotes_filled_session_gap_to_watch(self) -> None:
        session_gap = {
            "symbol": "SPY",
            "decision": "no_trade",
            "reasons": ["gap_already_filled"],
        }
        structure = {
            "available": True,
            "active_count": 3,
            "stack_label": "triple_dip_stack",
            "items": [{"target": 100}, {"target": 105}, {"target": 110}],
        }

        result = gap_watch.apply_refill_structure(session_gap, structure)

        self.assertEqual(result["session_gap_decision"], "no_trade")
        self.assertEqual(result["decision"], "watch")
        self.assertIn("triple_dip_stack_active", result["reasons"])

    def test_build_payload_fails_closed_for_missing_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "truth.db"
            self._seed_db(db_path)
            with (
                patch.object(
                    gap_watch,
                    "robinhood_bridge_status",
                    return_value={"available": False, "status": "unbound"},
                ),
                patch.object(
                    gap_watch,
                    "refill_structure",
                    return_value={"available": False, "reason": "fixture"},
                ),
            ):
                payload = gap_watch.build_payload(
                    db_path=db_path,
                    symbols=("SPY", "NVDA"),
                    now=datetime(2026, 11, 18, tzinfo=timezone.utc),
                )

        by_symbol = {item["symbol"]: item for item in payload["watches"]}
        self.assertEqual(payload["schema_version"], "sharpedge.gap-watch.v1")
        self.assertEqual(by_symbol["SPY"]["decision"], "watch")
        self.assertEqual(by_symbol["NVDA"]["decision"], "no_trade")
        self.assertIn("insufficient_daily_bars", by_symbol["NVDA"]["reasons"])
        self.assertIn("place_order", payload["robinhood_handoff"]["blocked_actions"])
        self.assertTrue(
            payload["robinhood_handoff"]["operator_confirmation_required_for_orders"]
        )

    @staticmethod
    def _bar(
        session_date: str, open_: float, high: float, low: float, close: float
    ) -> dict[str, object]:
        return {
            "date": session_date,
            "symbol": "SPY",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000,
            "source": "fixture",
            "ingest_ts": f"{session_date}T22:00:00+00:00",
        }

    def _seed_db(self, path: Path) -> None:
        con = sqlite3.connect(path)
        try:
            con.execute(
                """
                CREATE TABLE bars_daily (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT,
                    ingest_ts TEXT,
                    PRIMARY KEY (symbol, date)
                )
                """
            )
            con.executemany(
                "INSERT INTO bars_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("2026-11-17", "SPY", 100, 102, 99, 100, 1, "fixture", "2026-11-17T22:00:00+00:00"),
                    ("2026-11-18", "SPY", 98, 99, 96, 97, 1, "fixture", "2026-11-18T22:00:00+00:00"),
                ],
            )
            con.commit()
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
