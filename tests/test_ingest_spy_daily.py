from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.ingest_spy_daily as daily


class IngestSpyDailyTests(unittest.TestCase):
    def test_should_skip_network_when_cache_is_fresh(self) -> None:
        state = {"rows": 10, "latest_ingest_ts": "2026-06-10T10:00:00+00:00"}

        with (
            patch.object(daily, "CACHE_TTL_HOURS", 12),
            patch.object(daily, "FORCE_REFRESH", False),
            patch(
                "scripts.utils.pipeline_state.utc_now",
                return_value=dt.datetime(2026, 6, 10, 12, 0, 0),
            ),
        ):
            self.assertTrue(daily.should_skip_network(state))

    def test_fetch_period_uses_incremental_when_state_exists(self) -> None:
        state = {"rows": 10, "latest_ingest_ts": "2026-06-01T10:00:00+00:00"}

        with (
            patch.object(daily, "FORCE_REFRESH", False),
            patch.object(daily, "INCREMENTAL_PERIOD", "10d"),
        ):
            self.assertEqual(daily.fetch_period_for_state(state), "10d")

    def test_fetch_daily_wraps_shared_yahoo_rows(self) -> None:
        shared_rows = [
            {
                "date": "2026-06-24",
                "open": 600.0,
                "high": 605.0,
                "low": 599.5,
                "close": 604.5,
                "volume": 123456.0,
            }
        ]
        shared_source = {"provider": "yahoo", "bar_count": 1}

        with patch.object(
            daily,
            "fetch_yahoo_daily_bars",
            return_value=(shared_rows, shared_source),
        ):
            rows, source = daily.fetch_daily("10d")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], daily.SYMBOL)
        self.assertEqual(rows[0]["source"], daily.SOURCE)
        self.assertEqual(rows[0]["close"], 604.5)
        self.assertEqual(source, shared_source)
        self.assertTrue(rows[0]["ingest_ts"])

    def test_write_truth_csv_exports_rows_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "truth.db"
            output_root = Path(tmp) / "outputs"
            output_root.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(db_path)
            try:
                daily.ensure_truth_table(con)
                con.execute(
                    "INSERT INTO bars_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-06-24",
                        daily.SYMBOL,
                        600.0,
                        605.0,
                        599.5,
                        604.5,
                        123456.0,
                        daily.SOURCE,
                        "2026-06-25T00:00:00+00:00",
                    ),
                )
                con.commit()

                with patch(
                    "scripts.ingest_spy_daily.Path",
                    side_effect=lambda value: output_root / Path(value).name,
                ):
                    count = daily.write_truth_csv(con)
            finally:
                con.close()

            self.assertEqual(count, 1)
            csv_text = (
                output_root / f"{daily.SYMBOL.lower()}_truth_daily.csv"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "date,symbol,open,high,low,close,volume,source,ingest_ts", csv_text
            )
            self.assertIn("2026-06-24", csv_text)
            self.assertIn(daily.SOURCE, csv_text)


if __name__ == "__main__":
    unittest.main()
