from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.aggregate_options_positioning_metrics as options_metrics
import scripts.ingest_fred_overlays as fred
import scripts.ingest_spy_daily as daily


class Layer1CacheControlTests(unittest.TestCase):
    def test_daily_bars_skip_network_when_ingest_cache_is_fresh(self) -> None:
        state = {"rows": 10, "latest_ingest_ts": "2026-06-10T10:00:00+00:00"}

        with patch.object(daily, "CACHE_TTL_HOURS", 12), patch.object(
            daily, "FORCE_REFRESH", False
        ), patch("scripts.utils.pipeline_state.utc_now", return_value=dt.datetime(2026, 6, 10, 12, 0, 0)):
            self.assertTrue(daily.should_skip_network(state))

    def test_daily_bars_uses_incremental_period_when_state_exists(self) -> None:
        state = {"rows": 10, "latest_ingest_ts": "2026-06-01T10:00:00+00:00"}

        with patch.object(daily, "FORCE_REFRESH", False), patch.object(daily, "INCREMENTAL_PERIOD", "10d"):
            self.assertEqual(daily.fetch_period_for_state(state), "10d")

    def test_fred_macro_is_fresh_when_latest_overlay_within_lag(self) -> None:
        state = {"rows": 100, "latest_date": "2026-06-09"}

        with patch.object(fred, "MAX_LAG_DAYS", 5), patch.object(fred, "FORCE_REFRESH", False), patch.object(
            fred, "utc_now", return_value=dt.datetime(2026, 6, 10, 12, 0, 0)
        ):
            self.assertTrue(fred.macro_is_fresh(state))

    def test_fred_macro_is_stale_when_latest_overlay_too_old(self) -> None:
        state = {"rows": 100, "latest_date": "2026-05-01"}

        with patch.object(fred, "MAX_LAG_DAYS", 5), patch.object(fred, "FORCE_REFRESH", False), patch.object(
            fred, "utc_now", return_value=dt.datetime(2026, 6, 10, 12, 0, 0)
        ):
            self.assertFalse(fred.macro_is_fresh(state))

    def test_options_snapshots_to_process_skips_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            con = sqlite3.connect(db_path)
            try:
                con.executescript(
                    """
                    CREATE TABLE options_chain_snapshots (
                        snapshot_ts TEXT,
                        underlying TEXT,
                        dte INTEGER
                    );
                    CREATE TABLE options_positioning_metrics (
                        snapshot_ts TEXT,
                        underlying TEXT,
                        dte_min INTEGER,
                        dte_max INTEGER
                    );
                    """
                )
                con.executemany(
                    "INSERT INTO options_chain_snapshots VALUES (?, ?, ?)",
                    [("a", "SPY", 0), ("b", "SPY", 1), ("c", "SPY", 3)],
                )
                con.execute(
                    "INSERT INTO options_positioning_metrics VALUES (?, ?, ?, ?)",
                    ("a", "SPY", 0, 1),
                )
                con.commit()

                with patch.object(options_metrics, "UNDERLYING", "SPY"), patch.object(
                    options_metrics, "DTE_MIN", 0
                ), patch.object(options_metrics, "DTE_MAX", 1), patch.object(
                    options_metrics, "ONLY_SNAPSHOT_TS", ""
                ), patch.object(options_metrics, "FORCE_REBUILD", False):
                    self.assertEqual(options_metrics.snapshots_to_process(con), ["b"])
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
