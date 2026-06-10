from __future__ import annotations

import datetime as dt
import unittest

import pandas as pd
from unittest.mock import patch

import scripts.ingest_finra_darkpool_overlay as finra


class FinraDarkpoolOverlayTests(unittest.TestCase):
    def test_cache_is_fresh_inside_ttl(self) -> None:
        now = dt.datetime(2026, 6, 10, 12, 0, 0)
        latest = "2026-06-10T08:00:00"

        with patch.object(finra, "CACHE_TTL_HOURS", 6):
            self.assertTrue(finra.cache_is_fresh(latest, now=now))

    def test_cache_is_stale_outside_ttl(self) -> None:
        now = dt.datetime(2026, 6, 10, 12, 0, 0)
        latest = "2026-06-09T00:00:00"

        with patch.object(finra, "CACHE_TTL_HOURS", 6):
            self.assertFalse(finra.cache_is_fresh(latest, now=now))

    def test_weeks_to_fetch_skips_when_cache_fresh(self) -> None:
        state = {
            "latest_week_start": "2026-05-18",
            "latest_ingest_ts": "2026-06-10T08:00:00",
            "rows": 125,
        }

        with patch.object(finra, "CACHE_TTL_HOURS", 144):
            weeks = finra.weeks_to_fetch(state, today=dt.date(2026, 6, 10), force=False)

        self.assertEqual(weeks, [])

    def test_weeks_to_fetch_uses_recent_window_when_cache_stale(self) -> None:
        state = {
            "latest_week_start": "2026-05-18",
            "latest_ingest_ts": "2026-06-01T00:00:00",
            "rows": 125,
        }

        with patch.object(finra, "CACHE_TTL_HOURS", 24), patch.object(finra, "REFRESH_LOOKBACK_WEEKS", 2):
            weeks = finra.weeks_to_fetch(state, today=dt.date(2026, 6, 10), force=False)

        self.assertEqual(weeks[0], dt.date(2026, 5, 4))
        self.assertEqual(weeks[-1], dt.date(2026, 6, 8))
        self.assertLess(len(weeks), 10)

    def test_export_weekly_frame_preserves_legacy_week_start_column(self) -> None:
        frame = pd.DataFrame({"week_start": ["2026-05-18"], "symbol": ["SPY"]})

        exported = finra.export_weekly_frame(frame)

        self.assertEqual(list(exported.columns), ["weekStartDate", "week_start", "symbol"])
        self.assertEqual(exported.loc[0, "weekStartDate"], "2026-05-18")

    def test_weeks_to_fetch_force_refresh_starts_from_configured_start(self) -> None:
        state = {
            "latest_week_start": "2026-05-18",
            "latest_ingest_ts": "2026-06-10T08:00:00",
            "rows": 125,
        }

        with patch.object(finra, "START", "2026-05-01"):
            weeks = finra.weeks_to_fetch(state, today=dt.date(2026, 5, 20), force=True)

        self.assertEqual(weeks[0], dt.date(2026, 4, 27))
        self.assertEqual(weeks[-1], dt.date(2026, 5, 18))


if __name__ == "__main__":
    unittest.main()
