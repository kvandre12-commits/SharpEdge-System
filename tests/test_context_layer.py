from __future__ import annotations

import importlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_script(name: str, db_path: str):
    """Import (or reload) a scripts.* module with SPY_DB_PATH pointed at a temp DB."""
    with patch.dict(os.environ, {"SPY_DB_PATH": db_path, "SYMBOL": "SPY"}):
        mod = importlib.import_module(f"scripts.{name}")
        return importlib.reload(mod)


class FakeGDELTResponse:
    def __init__(self, payload: dict):
        self._buf = io.BytesIO(json.dumps(payload).encode())

    def read(self, *a):
        return self._buf.read(*a)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


GDELT_PAYLOAD = {
    "articles": [
        {
            "title": "Stock market rally surges to record high on rate cut optimism",
            "url": "https://example.com/a1",
            "domain": "example.com",
            "seendate": "20261005T093000Z",
        },
        {
            "title": "Market plunges on recession fears as selloff deepens",
            "url": "https://example.com/a2",
            "domain": "example.com",
            "seendate": "20261005T101500Z",
        },
        {
            "title": "Analysts discuss quarterly outlook for equities",
            "url": "https://example.com/a3",
            "domain": "example.com",
            "seendate": "20261005T110000Z",
        },
    ]
}


class SeedEventLibraryTests(unittest.TestCase):
    def test_seed_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            mod = load_script("seed_event_library", db)
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            n1 = con.execute("SELECT COUNT(*) FROM event_library").fetchone()[0]
            self.assertEqual(mod.main(), 0)
            n2 = con.execute("SELECT COUNT(*) FROM event_library").fetchone()[0]
            self.assertEqual(n1, n2)
            self.assertEqual(n1, len(mod.EVENTS))
            types = {r[0] for r in con.execute(
                "SELECT DISTINCT event_type FROM event_library")}
            self.assertTrue({"market_stress", "policy_rates", "trade"}.issubset(types))
            con.close()


class IngestCurrentEventsTests(unittest.TestCase):
    def test_tone_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mod = load_script("ingest_current_events", str(Path(tmp) / "t.db"))
            self.assertGreater(mod.score_tone("Market rally surges to record high"), 0)
            self.assertLess(mod.score_tone("Stocks plunge on recession fears"), 0)
            self.assertEqual(mod.score_tone("Analysts discuss quarterly outlook"), 0.0)

    def test_gdelt_date_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mod = load_script("ingest_current_events", str(Path(tmp) / "t.db"))
            self.assertEqual(mod.parse_gdelt_date("20261005T093000Z"), "2026-10-05")
            # fallback: garbage in -> today-shaped date out
            self.assertRegex(mod.parse_gdelt_date(""), r"^\d{4}-\d{2}-\d{2}$")

    def test_ingest_parses_and_stores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            mod = load_script("ingest_current_events", db)
            with patch.object(mod.urllib.request, "urlopen",
                              return_value=FakeGDELTResponse(GDELT_PAYLOAD)):
                self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            rows = con.execute(
                "SELECT title, tone FROM news_daily ORDER BY url").fetchall()
            self.assertEqual(len(rows), 3)
            tones = dict(rows)
            self.assertGreater(
                tones["Stock market rally surges to record high on rate cut optimism"], 0)
            self.assertLess(
                tones["Market plunges on recession fears as selloff deepens"], 0)
            con.close()

    def test_ingest_fail_open_on_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            mod = load_script("ingest_current_events", db)

            def boom(*a, **k):
                raise OSError("network down")

            with patch.object(mod.urllib.request, "urlopen", side_effect=boom):
                self.assertEqual(mod.main(), 0)  # fail-open: exit 0, no raise


class BuildContextOverlaysTests(unittest.TestCase):
    def test_materializes_events_and_news_tone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            seed = load_script("seed_event_library", db)
            seed.main()
            news = load_script("ingest_current_events", db)
            with patch.object(news.urllib.request, "urlopen",
                              return_value=FakeGDELTResponse(GDELT_PAYLOAD)):
                news.main()

            build = load_script("build_context_overlays", db)
            self.assertEqual(build.main(), 0)
            # idempotent: second run changes nothing
            self.assertEqual(build.main(), 0)

            con = sqlite3.connect(db)
            otypes = {r[0] for r in con.execute(
                "SELECT DISTINCT overlay_type FROM overlays_daily")}
            self.assertIn("news_tone", otypes)
            self.assertTrue(any(t.startswith("event_") for t in otypes))
            tone = con.execute(
                "SELECT overlay_strength FROM overlays_daily "
                "WHERE overlay_type='news_tone' AND date='2026-10-05'").fetchone()[0]
            # (+1 rally, -1 selloff, 0 neutral) / 3 ≈ 0
            self.assertAlmostEqual(tone, 0.0, places=2)
            n_events = con.execute(
                "SELECT COUNT(*) FROM overlays_daily WHERE overlay_type LIKE 'event\\_%' ESCAPE '\\'").fetchone()[0]
            self.assertEqual(n_events, len(seed.EVENTS))
            con.close()


class TradeCardContextTests(unittest.TestCase):
    def test_load_context_reads_event_and_tone_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            con.execute(
                """CREATE TABLE overlays_daily (
                     date TEXT, symbol TEXT, overlay_type TEXT,
                     overlay_strength REAL, notes TEXT,
                     PRIMARY KEY (symbol, date, overlay_type))""")
            con.execute(
                "INSERT INTO overlays_daily VALUES "
                "('2026-10-05','SPY','event_trade',0.9,'Tariff pause announced'),"
                "('2026-10-05','SPY','news_tone',0.35,'3 headlines'),"
                "('2026-10-05','SPY','darkpool',0.5,'unrelated overlay')")
            con.commit()
            ctx = card.load_context(con, "SPY", "2026-10-05")
            con.close()
            self.assertAlmostEqual(ctx["news_tone"], 0.35)
            self.assertEqual(len(ctx["events"]), 1)
            self.assertEqual(ctx["events"][0][0], "event_trade")

    def test_load_context_empty_when_no_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            ctx = card.load_context(con, "SPY", "2026-10-05")
            con.close()
            self.assertEqual(ctx, {"events": [], "news_tone": None})


if __name__ == "__main__":
    unittest.main()
