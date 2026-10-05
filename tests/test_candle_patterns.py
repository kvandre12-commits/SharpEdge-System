from __future__ import annotations

import importlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_script(name: str, db_path: str, extra_env: dict | None = None):
    env = {"SPY_DB_PATH": db_path, "SYMBOL": "SPY"}
    env.update(extra_env or {})
    with patch.dict(os.environ, env):
        mod = importlib.import_module(f"scripts.{name}")
        return importlib.reload(mod)


BARS_SCHEMA = """
CREATE TABLE bars_daily (
  date TEXT NOT NULL, symbol TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  PRIMARY KEY (date, symbol)
);
"""


def seed_bars(con, rows):
    con.execute(BARS_SCHEMA)
    con.executemany(
        "INSERT INTO bars_daily (date, symbol, open, high, low, close) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()


def bar(date, o, h, l, c):
    return (date, "SPY", o, h, l, c)


def downtrend(n=10, start=100.0):
    return [(f"2026-01-{i + 1:02d}", start - i) for i in range(n)]


class DetectPatternsTests(unittest.TestCase):
    def _run(self, tmp, bars):
        db = str(Path(tmp) / "t.db")
        con = sqlite3.connect(db)
        seed_bars(con, bars)
        con.close()
        mod = load_script("detect_candle_patterns", db)
        self.assertEqual(mod.main(), 0)
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT date, pattern FROM candle_patterns").fetchall()
        con.close()
        return {d: p for d, p in rows}, mod

    def test_hammer_in_downtrend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars = [bar(d, c, c + 0.5, c - 0.5, c) for d, c in downtrend()]
            bars.append(bar("2026-01-11", 89.5, 90.8, 84.0, 90.5))
            found, _ = self._run(tmp, bars)
            self.assertIn("hammer", found.get("2026-01-11", ""))
            self.assertNotIn("hanging_man", str(found.get("2026-01-11")))

    def test_hanging_man_shape_in_uptrend_is_not_hammer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars = [bar(f"2026-01-{i + 1:02d}", 100 + i, 100 + i + 0.5,
                        100 + i - 0.5, 100 + i) for i in range(10)]
            bars.append(bar("2026-01-11", 109.0, 109.5, 103.0, 109.2))
            found, _ = self._run(tmp, bars)
            pats = str(found.get("2026-01-11"))
            self.assertIn("hanging_man", pats)
            self.assertNotIn("hammer", pats.replace("hanging_man", ""))

    def test_doji(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars = [bar(f"2026-01-{i + 1:02d}", 100, 101, 99, 100 + (i % 2))
                    for i in range(10)]
            bars.append(bar("2026-01-11", 100.0, 100.6, 99.4, 100.05))
            found, _ = self._run(tmp, bars)
            self.assertIn("doji", str(found.get("2026-01-11")))

    def test_engulfing_bull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars = [bar(f"2026-01-{i + 1:02d}", 100, 101, 99, 100)
                    for i in range(10)]
            bars.append(bar("2026-01-11", 100.0, 100.5, 95.0, 95.5))
            bars.append(bar("2026-01-12", 95.0, 101.0, 94.5, 100.5))
            found, _ = self._run(tmp, bars)
            self.assertIn("engulfing_bull", str(found.get("2026-01-12")))

    def test_morning_star(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars = [bar(f"2026-01-{i + 1:02d}", 100, 101, 99, 100)
                    for i in range(10)]
            bars.append(bar("2026-01-11", 95.0, 95.5, 89.0, 90.0))   # long red
            bars.append(bar("2026-01-12", 89.5, 90.5, 88.5, 90.0))   # small star
            bars.append(bar("2026-01-13", 90.5, 96.0, 90.0, 95.5))   # long green
            found, _ = self._run(tmp, bars)
            self.assertIn("morning_star", str(found.get("2026-01-13")))

    def test_detect_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # flat, directional candles: no shadows worth naming, no patterns
            bars = [bar(f"2026-01-{i + 1:02d}", 100.0, 101.5, 99.8, 101.2)
                    for i in range(10)]
            bars.append(bar("2026-01-11", 100.0, 100.6, 99.4, 100.05))
            db = str(Path(tmp) / "t.db")
            con = sqlite3.connect(db)
            seed_bars(con, bars)
            con.close()
            mod = load_script("detect_candle_patterns", db)
            mod.main()
            mod.main()
            con = sqlite3.connect(db)
            n = con.execute("SELECT COUNT(*) FROM candle_patterns").fetchone()[0]
            con.close()
            # 1 doji row; re-run must not duplicate
            self.assertEqual(n, 1)

    def test_no_bars_table_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            sqlite3.connect(db).close()
            mod = load_script("detect_candle_patterns", db)
            self.assertEqual(mod.main(), 0)


class PatternExpectancyTests(unittest.TestCase):
    def test_forward_return_math(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            con = sqlite3.connect(db)
            seed_bars(con, [bar(f"2026-01-{i + 1:02d}", 100, 101, 99, 100 + i)
                            for i in range(8)])
            con.execute(
                """CREATE TABLE candle_patterns (
                     date TEXT, symbol TEXT, pattern TEXT,
                     direction_hint TEXT, strength REAL,
                     PRIMARY KEY (date, symbol, pattern))""")
            con.execute(
                "INSERT INTO candle_patterns VALUES "
                "('2026-01-03','SPY','hammer','bullish',0.8)")
            con.commit()
            con.close()
            mod = load_script("build_pattern_expectancy", db)
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            row = con.execute(
                "SELECT n, fwd_1d_mean, fwd_3d_mean, fwd_5d_mean, hit_rate_5d "
                "FROM pattern_expectancy WHERE pattern='hammer'").fetchone()
            con.close()
            self.assertEqual(row[0], 1)
            self.assertAlmostEqual(row[1], 103 / 102 - 1)
            self.assertAlmostEqual(row[2], 105 / 102 - 1)
            self.assertAlmostEqual(row[3], 107 / 102 - 1)
            self.assertEqual(row[4], 1.0)

    def test_missing_tables_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            sqlite3.connect(db).close()
            mod = load_script("build_pattern_expectancy", db)
            self.assertEqual(mod.main(), 0)


class PatternLibraryTests(unittest.TestCase):
    def test_seed_idempotent_and_teaches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            mod = load_script("seed_pattern_library", db)
            self.assertEqual(mod.main(), 0)
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            n = con.execute(
                "SELECT COUNT(*) FROM candle_pattern_library").fetchone()[0]
            self.assertEqual(n, len(mod.PATTERNS))
            desc = con.execute(
                "SELECT description FROM candle_pattern_library "
                "WHERE pattern='hammer'").fetchone()[0]
            self.assertIn("buyers took it all back", desc)
            con.close()


class CardPatternsTests(unittest.TestCase):
    def test_load_candle_patterns_joins_library_and_expectancy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            con.execute(
                """CREATE TABLE candle_patterns (
                     date TEXT, symbol TEXT, pattern TEXT,
                     direction_hint TEXT, strength REAL)""")
            con.execute(
                """CREATE TABLE candle_pattern_library (
                     pattern TEXT PRIMARY KEY, aka TEXT, description TEXT,
                     textbook_bias TEXT, candles_involved INTEGER)""")
            con.execute(
                """CREATE TABLE pattern_expectancy (
                     pattern TEXT, symbol TEXT, n INTEGER,
                     fwd_1d_mean REAL, fwd_3d_mean REAL, fwd_5d_mean REAL,
                     hit_rate_5d REAL)""")
            con.execute(
                "INSERT INTO candle_patterns VALUES "
                "('2026-10-05','SPY','hammer','bullish',0.8)")
            con.execute(
                "INSERT INTO candle_pattern_library VALUES "
                "('hammer','Hammer','desc','bullish',1)")
            con.execute(
                "INSERT INTO pattern_expectancy VALUES "
                "('hammer','SPY',42,0.001,0.002,0.0042,0.6)")
            con.commit()
            pats = card.load_candle_patterns(con, "SPY", "2026-10-05")
            con.close()
            self.assertEqual(len(pats), 1)
            self.assertEqual(pats[0][1], "Hammer")
            self.assertAlmostEqual(pats[0][4], 0.0042)
            self.assertEqual(pats[0][5], 42)

    def test_load_candle_patterns_empty_without_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            self.assertEqual(card.load_candle_patterns(con, "SPY", "2026-10-05"), [])
            con.close()


if __name__ == "__main__":
    unittest.main()
