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


SNAP_SCHEMA = """
CREATE TABLE options_chain_snapshots (
  snapshot_ts  TEXT NOT NULL,
  session_date TEXT NOT NULL,
  underlying   TEXT NOT NULL,
  expiry_date  TEXT NOT NULL,
  dte          INTEGER NOT NULL,
  strike       REAL NOT NULL,
  call_oi      INTEGER,
  put_oi       INTEGER,
  call_volume  INTEGER,
  put_volume   INTEGER,
  call_gamma   REAL,
  put_gamma    REAL,
  source       TEXT DEFAULT 'alpaca',
  PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike)
);
"""


def seed_snapshots(con, rows):
    con.execute(SNAP_SCHEMA)
    con.executemany(
        """INSERT INTO options_chain_snapshots
           (snapshot_ts, session_date, underlying, expiry_date, dte, strike,
            call_oi, put_oi, call_volume, put_volume)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()


def snap_row(ts, date, strike, c_oi, p_oi, c_v=0, p_v=0, exp="2026-10-09", dte=2):
    return (ts, date, "SPY", exp, dte, strike, c_oi, p_oi, c_v, p_v)


class DiffSnapshotsTests(unittest.TestCase):
    def _db_two_snaps(self, tmp):
        db = str(Path(tmp) / "t.db")
        con = sqlite3.connect(db)
        t1, t2 = "2026-10-05T14:30:00Z", "2026-10-05T17:00:00Z"
        rows = []
        for strike in (590.0, 600.0, 610.0):
            rows.append(snap_row(t1, "2026-10-05", strike, 1000, 1000))
            c_oi = 1000 + (5000 if strike == 600.0 else 0)
            rows.append(snap_row(t2, "2026-10-05", strike, c_oi, 1000))
        seed_snapshots(con, rows)
        con.close()
        return db

    def test_diff_flags_top_movers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_two_snaps(tmp)
            mod = load_script("diff_options_snapshots", db,
                              {"FLOW_MIN_OI": "100", "FLOW_TOP_N": "5"})
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            n1 = con.execute("SELECT COUNT(*) FROM options_flow_events").fetchone()[0]
            # only the 600 strike clears the 100-contract threshold -> 1 strike x 2 sides
            self.assertEqual(n1, 2)
            top = con.execute(
                "SELECT strike, side, d_oi FROM options_flow_events "
                "WHERE flow_rank = 1").fetchall()
            self.assertEqual(len(top), 2)
            self.assertTrue(all(r[0] == 600.0 for r in top))
            call_row = [r for r in top if r[1] == "call"][0]
            self.assertEqual(call_row[2], 5000)
            # idempotent
            self.assertEqual(mod.main(), 0)
            n2 = con.execute("SELECT COUNT(*) FROM options_flow_events").fetchone()[0]
            self.assertEqual(n1, n2)
            con.close()

    def test_diff_respects_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_two_snaps(tmp)
            mod = load_script("diff_options_snapshots", db,
                              {"FLOW_MIN_OI": "999999", "FLOW_TOP_N": "5"})
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            n = con.execute("SELECT COUNT(*) FROM options_flow_events").fetchone()[0]
            self.assertEqual(n, 0)
            con.close()

    def test_diff_single_snapshot_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            con = sqlite3.connect(db)
            seed_snapshots(con, [snap_row("2026-10-05T14:30:00Z", "2026-10-05", 600.0, 100, 100)])
            con.close()
            mod = load_script("diff_options_snapshots", db)
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            n = con.execute("SELECT COUNT(*) FROM options_flow_events").fetchone()[0]
            self.assertEqual(n, 0)
            con.close()

    def test_diff_no_table_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            sqlite3.connect(db).close()
            mod = load_script("diff_options_snapshots", db)
            self.assertEqual(mod.main(), 0)


class SnapshotFeaturesTests(unittest.TestCase):
    def _db_two_sessions(self, tmp):
        db = str(Path(tmp) / "t.db")
        con = sqlite3.connect(db)
        rows = []
        for date, ts, mp_strike in (("2026-10-02", "2026-10-02T20:05:00Z", 600.0),
                                    ("2026-10-05", "2026-10-05T20:05:00Z", 605.0)):
            for strike in (590.0, 600.0, 605.0, 610.0):
                c_oi = 5000 if strike == mp_strike else 100
                p_oi = 5000 if strike == mp_strike else 100
                rows.append(snap_row(ts, date, strike, c_oi, p_oi))
        seed_snapshots(con, rows)
        con.execute(
            """CREATE TABLE options_positioning_metrics (
                 snapshot_ts TEXT, session_date TEXT, underlying TEXT,
                 dte_min INTEGER, dte_max INTEGER,
                 pcr_oi REAL, pcr_vol REAL, spot REAL)""")
        con.execute(
            "INSERT INTO options_positioning_metrics VALUES "
            "('2026-10-02T20:05:00Z','2026-10-02','SPY',0,1,0.8,0.9,598.0),"
            "('2026-10-05T20:05:00Z','2026-10-05','SPY',0,1,1.1,1.0,603.0)")
        con.commit()
        con.close()
        return db

    def test_builds_features_with_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db_two_sessions(tmp)
            mod = load_script("build_options_snapshot_features", db)
            self.assertEqual(mod.main(), 0)
            con = sqlite3.connect(db)
            rows = {r[0]: r for r in con.execute(
                "SELECT session_date, n_snapshots, pcr_oi, max_pain, "
                "d_max_pain_1d, d_pcr_oi_1d FROM options_snapshot_features")}
            self.assertEqual(len(rows), 2)
            # max pain concentrates OI at the heavy strike
            self.assertEqual(rows["2026-10-02"][3], 600.0)
            self.assertEqual(rows["2026-10-05"][3], 605.0)
            # velocity vs previous session
            self.assertEqual(rows["2026-10-05"][4], 5.0)
            self.assertAlmostEqual(rows["2026-10-05"][5], 0.3)
            # first session has no previous -> NULL velocity
            self.assertIsNone(rows["2026-10-02"][4])
            con.close()

    def test_no_table_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            sqlite3.connect(db).close()
            mod = load_script("build_options_snapshot_features", db)
            self.assertEqual(mod.main(), 0)


class CardFlowTests(unittest.TestCase):
    def test_load_options_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            con.execute(
                """CREATE TABLE options_flow_events (
                     snapshot_ts TEXT, session_date TEXT, underlying TEXT,
                     expiry_date TEXT, dte INTEGER, strike REAL, side TEXT,
                     d_oi INTEGER, d_volume INTEGER, oi_now INTEGER,
                     volume_now INTEGER, flow_rank INTEGER)""")
            con.execute(
                "INSERT INTO options_flow_events VALUES "
                "('2026-10-05T20:05:00Z','2026-10-05','SPY','2026-10-09',2,600.0,'call',5000,200,6000,200,1),"
                "('2026-10-05T20:05:00Z','2026-10-05','SPY','2026-10-09',2,600.0,'put',-300,100,700,100,1)")
            con.commit()
            flow = card.load_options_flow(con, "SPY", "2026-10-05")
            con.close()
            self.assertEqual(len(flow["events"]), 2)
            self.assertEqual(flow["net_call"], 5000)
            self.assertEqual(flow["net_put"], -300)

    def test_load_options_flow_empty_without_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "t.db")
            card = load_script("print_today_trade_card", db)
            con = sqlite3.connect(db)
            flow = card.load_options_flow(con, "SPY", "2026-10-05")
            con.close()
            self.assertEqual(flow["events"], [])


if __name__ == "__main__":
    unittest.main()
