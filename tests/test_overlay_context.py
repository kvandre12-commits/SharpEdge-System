import importlib.util
import os
import sqlite3
import unittest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "build_overlay_context_daily.py"
)
_spec = importlib.util.spec_from_file_location("ovl", _PATH)
ovl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ovl)


def _seed():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bars_daily (date TEXT, symbol TEXT, close REAL)")
    con.executemany(
        "INSERT INTO bars_daily VALUES (?,?,?)",
        [
            ("2026-06-01", "SPY", 500),
            ("2026-06-02", "SPY", 501),
            ("2026-06-03", "SPY", 502),
        ],
    )
    con.execute(
        "CREATE TABLE overlays_daily (date TEXT, symbol TEXT, overlay_type TEXT, overlay_strength REAL, notes TEXT)"
    )
    con.executemany(
        "INSERT INTO overlays_daily VALUES (?,?,?,?,?)",
        [
            ("2026-06-01", "SPY", "vix", 0.5, ""),
            ("2026-06-01", "SPY", "vix3m", 0.7, ""),
            ("2026-06-01", "SPY", "darkpool", 1.0, ""),
            ("2026-06-03", "SPY", "vix", 0.2, ""),
        ],
    )
    con.execute(
        "CREATE TABLE ats_weekly (week_start TEXT, symbol TEXT, shares_z_26w REAL, "
        "trades_vs_13w_avg REAL, shares_vs_13w_avg REAL, avg_trade_size REAL, ingest_ts TEXT)"
    )
    con.executemany(
        "INSERT INTO ats_weekly VALUES (?,?,?,?,?,?,?)",
        [
            ("2026-05-25", "SPY", 1.5, 1.1, 1.2, 200.0, ""),
            ("2026-06-02", "SPY", -0.4, 0.9, 0.8, 150.0, ""),
        ],
    )
    return con


class OverlayContextTests(unittest.TestCase):
    def setUp(self):
        self.rows = ovl.build(_seed())
        self.by_date = {row["date"]: row for row in self.rows}

    def test_spine_is_trading_days(self):
        self.assertEqual(
            [row["date"] for row in self.rows],
            ["2026-06-01", "2026-06-02", "2026-06-03"],
        )

    def test_macro_forward_fill(self):
        self.assertAlmostEqual(self.by_date["2026-06-02"]["ovl_vix"], 0.5)
        self.assertAlmostEqual(self.by_date["2026-06-03"]["ovl_vix"], 0.2)

    def test_vix_contango_derived(self):
        self.assertAlmostEqual(self.by_date["2026-06-01"]["ovl_vix_contango"], 0.2)

    def test_weekly_darkpool_asof(self):
        self.assertAlmostEqual(self.by_date["2026-06-01"]["dp_shares_z_26w"], 1.5)
        self.assertAlmostEqual(self.by_date["2026-06-03"]["dp_shares_z_26w"], -0.4)


if __name__ == "__main__":
    unittest.main()
