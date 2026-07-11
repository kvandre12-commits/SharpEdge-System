from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cockpit.ace_snapshot import build_ace_snapshot, write_ace_snapshot


class AceSnapshotTests(unittest.TestCase):
    def test_build_snapshot_keeps_only_ace_fields(self) -> None:
        rows = [(0, 100.0, 101.0, 99.5, 100.5, 1200)]
        pa = {
            "spot": 100.5,
            "vwap": 100.1,
            "vs_vwap": 0.4,
            "mom15": 0.6,
            "vol_mult": 1.4,
            "rng_pos": 70,
            "balance_state": "above",
            "position_in_balance": 0.9,
            "balance_reference": "opening balance",
            "extra_noise": "ignore me",
        }
        levels = {"ORH": 100.2, "ORL": 99.8, "PDH": 100.7, "PDL": 99.4, "PDC": 100.0, "VWAP": 100.1}
        op = {"call_wall": 101.5, "put_wall": 99.9, "atm_iv": 0.21}
        gp = {"regime": "negative", "pin": 101.0, "max_pain": 100.0}

        snapshot = build_ace_snapshot(rows, pa, levels, op, gp)

        self.assertEqual(snapshot["schema"], "sharpedge.ace_snapshot.v1")
        self.assertEqual(snapshot["bars"], [[0, 100.0, 101.0, 99.5, 100.5, 1200]])
        self.assertNotIn("extra_noise", snapshot["pa"])
        self.assertNotIn("VWAP", snapshot["levels"])
        self.assertNotIn("atm_iv", snapshot["op"])
        self.assertNotIn("max_pain", snapshot["gp"])

    def test_write_snapshot_creates_artifact(self) -> None:
        rows = [(0, 100.0, 101.0, 99.5, 100.5, 1200)]
        pa = {"spot": 100.5, "vwap": 100.1, "vs_vwap": 0.4, "mom15": 0.6, "vol_mult": 1.4, "rng_pos": 70}
        levels = {"ORH": 100.2, "ORL": 99.8}
        op = {"call_wall": 101.5, "put_wall": 99.9}
        gp = {"regime": "negative", "pin": 101.0}

        with tempfile.TemporaryDirectory() as tmp:
            path = write_ace_snapshot(rows, pa, levels, op, gp, tmp)
            self.assertTrue(path.exists())
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "sharpedge.ace_snapshot.v1")


if __name__ == "__main__":
    unittest.main()
