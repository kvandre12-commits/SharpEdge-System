import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "reconcile_model_vs_reality.py")
_spec = importlib.util.spec_from_file_location("reconcile", _PATH)
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)


class MappingTests(unittest.TestCase):
    def test_predicted_regime(self):
        self.assertEqual(rec.predicted_regime("EXPANSION_FOLLOW_LONG"), "trend")
        self.assertEqual(rec.predicted_regime("EXPANSION_FOLLOW_SHORT"), "trend")
        self.assertEqual(rec.predicted_regime("RANGE_FADE"), "range")
        self.assertEqual(rec.predicted_regime("PIN_FADE"), "range")
        self.assertEqual(rec.predicted_regime("BALANCED_SMALL"), "neutral")
        self.assertEqual(rec.predicted_regime("WHIP_WAIT"), "neutral")
        self.assertEqual(rec.predicted_regime(None), "neutral")

    def test_predicted_direction(self):
        self.assertEqual(rec.predicted_direction("EXPANSION_FOLLOW_LONG"), 1)
        self.assertEqual(rec.predicted_direction("EXPANSION_FOLLOW_SHORT"), -1)
        self.assertEqual(rec.predicted_direction("RANGE_FADE"), 0)


class ReconcileTests(unittest.TestCase):
    def _frames(self):
        pred = pd.DataFrame({
            "session_date": ["2026-06-01", "2026-06-02", "2026-06-03"],
            "final_bias": ["EXPANSION_FOLLOW_LONG", "RANGE_FADE", "BALANCED_SMALL"],
            "prob_trend_fused": [0.7, 0.3, 0.5],
            "prob_range_fused": [0.3, 0.7, 0.5],
            "execution_score": [50.0, 5.0, 35.0],
        })
        actual = pd.DataFrame({
            "session_date": ["2026-06-01", "2026-06-02", "2026-06-03"],
            "day_type": ["trend", "range", "range"],
            "ret_1d": [0.9, -0.1, 0.2],
        })
        return pred, actual

    def test_regime_and_direction_hits(self):
        df = rec.reconcile(*self._frames())
        row = df.set_index("session_date")
        # trend call on a trend day that went up -> regime hit + dir hit
        self.assertTrue(bool(row.loc["2026-06-01", "regime_hit"]))
        self.assertTrue(bool(row.loc["2026-06-01", "dir_hit"]))
        # range_fade call on a range day -> regime hit, no directional call
        self.assertTrue(bool(row.loc["2026-06-02", "regime_hit"]))
        self.assertEqual(row.loc["2026-06-02", "pred_dir"], 0)
        # neutral (balanced) -> excluded from regime accuracy
        self.assertTrue(np.isnan(row.loc["2026-06-03", "regime_hit"]))

    def test_summary_counts(self):
        df = rec.reconcile(*self._frames())
        s = rec.summarize(df, None)
        self.assertEqual(s["sessions"], 3)
        self.assertEqual(s["decisive_regime_calls"], 2)
        self.assertEqual(s["regime_accuracy"], 1.0)
        self.assertEqual(s["directional_calls"], 1)


if __name__ == "__main__":
    unittest.main()
