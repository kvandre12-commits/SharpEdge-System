from __future__ import annotations

import unittest

from scripts.agents.overnight_carry_logic import build_payload


class OvernightCarryTests(unittest.TestCase):
    def test_build_payload_reproduces_qqq_field_brief_core_numbers(self) -> None:
        payload = build_payload(
            {
                "symbol": "QQQ",
                "spot": 620.76,
                "strike": 621.0,
                "option_type": "call",
                "expiration": "2026-01-23",
                "delta": 0.488,
                "gamma": 0.0805,
                "theta": -1.04,
                "iv": 0.153,
                "close_timestamp": "2026-01-22T21:00:00+00:00",
                "close_to_open_hours": 17.5,
            },
            {"source": "unit", "rows": []},
        )

        open_ctx = payload["overnight_open"]
        self.assertEqual(payload["schema"], "sharpedge.overnight_carry_brief.v1")
        self.assertAlmostEqual(open_ctx["theta_carry_per_share"], 0.7583, places=4)
        self.assertAlmostEqual(open_ctx["theta_carry_contract"], 75.83, places=2)
        self.assertAlmostEqual(open_ctx["break_even_move_dollars"], 1.394, places=3)
        self.assertAlmostEqual(open_ctx["break_even_move_pct"], 0.225, places=3)
        self.assertEqual(payload["assumptions"]["u_shape_normalization_scalar"], 0.91549)
        first_u = payload["intraday"]["u_shape_bands"][0]
        last_u = payload["intraday"]["u_shape_bands"][-1]
        self.assertEqual(first_u["time_et"], "10:30")
        self.assertAlmostEqual(first_u["band_1sigma_dollars"], 2.84, places=2)
        self.assertAlmostEqual(last_u["band_95_dollars"], 11.73, places=2)
        self.assertEqual(open_ctx["open_iv_shock_scenarios"][0]["iv_points"], -5)
        self.assertEqual(payload["contract"]["vega_source"], "estimated")

    def test_gap_probabilities_compute_from_history_rows(self) -> None:
        payload = build_payload(
            {
                "symbol": "QQQ",
                "spot": 620.76,
                "strike": 621.0,
                "option_type": "call",
                "expiration": "2026-01-23",
                "delta": 0.488,
                "gamma": 0.0805,
                "theta": -1.04,
                "iv": 0.153,
                "close_timestamp": "2026-01-21T21:00:00+00:00",
                "close_to_open_hours": 17.5,
            },
            {
                "source": "unit",
                "rows": [
                    {"session_date": "2026-01-01", "gap_pct": 0.0030},
                    {"session_date": "2026-01-08", "gap_pct": 0.0010},
                    {"session_date": "2026-01-15", "gap_pct": -0.0040},
                    {"session_date": "2026-01-22", "gap_pct": 0.0030},
                ],
            },
        )

        empirical = payload["overnight_open"]["empirical_gap_context"]
        self.assertTrue(empirical["available"])
        self.assertEqual(empirical["sample_size"], 4)
        self.assertEqual(empirical["weekday_sample_size"], 4)
        self.assertEqual(empirical["probability_favorable_gap_pct"], 50.0)
        self.assertEqual(empirical["probability_adverse_gap_pct"], 25.0)
        self.assertEqual(empirical["probability_abs_gap_pct"], 75.0)
        self.assertEqual(empirical["weekday_probability_favorable_gap_pct"], 50.0)

    def test_conditioned_gap_probabilities_use_overlap_dates(self) -> None:
        payload = build_payload(
            {
                "symbol": "QQQ",
                "spot": 620.76,
                "strike": 621.0,
                "option_type": "call",
                "expiration": "2026-01-23",
                "delta": 0.488,
                "gamma": 0.0805,
                "theta": -1.04,
                "iv": 0.153,
                "close_timestamp": "2026-01-21T21:00:00+00:00",
                "close_to_open_hours": 17.5,
            },
            {
                "source": "unit",
                "rows": [
                    {"session_date": "2026-01-01", "gap_pct": 0.0030},
                    {"session_date": "2026-01-08", "gap_pct": 0.0010},
                    {"session_date": "2026-01-15", "gap_pct": -0.0040},
                    {"session_date": "2026-01-22", "gap_pct": 0.0030},
                ],
            },
            {
                "source": "spy_truth.db",
                "proxy_symbol": "SPY",
                "filters": {"event_type": "FAILED_BREAKDOWN"},
                "session_dates": ["2026-01-01", "2026-01-15"],
                "match_count": 2,
                "reason": None,
            },
        )

        conditioned = payload["overnight_open"]["conditioned_gap_context"]
        self.assertTrue(conditioned["available"])
        self.assertEqual(conditioned["proxy_symbol"], "SPY")
        self.assertEqual(conditioned["context_match_count"], 2)
        self.assertEqual(conditioned["overlap_sample_size"], 2)
        self.assertEqual(conditioned["probability_favorable_gap_pct"], 50.0)
        self.assertEqual(conditioned["probability_adverse_gap_pct"], 50.0)
        self.assertEqual(conditioned["probability_abs_gap_pct"], 100.0)

    def test_comparison_presets_are_carried_into_payload(self) -> None:
        payload = build_payload(
            {
                "symbol": "QQQ",
                "spot": 620.76,
                "strike": 621.0,
                "option_type": "call",
                "expiration": "2026-01-23",
                "delta": 0.488,
                "gamma": 0.0805,
                "theta": -1.04,
                "iv": 0.153,
                "close_timestamp": "2026-01-21T21:00:00+00:00",
                "close_to_open_hours": 17.5,
            },
            {
                "source": "unit",
                "rows": [
                    {"session_date": "2026-01-01", "gap_pct": 0.0030},
                    {"session_date": "2026-01-08", "gap_pct": 0.0010},
                    {"session_date": "2026-01-15", "gap_pct": -0.0040},
                    {"session_date": "2026-01-22", "gap_pct": 0.0030},
                ],
            },
            None,
            [
                {
                    "label": "failed_breakdown",
                    "source": "spy_truth.db",
                    "proxy_symbol": "SPY",
                    "filters": {"event_type": "FAILED_BREAKDOWN"},
                    "session_dates": ["2026-01-01", "2026-01-15"],
                    "match_count": 2,
                    "reason": None,
                }
            ],
        )

        comparison = payload["overnight_open"]["comparison_presets"]
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["label"], "failed_breakdown")
        self.assertEqual(comparison[0]["gap_context"]["context_match_count"], 2)
        self.assertEqual(comparison[0]["gap_context"]["probability_abs_gap_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
