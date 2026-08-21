from __future__ import annotations

import unittest

from scripts.agents import position_lab as lab


class PositionLabTests(unittest.TestCase):
    def test_single_leg_preference_requires_cheap_confirmed_runner(self) -> None:
        geometry = {
            "gamma_regime": "negative",
            "premium_read": "cheap",
            "vol_mult": 1.7,
            "setup_tag": "EXHAUSTION -> RUNNER HANDOFF",
        }

        self.assertTrue(lab.prefer_single_leg(geometry))
        self.assertFalse(lab.prefer_single_leg({**geometry, "premium_read": "rich"}))
        self.assertFalse(lab.prefer_single_leg({**geometry, "vol_mult": 0.9}))
        self.assertFalse(
            lab.prefer_single_leg({**geometry, "gamma_regime": "positive"})
        )

    def test_build_payload_prefers_branch_defined_spreads_in_positive_gamma(
        self,
    ) -> None:
        signal = {
            "symbol": "SPY",
            "display_spot": 771.34,
            "vwap": 772.04,
            "vs_vwap": -0.061,
            "vol_mult": 0.72,
            "balance_low": 771.63,
            "balance_high": 772.27,
            "pin": 772.0,
            "call_wall": 775.0,
            "put_wall": 750.0,
            "price_authority": {"display_time_utc": "2026-08-07T18:33:00+00:00"},
            "event_radar": {"headline": "Jobs report (NFP) TODAY"},
            "dealer_positioning": {
                "expiries_used": ["2026-08-07"],
                "gamma_regime": "positive",
                "dealer_state": "DEFENSIVE",
                "story": "Put-heavy OI proxy.",
            },
            "trade_permission": {
                "trade_gate": "CAUTION",
                "trade_permission_score": 70,
                "setup_conviction": {
                    "setup_tag": "STICKY DAY (calm/chop)",
                    "reason": "wait for acceptance",
                },
            },
            "premium_read": "rich",
            "exp_move_implied_usd": 2.84,
        }
        snapshot = {
            "quotes": [
                _quote(
                    "SPY260807C00772000",
                    "call",
                    772,
                    0.42,
                    0.43,
                    506867,
                    11291,
                    delta=0.45,
                ),
                _quote(
                    "SPY260807C00773000",
                    "call",
                    773,
                    0.15,
                    0.16,
                    688184,
                    9701,
                    delta=0.35,
                ),
                _quote(
                    "SPY260807P00771000",
                    "put",
                    771,
                    0.30,
                    0.31,
                    442468,
                    7012,
                    delta=-0.45,
                ),
                _quote(
                    "SPY260807P00770000",
                    "put",
                    770,
                    0.13,
                    0.14,
                    445409,
                    12936,
                    delta=-0.35,
                ),
            ]
        }
        curator = {"headline": "Watch reclaim path.", "stance": "watch_reclaim_path"}
        approval = {
            "trade_allowed": False,
            "broker_order_allowed": False,
            "decision": "monitor",
        }

        payload = lab.build_payload(
            signal, snapshot, curator, approval, requested_symbol="SPY"
        )

        self.assertEqual(payload["primary_posture"], "branch_defined_debit_spread")
        self.assertEqual(
            payload["branches"][0]["structure_family"], "call_debit_spread"
        )
        self.assertEqual(payload["branches"][1]["structure_family"], "put_debit_spread")
        self.assertEqual(payload["branches"][0]["pricing"]["breakeven"], 772.27)
        self.assertEqual(payload["branches"][1]["pricing"]["breakeven"], 770.83)
        self.assertEqual(
            payload["branches"][2]["structure_family"], "no_forced_position"
        )
        self.assertEqual(payload["branches"][0]["legs"][0]["delta"], 0.45)
        self.assertEqual(
            payload["branches"][0]["greek_dollar_plan"]["net_delta_shares"], 10.0
        )

    def test_build_payload_prefers_next_expiry_and_allows_single_put_fallback(
        self,
    ) -> None:
        signal = {
            "symbol": "SPY",
            "ts": "2026-08-07T18:44:00+00:00",
            "display_spot": 771.19,
            "vwap": 772.02,
            "vs_vwap": -0.108,
            "vol_mult": 1.7,
            "balance_low": 771.42,
            "balance_high": 772.27,
            "pin": 772.0,
            "call_wall": 775.0,
            "put_wall": 720.0,
            "dealer_positioning": {
                "gamma_regime": "positive",
                "dealer_state": "DEFENSIVE",
            },
            "trade_permission": {
                "trade_gate": "CAUTION",
                "trade_permission_score": 64,
                "setup_conviction": {
                    "setup_tag": "STICKY DAY (calm/chop)",
                    "reason": "wait",
                },
            },
            "premium_read": "rich",
            "exp_move_implied_usd": 2.69,
        }
        snapshot = {
            "quotes": [
                _quote("SPY260807C00772000", "call", 772, 0.42, 0.43, 506867, 11291),
                _quote("SPY260807P00771000", "put", 771, 0.30, 0.31, 442468, 7012),
                _quote(
                    "SPY260810C00772000",
                    "call",
                    772,
                    2.46,
                    2.53,
                    64229,
                    2775,
                    expiration="2026-08-10",
                    delta=0.46,
                ),
                _quote(
                    "SPY260810C00773000",
                    "call",
                    773,
                    1.93,
                    1.96,
                    90163,
                    1997,
                    expiration="2026-08-10",
                    delta=0.36,
                ),
                _quote(
                    "SPY260810P00771000",
                    "put",
                    771,
                    1.29,
                    1.30,
                    55870,
                    1691,
                    expiration="2026-08-10",
                    delta=-0.44,
                ),
                _quote(
                    "SPY260814C00772000",
                    "call",
                    772,
                    4.70,
                    4.74,
                    22228,
                    1689,
                    expiration="2026-08-14",
                ),
                _quote(
                    "SPY260821C00772000",
                    "call",
                    772,
                    6.20,
                    6.28,
                    18228,
                    2689,
                    expiration="2026-08-21",
                ),
                _quote(
                    "SPY260904C00772000",
                    "call",
                    772,
                    8.70,
                    8.84,
                    12228,
                    3689,
                    expiration="2026-09-04",
                ),
            ]
        }
        payload = lab.build_payload(signal, snapshot, {}, {}, requested_symbol="SPY")

        self.assertEqual(
            payload["calendar_context"]["selected_expiration"], "2026-08-10"
        )
        self.assertEqual(
            payload["calendar_context"]["selection_reason"],
            "preferred next expiry over same-day 0DTE",
        )
        self.assertEqual(
            payload["branches"][0]["structure_label"],
            "2026-08-10 772/773 call debit spread",
        )
        self.assertEqual(payload["branches"][1]["structure_family"], "long_put")
        self.assertEqual(
            payload["calendar_context"]["highlighted_expirations"],
            {
                "same_day": "2026-08-07",
                "next_expiration": "2026-08-10",
                "weekly_anchor": "2026-08-14",
                "next_weekly_anchor": "2026-08-21",
                "monthlyish_anchor": "2026-09-04",
            },
        )

    def test_build_payload_allows_single_call_when_no_clean_short_wing(self) -> None:
        signal = {
            "symbol": "SPY",
            "ts": "2026-08-07T18:44:00+00:00",
            "display_spot": 771.19,
            "vwap": 771.0,
            "vs_vwap": 0.025,
            "vol_mult": 1.7,
            "balance_low": 770.5,
            "balance_high": 772.0,
            "pin": 771.0,
            "call_wall": 775.0,
            "put_wall": 760.0,
            "dealer_positioning": {
                "gamma_regime": "negative",
                "dealer_state": "DEFENSIVE",
            },
            "trade_permission": {
                "trade_gate": "CAUTION",
                "trade_permission_score": 70,
                "setup_conviction": {
                    "setup_tag": "DOWNSIDE EXHAUSTION",
                    "reason": "reclaim",
                },
            },
            "premium_read": "normal",
            "exp_move_implied_usd": 3.0,
        }
        snapshot = {
            "quotes": [
                _quote(
                    "SPY260810C00772000",
                    "call",
                    772,
                    2.0,
                    2.1,
                    5000,
                    3000,
                    expiration="2026-08-10",
                ),
                _quote(
                    "SPY260810P00771000",
                    "put",
                    771,
                    1.5,
                    1.6,
                    5000,
                    3000,
                    expiration="2026-08-10",
                ),
            ]
        }

        payload = lab.build_payload(signal, snapshot, {}, {}, requested_symbol="SPY")

        self.assertEqual(payload["branches"][0]["structure_family"], "long_call")
        self.assertIn("single defined-debit call", payload["branches"][0]["thesis"])

    def test_render_text_mentions_execution_boundary(self) -> None:
        payload = {
            "generated_at_utc": "2026-08-07T18:40:00+00:00",
            "symbol": "SPY",
            "primary_posture": "branch_defined_debit_spread",
            "posture_reason": "Use both sides.",
            "geometry": {
                "spot": 771.34,
                "vwap": 772.04,
                "vs_vwap": -0.061,
                "gamma_regime": "positive",
                "dealer_state": "DEFENSIVE",
                "pin": 772.0,
                "call_wall": 775.0,
                "put_wall": 750.0,
                "balance_low": 771.63,
                "balance_high": 772.27,
                "setup_tag": "STICKY DAY (calm/chop)",
                "trade_gate": "CAUTION",
                "trade_permission_score": 70,
                "vol_mult": 0.72,
                "premium_read": "rich",
            },
            "freshness": {
                "quote_minutes_old_min": 2.0,
                "quote_minutes_old_max": 3.0,
                "expiration": "2026-08-07",
            },
            "branches": [
                {
                    "branch_id": "neutral_wait_branch",
                    "structure_label": "wait",
                    "status": "preferred_right_now",
                    "trigger": "wait",
                    "invalidation": "N/A",
                    "thesis": "Do not force it.",
                    "caution": "Premium is rich.",
                }
            ],
            "execution_boundary": {
                "trade_allowed": False,
                "broker_order_allowed": False,
                "decision": "monitor",
                "blocking_reasons": ["controller_hold"],
                "note": "Study only.",
            },
        }

        text = lab.render_text(payload)

        self.assertIn("SHARPEDGE POSITION LAB", text)
        self.assertIn("EXECUTION BOUNDARY", text)
        self.assertIn("controller_hold", text)


def _quote(
    symbol: str,
    option_type: str,
    strike: float,
    bid: float,
    ask: float,
    volume: int,
    open_interest: int,
    *,
    expiration: str = "2026-08-07",
    implied_volatility: float = 0.3,
    delta: float | None = None,
    gamma: float = 0.05,
    theta: float = -0.04,
    vega: float = 0.08,
) -> dict[str, object]:
    if delta is None:
        delta = 0.45 if option_type == "call" else -0.45
    return {
        "contract_symbol": symbol,
        "underlying": "SPY",
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "midpoint": round((bid + ask) / 2, 3),
        "volume": volume,
        "open_interest": open_interest,
        "width_pct": round((ask - bid) / ((bid + ask) / 2), 6),
        "implied_volatility": implied_volatility,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "quote_timestamp": "2026-08-07T18:21:40+00:00",
        "fetch_timestamp": "2026-08-07T18:36:41+00:00",
        "fresh_quote_required": True,
        "manual_validation_priority": "high",
        "rejection_flags": "",
    }


if __name__ == "__main__":
    unittest.main()
