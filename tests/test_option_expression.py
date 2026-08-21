from __future__ import annotations

import unittest

from scripts.agents import option_expression as expr


class OptionExpressionTests(unittest.TestCase):
    def test_build_payload_translates_greeks_into_dollars(self) -> None:
        payload = expr.build_payload(
            signal={"symbol": "SPY"},
            position_lab={
                "symbol": "SPY",
                "geometry": {
                    "setup_tag": "UPSIDE EXHAUSTION",
                    "gamma_regime": "positive",
                    "dealer_state": "DEFENSIVE",
                    "premium_read": "rich",
                    "spot": 50.0,
                    "pin": 50.0,
                    "call_wall": 52.0,
                    "put_wall": 48.0,
                    "exp_move_implied_usd": 2.0,
                },
                "branches": [
                    {
                        "branch_id": "bull_reclaim_branch",
                        "direction": "CALLS",
                        "status": "watch_only_until_trigger",
                        "structure_family": "call_debit_spread",
                        "structure_label": "2026-08-10 50/51 call debit spread",
                        "trigger": "reclaim 50.5",
                        "invalidation": "lose 49.5",
                        "thesis": "Defined upside.",
                        "caution": "Do not chase.",
                        "pricing": {"debit": 0.8, "max_loss": 0.8, "max_gain": 0.2},
                        "levels": {"spot": 50.0, "trigger_level": 50.5, "invalidation_level": 49.5},
                        "legs": [
                            {
                                "side": "buy",
                                "delta": 0.45,
                                "gamma": 0.06,
                                "theta": -0.04,
                                "vega": 0.09,
                            },
                            {
                                "side": "sell",
                                "delta": 0.30,
                                "gamma": 0.03,
                                "theta": -0.02,
                                "vega": 0.04,
                            },
                        ],
                    }
                ],
            },
            approval={"decision": "monitor", "trade_allowed": False},
        )

        branch = payload["branch_expressions"][0]
        greek_plan = branch["greek_dollar_plan"]
        self.assertEqual(payload["schema"], "sharpedge.option_expression.v1")
        self.assertEqual(greek_plan["net_delta_shares"], 15.0)
        self.assertEqual(greek_plan["net_gamma_share_change_per_1pt"], 3.0)
        self.assertEqual(greek_plan["theta_dollars_per_day"], -2.0)
        self.assertEqual(greek_plan["vega_dollars_per_5iv"], 25.0)
        self.assertIn("donating", payload["expression_doctrine"]["core_rule"])


if __name__ == "__main__":
    unittest.main()
