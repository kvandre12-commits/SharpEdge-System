from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.build_robinhood_fvg_monitor as monitor


class RobinhoodFvgMonitorTests(unittest.TestCase):
    def test_build_decision_blocks_completed_gap_and_low_sample(self) -> None:
        latest_event = {"fill_completed": "1", "gap_direction": "UP"}
        edge = {"n": "2", "fill_rate": "0.80", "tradability_score": "0.90"}
        risk = {"deployment_state": "STANDARD"}

        decision, reasons = monitor.build_decision(latest_event, edge, risk)

        self.assertEqual(decision, "no_trade")
        self.assertIn("gap_already_filled", reasons)
        self.assertIn("low_sample_n<5", reasons)

    def test_build_decision_allows_watch_for_clean_monitoring_setup(self) -> None:
        latest_event = {"fill_completed": "0", "gap_direction": "DOWN"}
        edge = {"n": "12", "fill_rate": "0.70", "tradability_score": "0.60"}
        risk = {"deployment_state": "STANDARD"}

        decision, reasons = monitor.build_decision(latest_event, edge, risk)

        self.assertEqual(decision, "watch")
        self.assertEqual(reasons, ["eligible_for_robinhood_quote_monitoring_only"])

    def test_build_payload_blocks_orders_and_uses_fixture_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            config_dir = root / ".code_puppy"
            db_path = root / "spy_truth.db"
            self._write_csvs(outputs)
            self._seed_db(db_path)

            with patch.multiple(
                monitor,
                DB_PATH=db_path,
                OUTDIR=outputs,
                GAP_EVENTS_CSV=outputs / "gap_excursion_metrics.csv",
                TOP_EDGES_CSV=outputs / "top_gap_fill_edges.csv",
                OUT_JSON=outputs / "robinhood_fvg_monitor.json",
                OUT_TXT=outputs / "robinhood_fvg_monitor.txt",
                CODE_PUPPY_CONFIG_DIR=config_dir,
                MCP_REGISTRY_JSON=config_dir / "mcp_registry.json",
                MCP_BINDINGS_JSON=config_dir / "mcp_agent_bindings.json",
            ):
                payload = monitor.build_payload()

        handoff = payload["robinhood_mcp_handoff"]
        self.assertEqual(payload["decision"], "watch")
        self.assertEqual(payload["directional_hypothesis"]["option_side_watch"], "puts_or_put_spreads")
        self.assertIn("place_order", handoff["blocked_actions"])
        self.assertNotIn("place_order", handoff["permitted_actions"])
        self.assertTrue(handoff["operator_confirmation_required_for_orders"])
        self.assertTrue(handoff["manual_review_required"])
        self.assertEqual(handoff["bridge_status"]["status"], "unconfigured")
        self.assertEqual(payload["risk_context"]["deployment_state"], "STANDARD")
        self.assertEqual(payload["options_context"]["dealer_state_hint"], "short_gamma")

    def test_bridge_status_reports_disabled_server_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".code_puppy"
            config_dir.mkdir()
            (config_dir / "mcp_registry.json").write_text(
                json.dumps(
                    {
                        "abc123": {
                            "name": "robinhood-trading",
                            "enabled": False,
                            "config": {"url": "https://agent.robinhood.com/mcp/trading"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "mcp_agent_bindings.json").write_text(
                json.dumps(
                    {
                        "bindings": {
                            "code-puppy": {"robinhood-trading": {"auto_start": True}}
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.multiple(
                monitor,
                CODE_PUPPY_CONFIG_DIR=config_dir,
                MCP_REGISTRY_JSON=config_dir / "mcp_registry.json",
                MCP_BINDINGS_JSON=config_dir / "mcp_agent_bindings.json",
            ):
                status = monitor.robinhood_bridge_status()

        self.assertFalse(status["available"])
        self.assertTrue(status["registered"])
        self.assertTrue(status["agent_bound"])
        self.assertEqual(status["status"], "disabled")
        self.assertEqual(status["reason"], "server_disabled")

    def _write_csvs(self, outputs: Path) -> None:
        self._write_rows(
            outputs / "gap_excursion_metrics.csv",
            [
                {
                    "symbol": "SPY",
                    "session_date": "2026-06-10",
                    "event_type": "GAP",
                    "gap_direction": "UP",
                    "gap_pct": "0.5",
                    "gap_fill_level": "500",
                    "fill_completed": "0",
                    "vol_state": "LOW",
                    "macro_state": "CALM",
                    "dp_state": "NEUTRAL",
                    "open_regime_label": "OPEN_DRIVE",
                    "fill_path_type": "DIRECT",
                    "setup_dir": "SHORT",
                    "key_source": "fixture",
                }
            ],
        )
        self._write_rows(
            outputs / "top_gap_fill_edges.csv",
            [
                {
                    "event_type": "GAP",
                    "gap_direction": "UP",
                    "fill_path_type": "DIRECT",
                    "n": "10",
                    "fill_rate": "0.70",
                    "direct_fill_rate": "0.60",
                    "failed_fill_rate": "0.20",
                    "avg_time_to_fill_minutes": "45",
                    "avg_MAE_pct": "0.1",
                    "avg_MFE_pct": "0.3",
                    "tradability_score": "0.80",
                    "sample_quality": "ok",
                    "vol_state": "LOW",
                    "macro_state": "CALM",
                    "dp_state": "NEUTRAL",
                    "open_regime_label": "OPEN_DRIVE",
                    "setup_dir": "SHORT",
                    "key_source": "fixture",
                }
            ],
        )

    def _write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _seed_db(self, db_path: Path) -> None:
        con = sqlite3.connect(db_path)
        try:
            con.executescript(
                """
                CREATE TABLE options_positioning_metrics (
                    underlying TEXT,
                    session_date TEXT,
                    snapshot_ts TEXT,
                    spot REAL,
                    atm_strike REAL,
                    dealer_state_hint TEXT
                );
                CREATE TABLE risk_decision_layer (
                    symbol TEXT,
                    date TEXT,
                    decision_ts TEXT,
                    deployment_state TEXT,
                    deployment_confidence REAL,
                    capital_risk_pct REAL,
                    tradability_score REAL,
                    sample_n INTEGER
                );
                """
            )
            con.execute(
                "INSERT INTO options_positioning_metrics VALUES (?, ?, ?, ?, ?, ?)",
                ("SPY", "2026-06-10", "2026-06-10T20:00:00", 501.0, 501.0, "short_gamma"),
            )
            con.execute(
                "INSERT INTO risk_decision_layer VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("SPY", "2026-06-10", "2026-06-10T20:00:00", "STANDARD", 0.7, 0.25, 0.8, 40),
            )
            con.commit()
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
