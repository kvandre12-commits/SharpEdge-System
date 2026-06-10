from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts.agents import agent_v1_decision as agent


class AgentV1DecisionTests(unittest.TestCase):
    def test_stale_inputs_flags_missing_and_old_dates(self) -> None:
        monitor = {"latest_gap_event": {"session_date": "2026-06-01"}}
        context = {
            "risk": {"date": "2026-06-10"},
            "signal": {"date": "not-a-date"},
            "regime": {"date": "2026-06-09"},
            "options": {"session_date": None},
        }

        with patch.object(agent, "utc_now", return_value=datetime(2026, 6, 10, tzinfo=UTC)):
            stale = agent.stale_inputs(monitor, context)

        stale_by_input = {item["input"]: item for item in stale}
        self.assertEqual(stale_by_input["monitor_gap"]["age_days"], 9)
        self.assertEqual(stale_by_input["signal"]["reason"], "missing_or_invalid_date")
        self.assertEqual(stale_by_input["options_positioning"]["reason"], "missing_or_invalid_date")
        self.assertNotIn("risk_layer", stale_by_input)
        self.assertNotIn("regime", stale_by_input)

    def test_build_contract_blocks_broker_orders_even_when_trade_edge_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            health = outputs / "health"
            health.mkdir(parents=True)
            db_path = root / "spy_truth.db"
            self._seed_db(db_path)

            (outputs / "agent_controller_decision.json").write_text(
                json.dumps({"decision": "post", "confidence": 0.8}),
                encoding="utf-8",
            )
            (outputs / "robinhood_fvg_monitor.json").write_text(
                json.dumps(
                    {
                        "decision": "watch",
                        "latest_gap_event": {"session_date": "2026-06-10"},
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(outputs, db_path):
                with patch.object(agent, "utc_now", return_value=datetime(2026, 6, 10, tzinfo=UTC)):
                    contract = agent.build_contract()

        self.assertTrue(contract["trade_allowed"])
        self.assertFalse(contract["broker_order_allowed"])
        self.assertFalse(contract["permissions"]["place_order"])
        self.assertEqual(contract["required_human_action"], "confirm_order")
        self.assertEqual(contract["decision"], "operator_confirm_required")
        self.assertEqual(contract["blocking_reasons"], [])

    def test_build_contract_blocks_low_sample_and_stale_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir(parents=True)
            db_path = root / "spy_truth.db"
            self._seed_db(db_path, risk_date="2026-06-01", sample_n=3)

            (outputs / "agent_controller_decision.json").write_text(
                json.dumps({"decision": "post", "confidence": 0.7}),
                encoding="utf-8",
            )
            (outputs / "robinhood_fvg_monitor.json").write_text(
                json.dumps(
                    {
                        "decision": "watch",
                        "latest_gap_event": {"session_date": "2026-06-01"},
                    }
                ),
                encoding="utf-8",
            )

            with self._patched_paths(outputs, db_path):
                with patch.object(agent, "utc_now", return_value=datetime(2026, 6, 10, tzinfo=UTC)):
                    contract = agent.build_contract()

        self.assertFalse(contract["trade_allowed"])
        self.assertFalse(contract["broker_order_allowed"])
        self.assertIn("sample_n_below_30", contract["blocking_reasons"])
        self.assertIn("stale_or_missing_inputs", contract["blocking_reasons"])
        self.assertIn("freshness_gate_failed", contract["risk_flags"])

    def _patched_paths(self, outputs: Path, db_path: Path):
        return patch.multiple(
            agent,
            DB_PATH=db_path,
            OUTDIR=outputs,
            CONTROLLER_JSON=outputs / "agent_controller_decision.json",
            MONITOR_JSON=outputs / "robinhood_fvg_monitor.json",
            HEALTH_WARNINGS=outputs / "health" / "warnings.log",
            OUT_JSON=outputs / "agent_v1_decision.json",
            OUT_TXT=outputs / "agent_v1_decision.txt",
        )

    def _seed_db(
        self,
        db_path: Path,
        risk_date: str = "2026-06-10",
        sample_n: int = 50,
    ) -> None:
        con = sqlite3.connect(db_path)
        try:
            con.executescript(
                """
                CREATE TABLE risk_decision_layer (
                    symbol TEXT,
                    date TEXT,
                    decision_ts TEXT,
                    deployment_state TEXT,
                    sample_n INTEGER,
                    deployment_confidence REAL,
                    capital_risk_pct REAL
                );
                CREATE TABLE signals_daily (symbol TEXT, date TEXT);
                CREATE TABLE regime_daily (symbol TEXT, date TEXT);
                CREATE TABLE options_positioning_metrics (
                    underlying TEXT,
                    session_date TEXT,
                    snapshot_ts TEXT
                );
                """
            )
            con.execute(
                "INSERT INTO risk_decision_layer VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("SPY", risk_date, f"{risk_date}T20:00:00", "STANDARD", sample_n, 0.72, 0.25),
            )
            con.execute("INSERT INTO signals_daily VALUES (?, ?)", ("SPY", risk_date))
            con.execute("INSERT INTO regime_daily VALUES (?, ?)", ("SPY", risk_date))
            con.execute(
                "INSERT INTO options_positioning_metrics VALUES (?, ?, ?)",
                ("SPY", risk_date, f"{risk_date}T20:00:00"),
            )
            con.commit()
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
