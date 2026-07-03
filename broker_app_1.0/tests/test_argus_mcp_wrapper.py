from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

wrapper = importlib.import_module("argus_mcp_wrapper")


class ArgusWrapperTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.broker_app_root = root / "broker_app_1.0"
        self.outputs_dir = root / "outputs"
        self.bridge_root = root / "SharpEdge-Robinhood-Bridge"
        (self.broker_app_root / "bridge").mkdir(parents=True)
        (self.broker_app_root / "tools").mkdir(parents=True)
        (self.broker_app_root / "manifests").mkdir(parents=True)
        (self.broker_app_root / "docs").mkdir(parents=True)
        self.outputs_dir.mkdir(parents=True)
        (self.bridge_root / "src" / "sharpedge_robinhood_bridge").mkdir(parents=True)
        self.context = wrapper.WrapperContext(
            broker_app_root=self.broker_app_root,
            sharpedge_root=root,
            bridge_root=self.bridge_root,
            outputs_dir=self.outputs_dir,
        )
        self._write_contract_files()
        self._write_signal()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_contract_files(self) -> None:
        self._write_json(
            self.broker_app_root / "manifests" / "argus_mcp_manifest.json",
            {
                "authority_boundary": {
                    "argus": ["discover_available_surfaces"],
                    "sharpedge": ["execution_permission_authority"],
                    "bridge": ["broker_execution_routing"],
                }
            },
        )
        self._write_json(
            self.broker_app_root / "tools" / "argus_tool_aliases.json",
            {
                "canonical_names": list(wrapper.TOOL_NAMES),
                "aliases": [{"argus_tool": name} for name in wrapper.TOOL_NAMES],
            },
        )
        self._write_json(
            self.broker_app_root / "bridge" / "real_surface_inventory.json",
            {
                "tools": [
                    {
                        "name": "sharpedge.discover_surface",
                        "authority": "Argus-MCP-Wrapper",
                        "purpose": "discover",
                        "mutability": "read_only",
                        "status": "wrapper_contract_defined",
                        "backing": ["inventory"],
                    }
                ],
                "resources": [
                    {
                        "name": "sharpedge://state/latest",
                        "authority": "SharpEdge",
                        "purpose": "state",
                        "mutability": "read_only",
                        "status": "semantic_backing_exists",
                        "backing": ["signal.json"],
                    }
                ],
            },
        )
        (self.broker_app_root / "docs" / "authority_map.md").write_text(
            "authority map", encoding="utf-8"
        )

    def _write_signal(self) -> None:
        self._write_json(
            self.outputs_dir / "signal.json",
            {
                "schema": "sharpedge.signal.v1",
                "symbol": "SPY",
                "spot": 620.15,
                "gamma_regime": "negative",
                "setup_tag": "failed_break_reclaim",
                "trade_permission": {
                    "schema": "sharpedge.trade_permission.v1",
                    "trade_permission_score": 73,
                    "trade_gate": "pass",
                    "bias": "BULLISH",
                    "execution_flow": {"gate": "pass"},
                    "execution_hierarchy": {"primary": "flow"},
                    "supporting_reasons": ["support one"],
                    "warning_reasons": ["warning one"],
                },
            },
        )

    def test_discover_surface_returns_inventory(self) -> None:
        payload = wrapper.discover_surface(context=self.context)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["tool_name"], "sharpedge.discover_surface")
        self.assertTrue(payload["surface"]["tools"])
        self.assertIn("argus", payload["surface"]["authority_boundary"])

    def test_get_execution_card_and_explain_permission_use_signal(self) -> None:
        card_payload = wrapper.get_execution_card(context=self.context)
        self.assertEqual(card_payload["status"], "ok")
        self.assertEqual(
            card_payload["execution_card"]["trade_permission_score"],
            73,
        )
        explain_payload = wrapper.explain_permission(context=self.context)
        self.assertEqual(explain_payload["status"], "ok")
        self.assertEqual(explain_payload["explanation"]["score"], 73)
        self.assertIn(
            "Permission is 73", explain_payload["explanation"]["plain_language_summary"]
        )

    def test_prepare_broker_handoff_blocks_without_operator_approval(self) -> None:
        payload = wrapper.prepare_broker_handoff(
            operator_approved=False,
            context=self.context,
        )
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["error_code"], "operator_approval_required")

    def test_prepare_broker_handoff_uses_bridge_exports(self) -> None:
        handoff = {
            "schema": "sharpedge.robinhood_execution_handoff.v1",
            "command_plan": {
                "command": "order_submit",
                "route": "chatgpt_delegate",
                "approval_policy": "operator_confirm_required",
                "status": "awaiting_operator_confirm",
            },
            "delegation": {
                "broker_payload": {
                    "payload_contracts": {
                        "schema": "sharpedge.connector_payload_contracts.v1"
                    }
                }
            },
            "operator_gate": {"required": True},
        }

        def fake_plan(signal_path: Path, *, command: str, test: bool) -> dict:
            self.assertEqual(signal_path, self.context.signal_path)
            self.assertEqual(command, "order_submit")
            self.assertFalse(test)
            return handoff

        def fake_write(
            payload: dict, out_dir: Path | None = None, *, latest_name: str = ""
        ) -> Path:
            self.assertEqual(payload, handoff)
            self.assertEqual(out_dir, self.context.outputs_dir)
            self.assertEqual(latest_name, "robinhood_execution_handoff.json")
            target = self.context.outputs_dir / latest_name
            target.write_text(json.dumps(payload), encoding="utf-8")
            return target

        with patch.object(
            wrapper,
            "_bridge_exports",
            return_value={
                "plan_signal_handoff": fake_plan,
                "write_handoff_artifact": fake_write,
            },
        ):
            payload = wrapper.prepare_broker_handoff(context=self.context)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["artifact_path"], str(self.context.handoff_path))
        self.assertEqual(
            payload["handoff"]["schema"], "sharpedge.robinhood_execution_handoff.v1"
        )

    def test_validate_handoff_checks_payload_contracts(self) -> None:
        good_handoff = {
            "schema": "sharpedge.robinhood_execution_handoff.v1",
            "command_plan": {
                "route": "chatgpt_delegate",
                "approval_policy": "operator_confirm_required",
                "status": "awaiting_operator_confirm",
            },
            "delegation": {
                "broker_payload": {
                    "payload_contracts": {
                        "schema": "sharpedge.connector_payload_contracts.v1"
                    }
                }
            },
            "operator_gate": {"required": True},
        }
        self._write_json(self.context.handoff_path, good_handoff)
        payload = wrapper.validate_handoff(context=self.context)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["validation"]["valid"])

        bad_handoff = dict(good_handoff)
        bad_handoff["delegation"] = {"broker_payload": {}}
        self._write_json(self.context.handoff_path, bad_handoff)
        blocked = wrapper.validate_handoff(context=self.context)
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["error_code"], "handoff_not_ready")
        self.assertFalse(blocked["validation"]["ready_for_delegation"])


if __name__ == "__main__":
    unittest.main()
