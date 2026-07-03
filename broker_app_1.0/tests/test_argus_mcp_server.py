from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

BROKER_APP_ROOT = Path(__file__).resolve().parents[1]
if str(BROKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(BROKER_APP_ROOT))

CapabilityProfile = importlib.import_module("mcp.auth").CapabilityProfile
ArgusMCPServer = importlib.import_module("mcp.server").ArgusMCPServer
TRACE_FILE_NAME = importlib.import_module("mcp.tracing").TRACE_FILE_NAME
WrapperContext = importlib.import_module("runtime.argus_mcp_wrapper").WrapperContext


class ArgusMCPServerTestCase(unittest.TestCase):
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
        self.context = WrapperContext(
            broker_app_root=self.broker_app_root,
            sharpedge_root=root,
            bridge_root=self.bridge_root,
            outputs_dir=self.outputs_dir,
        )
        self._write_json(
            self.broker_app_root / "manifests" / "argus_mcp_manifest.json",
            {"authority_boundary": {"argus": [], "sharpedge": [], "bridge": []}},
        )
        self._write_json(
            self.broker_app_root / "tools" / "argus_tool_aliases.json",
            {
                "canonical_names": [
                    "sharpedge.discover_surface",
                    "sharpedge.get_latest_state",
                    "sharpedge.get_execution_card",
                    "sharpedge.explain_permission",
                    "sharpedge.prepare_broker_handoff",
                    "sharpedge.validate_handoff",
                ],
                "aliases": [],
            },
        )
        self._write_json(
            self.broker_app_root / "bridge" / "real_surface_inventory.json",
            {"tools": [], "resources": []},
        )
        (self.broker_app_root / "docs" / "authority_map.md").write_text(
            "authority map", encoding="utf-8"
        )
        self._write_json(
            self.outputs_dir / "signal.json",
            {
                "schema": "sharpedge.signal.v1",
                "symbol": "SPY",
                "trade_permission": {
                    "schema": "sharpedge.trade_permission.v1",
                    "trade_permission_score": 73,
                    "trade_gate": "pass",
                    "bias": "BULLISH",
                    "supporting_reasons": ["support one"],
                    "warning_reasons": ["warning one"],
                },
            },
        )
        self._write_json(
            self.outputs_dir / "robinhood_live_positions.json",
            {"positions": []},
        )
        self._write_json(
            self.outputs_dir / "robinhood_execution_handoff.json",
            {
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
            },
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _read_trace_events(self) -> list[dict]:
        trace_path = self.outputs_dir / TRACE_FILE_NAME
        if not trace_path.exists():
            return []
        return [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_server_describe_exposes_capabilities(self) -> None:
        server = ArgusMCPServer(context=self.context)
        payload = server.describe()
        self.assertIn("capabilities", payload)
        self.assertTrue(payload["capabilities"]["read_state"])
        self.assertFalse(payload["capabilities"]["execute_handoff"])
        self.assertTrue(payload["tools"])
        self.assertTrue(payload["resources"])

    def test_server_call_tool_delegates(self) -> None:
        server = ArgusMCPServer(context=self.context)
        payload = server.call_tool("sharpedge.get_latest_state")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["tool_name"], "sharpedge.get_latest_state")
        self.assertEqual(payload["state"]["symbol"], "SPY")
        self.assertEqual(payload["api_version"], 1)
        self.assertIn("request_id", payload)
        self.assertIn("duration_ms", payload)
        trace_events = self._read_trace_events()
        self.assertEqual(trace_events[-1]["target_name"], "sharpedge.get_latest_state")
        self.assertEqual(trace_events[-1]["wrapper"], "get_latest_state")

    def test_server_blocks_capability_gated_tool(self) -> None:
        caps = CapabilityProfile(prepare_handoff=False)
        server = ArgusMCPServer(context=self.context, capabilities=caps)
        payload = server.call_tool(
            "sharpedge.prepare_broker_handoff",
            {"operator_approved": True},
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_code"], "capability_denied")

    def test_server_reads_resources(self) -> None:
        server = ArgusMCPServer(context=self.context)
        state_payload = server.read_resource("sharpedge://state/latest")
        self.assertEqual(state_payload["status"], "ok")
        self.assertEqual(state_payload["contents"]["symbol"], "SPY")
        self.assertIn("trace_artifact_path", state_payload)
        handoff_payload = server.read_resource("sharpedge://handoff/latest")
        self.assertEqual(handoff_payload["status"], "ok")
        self.assertEqual(
            handoff_payload["contents"]["schema"],
            "sharpedge.robinhood_execution_handoff.v1",
        )
        trace_events = self._read_trace_events()
        self.assertEqual(trace_events[-1]["target_name"], "sharpedge://handoff/latest")
        self.assertEqual(trace_events[-1]["target_kind"], "resource")

    def test_server_returns_unknown_tool_error(self) -> None:
        server = ArgusMCPServer(context=self.context)
        payload = server.call_tool("sharpedge.nope")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_code"], "unknown_tool")


if __name__ == "__main__":
    unittest.main()
