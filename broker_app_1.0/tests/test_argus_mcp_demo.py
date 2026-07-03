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

run_execution_card_demo = importlib.import_module("mcp.demo").run_execution_card_demo
WrapperContext = importlib.import_module("runtime.argus_mcp_wrapper").WrapperContext


class ArgusMCPDemoTestCase(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_run_execution_card_demo_writes_artifact(self) -> None:
        payload = run_execution_card_demo(context=self.context)
        self.assertEqual(payload["api_version"], 1)
        self.assertEqual(payload["execution_card_response"]["status"], "ok")
        self.assertEqual(payload["explanation_response"]["status"], "ok")
        artifact_path = Path(payload["artifact_path"])
        self.assertTrue(artifact_path.exists())
        written = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(written["flow"]["tool"], "sharpedge.get_execution_card")
        self.assertEqual(
            written["execution_card_response"]["execution_card"][
                "trade_permission_score"
            ],
            73,
        )


if __name__ == "__main__":
    unittest.main()
