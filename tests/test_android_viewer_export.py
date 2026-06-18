from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion"))

from export_signal_to_android_viewer import export_signal  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_export_signal_to_android_viewer_copies_trade_gate_contract(tmp_path):
    signal_path = tmp_path / "outputs/signal.json"
    android_root = tmp_path / "SharpEdge-Android"
    proof_path = tmp_path / "phone_companion/views/trading/android_viewer_export.json"
    signal = {
        "schema": "sharpedge.signal.v1",
        "ts": "2026-06-18T15:40:58",
        "symbol": "SPY",
        "spot": 746.94,
        "trade_permission": {
            "trade_gate": "CAUTION",
            "trade_permission_score": 67,
            "bias": "NEUTRAL",
            "supporting_reasons": ["location: at VWAP"],
            "warning_reasons": ["volume: thin"],
            "scores": {},
        },
    }
    _write_json(signal_path, signal)

    proof = export_signal(signal_path, android_root, proof_path)

    asset = android_root / "app/src/main/assets/sample_signal.json"
    contract = android_root / "app_contracts/sharpedge.signal.v1.sample.json"
    assert proof["status"] == "exported"
    assert proof["trade_permission"] == {
        "trade_gate": "CAUTION",
        "trade_permission_score": 67,
        "bias": "NEUTRAL",
    }
    assert json.loads(asset.read_text()) == signal
    assert json.loads(contract.read_text()) == signal
    assert json.loads(proof_path.read_text())["status"] == "exported"
