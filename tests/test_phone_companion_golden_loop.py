from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion"))

from build_golden_loop_view_model import build_view_model  # noqa: E402
from consume_golden_loop_request import consume_request  # noqa: E402
from emit_golden_loop_prelaunch_trace import emit_prelaunch_trace  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_contract_reaches_companion_view_and_preserves_trade_gate(tmp_path):
    request_path = tmp_path / "requests/golden_loop_request.json"
    trace_path = tmp_path / "requests/golden_loop_request_trace.json"
    view_model_path = tmp_path / "views/trading/golden_loop_view_model.json"
    prelaunch_path = tmp_path / "launchers/prelaunch_trace.json"
    signal_path = tmp_path / "outputs/signal.json"

    request = {
        "request_id": "req-trading-golden-test",
        "intent_type": "open_trading_dashboard",
        "domain": "trading",
        "artifact_inputs": [
            "cockpit/cockpit.html",
            "outputs/signal.json",
            "outputs/approval_decision.json",
            "outputs/workflow_state.json",
        ],
        "preferred_view": "cockpit",
        "preferred_channel": "brave",
        "requires_confirmation": False,
    }
    signal = {
        "schema": "sharpedge.signal.v1",
        "ts": "2026-06-18T15:00:00",
        "symbol": "SPY",
        "spot": 746.43,
        "trade_permission": {
            "trade_gate": "CAUTION",
            "trade_permission_score": 63,
            "bias": "NEUTRAL",
            "bias_strength": 0.12,
            "supporting_reasons": ["trend: above VWAP"],
            "warning_reasons": ["pressure: no clear trapped side"],
        },
    }
    _write_json(request_path, request)
    _write_json(signal_path, signal)

    consumed = consume_request(request_path, trace_path)
    view_model = build_view_model(request_path, view_model_path, signal_path)
    prelaunch = emit_prelaunch_trace(view_model_path, prelaunch_path)

    assert consumed["request_id"] == request["request_id"]
    assert view_model["request_id"] == request["request_id"]
    assert prelaunch["request_id"] == request["request_id"]
    assert view_model["data"]["signal_summary"]["status"] == "ready"
    assert view_model["data"]["signal_summary"]["trade_permission"] == {
        "gate": "CAUTION",
        "score": 63,
        "bias": "NEUTRAL",
        "bias_strength": 0.12,
        "supporting_reasons": ["trend: above VWAP"],
        "warning_reasons": ["pressure: no clear trapped side"],
    }
    assert prelaunch["signal_summary"] == view_model["data"]["signal_summary"]
