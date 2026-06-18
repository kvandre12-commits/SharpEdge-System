"""Tiny Golden Loop trading view-model builder.

Purpose:
- read ``golden_loop_request.json``
- validate the minimum request fields needed for view-model packaging
- extract ``request_id``
- write ``golden_loop_view_model.json``

Non-goals:
- no DroidPuppy launch
- no Brave launch
- no observation writing
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REQUIRED_REQUEST_FIELDS = [
    "request_id",
    "intent_type",
    "domain",
    "artifact_inputs",
]
DEFAULT_REQUEST_PATH = Path("phone_companion/requests/golden_loop_request.json")
DEFAULT_VIEW_MODEL_PATH = Path("phone_companion/views/trading/golden_loop_view_model.json")
COCKPIT_URL = "http://127.0.0.1:8777/cockpit.html"
SIGNAL_PATH = Path("outputs/signal.json")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def _validate_request(payload: dict) -> None:
    missing = [field for field in REQUIRED_REQUEST_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required request fields: {', '.join(missing)}")

    if payload["intent_type"] != "open_trading_dashboard":
        raise ValueError("intent_type must be 'open_trading_dashboard'")
    if payload["domain"] != "trading":
        raise ValueError("domain must be 'trading'")
    if not isinstance(payload["request_id"], str) or not payload["request_id"].strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(payload["artifact_inputs"], list) or not payload["artifact_inputs"]:
        raise ValueError("artifact_inputs must be a non-empty list")



def _build_view_id(request_id: str) -> str:
    suffix = request_id.removeprefix("req-")
    return f"view-{suffix}" if suffix else "view-trading-golden"


def _load_signal_summary(signal_path: Path = SIGNAL_PATH) -> dict:
    if not signal_path.exists():
        return {
            "status": "missing",
            "path": str(signal_path),
            "warning": "signal.json not found; dashboard can still open",
        }
    signal = _load_json(signal_path)
    trade_permission = signal.get("trade_permission") or {}
    return {
        "status": "ready" if trade_permission else "missing_trade_permission",
        "path": str(signal_path),
        "signal_ts": signal.get("ts"),
        "symbol": signal.get("symbol"),
        "spot": signal.get("spot"),
        "trade_permission": {
            "gate": trade_permission.get("trade_gate"),
            "score": trade_permission.get("trade_permission_score"),
            "bias": trade_permission.get("bias"),
            "bias_strength": trade_permission.get("bias_strength"),
            "supporting_reasons": trade_permission.get("supporting_reasons", []),
            "warning_reasons": trade_permission.get("warning_reasons", []),
        },
    }



def _validate_view_model(view_model: dict, expected_request_id: str) -> None:
    required_fields = [
        "view_id",
        "request_id",
        "domain",
        "view_type",
        "headline",
        "status",
        "data",
    ]
    missing = [field for field in required_fields if field not in view_model]
    if missing:
        raise ValueError(f"missing required view-model fields: {', '.join(missing)}")

    data = view_model["data"]
    if not isinstance(view_model["request_id"], str) or not view_model["request_id"].strip():
        raise ValueError("view-model request_id must be a non-empty string")
    if view_model["request_id"] != expected_request_id:
        raise ValueError("view-model request_id must exactly match request request_id")
    if view_model["domain"] != "trading":
        raise ValueError("view-model domain must be 'trading'")
    if view_model["view_type"] != "cockpit":
        raise ValueError("view-model view_type must be 'cockpit'")
    if view_model["status"] != "ready":
        raise ValueError("view-model status must be 'ready'")
    if not isinstance(data, dict):
        raise ValueError("view-model data must be a JSON object")
    if data.get("url") != COCKPIT_URL:
        raise ValueError(f"view-model data.url must be '{COCKPIT_URL}'")
    if not isinstance(data.get("artifact_inputs"), list) or not data["artifact_inputs"]:
        raise ValueError("view-model data.artifact_inputs must be a non-empty list")
    if "preferred_channel" not in data:
        raise ValueError("view-model data.preferred_channel is required")
    signal_summary = data.get("signal_summary")
    if not isinstance(signal_summary, dict):
        raise ValueError("view-model data.signal_summary must be a JSON object")
    trade_permission = signal_summary.get("trade_permission", {})
    if signal_summary.get("status") == "ready" and not trade_permission.get("gate"):
        raise ValueError("ready signal_summary must include trade_permission.gate")



def build_view_model(
    request_path: Path = DEFAULT_REQUEST_PATH,
    view_model_path: Path = DEFAULT_VIEW_MODEL_PATH,
    signal_path: Path = SIGNAL_PATH,
) -> dict:
    request = _load_json(request_path)
    _validate_request(request)

    view_model = {
        "view_id": _build_view_id(request["request_id"]),
        "request_id": request["request_id"],
        "domain": "trading",
        "view_type": "cockpit",
        "headline": "Open trading cockpit",
        "status": "ready",
        "data": {
            "url": COCKPIT_URL,
            "artifact_inputs": request["artifact_inputs"],
            "preferred_channel": request.get("preferred_channel", "brave"),
            "signal_summary": _load_signal_summary(signal_path),
        },
    }

    _validate_view_model(view_model, expected_request_id=request["request_id"])

    view_model_path.parent.mkdir(parents=True, exist_ok=True)
    with view_model_path.open("w", encoding="utf-8") as handle:
        json.dump(view_model, handle, indent=2)
        handle.write("\n")

    return view_model



def main(argv: list[str]) -> int:
    request_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_REQUEST_PATH
    view_model_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_VIEW_MODEL_PATH
    signal_path = Path(argv[3]) if len(argv) > 3 else SIGNAL_PATH

    view_model = build_view_model(
        request_path=request_path,
        view_model_path=view_model_path,
        signal_path=signal_path,
    )
    print(json.dumps(view_model, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
