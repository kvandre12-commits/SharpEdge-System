"""Tiny Golden Loop prelaunch trace emitter.

Purpose:
- read ``golden_loop_view_model.json``
- preserve ``request_id`` exactly
- preserve ``view_id`` exactly
- write ``prelaunch_trace.json``

Non-goals:
- no DroidPuppy
- no Brave launch
- no Android interaction
- no observation writing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

DEFAULT_VIEW_MODEL_PATH = Path("phone_companion/views/trading/golden_loop_view_model.json")
DEFAULT_TRACE_PATH = Path("phone_companion/launchers/prelaunch_trace.json")



def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("view-model payload must be a JSON object")
    return payload



def _validate_view_model(view_model: dict) -> None:
    required_fields = ["view_id", "request_id", "data"]
    missing = [field for field in required_fields if field not in view_model]
    if missing:
        raise ValueError(f"missing required view-model fields: {', '.join(missing)}")

    if not isinstance(view_model["request_id"], str) or not view_model["request_id"].strip():
        raise ValueError("view-model request_id must be a non-empty string")
    if not isinstance(view_model["view_id"], str) or not view_model["view_id"].strip():
        raise ValueError("view-model view_id must be a non-empty string")
    if not isinstance(view_model["data"], dict):
        raise ValueError("view-model data must be a JSON object")
    if "url" not in view_model["data"]:
        raise ValueError("view-model data.url is required")
    if "preferred_channel" not in view_model["data"]:
        raise ValueError("view-model data.preferred_channel is required")
    if "artifact_inputs" not in view_model["data"]:
        raise ValueError("view-model data.artifact_inputs is required")
    if "signal_summary" not in view_model["data"]:
        raise ValueError("view-model data.signal_summary is required")



def _validate_trace(trace: dict, view_model: dict) -> None:
    if trace["request_id"] != view_model["request_id"]:
        raise ValueError("trace request_id must exactly match the view-model request_id")
    if trace["view_id"] != view_model["view_id"]:
        raise ValueError("trace view_id must exactly match the view-model view_id")



def emit_prelaunch_trace(
    view_model_path: Path = DEFAULT_VIEW_MODEL_PATH,
    trace_path: Path = DEFAULT_TRACE_PATH,
) -> dict:
    view_model = _load_json(view_model_path)
    _validate_view_model(view_model)

    trace = {
        "trace_type": "phone_companion_prelaunch_trace",
        "status": "prepared",
        "request_id": view_model["request_id"],
        "view_id": view_model["view_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intended_url": view_model["data"]["url"],
        "intended_channel": view_model["data"]["preferred_channel"],
        "artifact_inputs": view_model["data"]["artifact_inputs"],
        "signal_summary": view_model["data"]["signal_summary"],
    }

    _validate_trace(trace, view_model)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
        handle.write("\n")

    return trace



def main(argv: list[str]) -> int:
    view_model_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_VIEW_MODEL_PATH
    trace_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_TRACE_PATH

    trace = emit_prelaunch_trace(view_model_path=view_model_path, trace_path=trace_path)
    print(json.dumps(trace, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
