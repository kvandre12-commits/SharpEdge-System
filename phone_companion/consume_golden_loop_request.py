"""Tiny Golden Loop request consumer.

Purpose:
- read ``golden_loop_request.json``
- validate required fields
- extract ``request_id``
- write a trace artifact proving the contract was consumed

Non-goals:
- no Brave launch
- no view-model creation
- no observation writing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

REQUIRED_FIELDS = [
    "request_id",
    "intent_type",
    "domain",
    "artifact_inputs",
]
DEFAULT_REQUEST_PATH = Path("phone_companion/requests/golden_loop_request.json")
DEFAULT_TRACE_PATH = Path("phone_companion/requests/golden_loop_request_trace.json")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("request payload must be a JSON object")
    return data


def _validate_required_fields(payload: dict) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    if not isinstance(payload["request_id"], str) or not payload["request_id"].strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(payload["intent_type"], str) or not payload["intent_type"].strip():
        raise ValueError("intent_type must be a non-empty string")
    if not isinstance(payload["domain"], str) or not payload["domain"].strip():
        raise ValueError("domain must be a non-empty string")
    if not isinstance(payload["artifact_inputs"], list) or not payload["artifact_inputs"]:
        raise ValueError("artifact_inputs must be a non-empty list")



def consume_request(request_path: Path = DEFAULT_REQUEST_PATH, trace_path: Path = DEFAULT_TRACE_PATH) -> dict:
    payload = _load_json(request_path)
    _validate_required_fields(payload)

    trace = {
        "trace_type": "phone_companion_request_consumed",
        "status": "consumed",
        "consumer": "phone_companion/consume_golden_loop_request.py",
        "source_request_path": str(request_path),
        "request_id": payload["request_id"],
        "validated_required_fields": REQUIRED_FIELDS,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
    }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
        handle.write("\n")

    return trace



def main(argv: list[str]) -> int:
    request_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_REQUEST_PATH
    trace_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_TRACE_PATH

    trace = consume_request(request_path=request_path, trace_path=trace_path)
    print(json.dumps(trace, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
