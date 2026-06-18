"""Tiny Golden Loop observation emitter.

Purpose:
- read ``launch_result.json``
- preserve ``request_id`` exactly
- write ``golden_loop_latest.json``

Non-goals:
- no DroidPuppy launch
- no Brave launch
- no Android interaction
- no replay logic
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

DEFAULT_LAUNCH_RESULT_PATH = Path("phone_companion/launchers/launch_result.json")
DEFAULT_OBSERVATION_PATH = Path("phone_companion/observations/golden_loop_latest.json")



def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("launch result payload must be a JSON object")
    return payload



def _validate_launch_result(payload: dict) -> None:
    required_fields = [
        "request_id",
        "status",
        "started_at",
        "ended_at",
        "intended_url",
        "intended_channel",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"missing required launch result fields: {', '.join(missing)}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"].strip():
        raise ValueError("launch result request_id must be a non-empty string")



def _build_observation_id(request_id: str) -> str:
    suffix = request_id.removeprefix("req-")
    return f"obs-{suffix}" if suffix else "obs-golden-loop"



def _build_status(launch_status: str) -> str:
    return "success" if launch_status == "accepted_by_android" else "failed"



def _validate_observation(observation: dict, expected_request_id: str) -> None:
    required_fields = [
        "observation_id",
        "request_id",
        "status",
        "action_type",
        "target",
        "started_at",
        "ended_at",
    ]
    missing = [field for field in required_fields if field not in observation]
    if missing:
        raise ValueError(f"missing required observation fields: {', '.join(missing)}")
    if observation["request_id"] != expected_request_id:
        raise ValueError("observation request_id must exactly match launch result request_id")



def emit_observation(
    launch_result_path: Path = DEFAULT_LAUNCH_RESULT_PATH,
    observation_path: Path = DEFAULT_OBSERVATION_PATH,
) -> dict:
    launch_result = _load_json(launch_result_path)
    _validate_launch_result(launch_result)

    observation = {
        "observation_id": _build_observation_id(launch_result["request_id"]),
        "request_id": launch_result["request_id"],
        "status": _build_status(launch_result["status"]),
        "action_type": "open_dashboard",
        "target": f"{launch_result['intended_channel']}:{launch_result['intended_url']}",
        "started_at": launch_result["started_at"],
        "ended_at": launch_result["ended_at"],
        "result_summary": (
            "Android accepted the trading dashboard launch command."
            if launch_result["status"] == "accepted_by_android"
            else "Android did not accept the trading dashboard launch command."
        ),
        "artifacts_created": [
            str(launch_result_path),
            str(observation_path),
        ],
        "fallback_used": False,
    }

    _validate_observation(observation, expected_request_id=launch_result["request_id"])

    observation_path.parent.mkdir(parents=True, exist_ok=True)
    with observation_path.open("w", encoding="utf-8") as handle:
        json.dump(observation, handle, indent=2)
        handle.write("\n")

    return observation



def main(argv: list[str]) -> int:
    launch_result_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LAUNCH_RESULT_PATH
    observation_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OBSERVATION_PATH

    observation = emit_observation(
        launch_result_path=launch_result_path,
        observation_path=observation_path,
    )
    print(json.dumps(observation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
