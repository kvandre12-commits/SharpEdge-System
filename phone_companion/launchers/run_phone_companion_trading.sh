#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

REPO_ROOT = Path(os.environ["REPO_ROOT"])
PRELAUNCH_TRACE_PATH = REPO_ROOT / "phone_companion/launchers/prelaunch_trace.json"
LAUNCH_ATTEMPT_PATH = REPO_ROOT / "phone_companion/launchers/launch_attempt.json"
LAUNCH_RESULT_PATH = REPO_ROOT / "phone_companion/launchers/launch_result.json"
COCKPIT_DIR = REPO_ROOT / "cockpit"
COCKPIT_SERVER_LOG = Path(os.environ.get("HOME", str(REPO_ROOT))) / ".cache" / "phone_companion_cockpit_server.log"
CHANNEL_PACKAGES = {
    "brave": "com.brave.browser",
    "chrome": "com.android.chrome",
    "system": "",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")



def _load_prelaunch_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prelaunch trace must be a JSON object")
    return payload



def _validate_trace(trace: dict) -> None:
    required_fields = [
        "request_id",
        "view_id",
        "intended_url",
        "intended_channel",
        "artifact_inputs",
    ]
    missing = [field for field in required_fields if field not in trace]
    if missing:
        raise ValueError(f"missing required prelaunch trace fields: {', '.join(missing)}")
    if not isinstance(trace["request_id"], str) or not trace["request_id"].strip():
        raise ValueError("prelaunch trace request_id must be a non-empty string")
    if not isinstance(trace["view_id"], str) or not trace["view_id"].strip():
        raise ValueError("prelaunch trace view_id must be a non-empty string")
    if not isinstance(trace["intended_url"], str) or not trace["intended_url"].strip():
        raise ValueError("prelaunch trace intended_url must be a non-empty string")
    if trace["intended_channel"] not in CHANNEL_PACKAGES:
        raise ValueError("prelaunch trace intended_channel must map to a known launcher package")
    if not isinstance(trace["artifact_inputs"], list) or not trace["artifact_inputs"]:
        raise ValueError("prelaunch trace artifact_inputs must be a non-empty list")



def _verify_artifact_inputs(trace: dict) -> None:
    missing_paths = [
        artifact_path
        for artifact_path in trace["artifact_inputs"]
        if not (REPO_ROOT / artifact_path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"missing required upstream artifacts: {', '.join(missing_paths)}"
        )



def _build_command(trace: dict) -> tuple[list[str], str]:
    package_name = CHANNEL_PACKAGES[trace["intended_channel"]]
    command = [
        "am",
        "start",
        "-a",
        "android.intent.action.VIEW",
        "-d",
        trace["intended_url"],
    ]
    if package_name:
        command.extend(["-p", package_name])
    command_string = " ".join(command)
    return command, command_string



def _port_open(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False



def _ensure_local_cockpit_server(trace: dict) -> None:
    parsed = urlparse(trace["intended_url"])
    if parsed.hostname != "127.0.0.1" or parsed.path != "/cockpit.html" or not parsed.port:
        return
    if _port_open(parsed.hostname, parsed.port):
        return

    COCKPIT_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COCKPIT_SERVER_LOG.open("ab") as handle:
        subprocess.Popen(
            ["python3", "-m", "http.server", str(parsed.port)],
            cwd=str(COCKPIT_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(10):
        if _port_open(parsed.hostname, parsed.port):
            return
        time.sleep(0.2)
    raise RuntimeError(f"local cockpit server did not start on {parsed.hostname}:{parsed.port}")



def main() -> int:
    trace = _load_prelaunch_trace(PRELAUNCH_TRACE_PATH)
    _validate_trace(trace)
    _verify_artifact_inputs(trace)
    _ensure_local_cockpit_server(trace)
    command, command_string = _build_command(trace)

    started_at = _timestamp()
    attempt = {
        "artifact_type": "phone_companion_launch_attempt",
        "status": "starting",
        "request_id": trace["request_id"],
        "view_id": trace["view_id"],
        "started_at": started_at,
        "intended_url": trace["intended_url"],
        "intended_channel": trace["intended_channel"],
        "artifact_inputs": trace["artifact_inputs"],
        "signal_summary": trace.get("signal_summary", {}),
        "command": command,
        "command_string": command_string,
    }
    _write_json(LAUNCH_ATTEMPT_PATH, attempt)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    ended_at = _timestamp()
    result = {
        "artifact_type": "phone_companion_launch_result",
        "status": "accepted_by_android" if completed.returncode == 0 else "shell_failed",
        "request_id": trace["request_id"],
        "view_id": trace["view_id"],
        "started_at": started_at,
        "ended_at": ended_at,
        "intended_url": trace["intended_url"],
        "intended_channel": trace["intended_channel"],
        "artifact_inputs": trace["artifact_inputs"],
        "signal_summary": trace.get("signal_summary", {}),
        "command": command,
        "command_string": command_string,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _write_json(LAUNCH_RESULT_PATH, result)
    print(json.dumps(result, indent=2))
    return 0 if completed.returncode == 0 else completed.returncode



if __name__ == "__main__":
    raise SystemExit(main())
PY
