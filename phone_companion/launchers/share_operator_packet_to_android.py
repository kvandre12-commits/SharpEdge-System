from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phone_companion"))

from export_operator_packet_to_android import (  # noqa: E402
    DEFAULT_LIVE_IMPORT_PATH,
    DEFAULT_PROOF_PATH,
    DEFAULT_SIGNAL_PATH,
    export_operator_packet,
)
from export_signal_to_android_viewer import DEFAULT_ANDROID_ROOT  # noqa: E402

ANDROID_PACKAGE = "com.sharpedge.cockpit"
ANDROID_COMPONENT = f"{ANDROID_PACKAGE}/.MainActivity"
ATTEMPT_PATH = ROOT / "phone_companion/launchers/android_operator_import_attempt.json"
RESULT_PATH = ROOT / "phone_companion/launchers/android_operator_import_result.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_minified_payload(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _package_query_commands(package_name: str) -> list[list[str]]:
    return [
        ["pm", "list", "packages", package_name],
        ["cmd", "package", "list", "packages", package_name],
        ["adb", "shell", "pm", "list", "packages", package_name],
    ]


def app_installed(package_name: str = ANDROID_PACKAGE) -> bool:
    for command in _package_query_commands(package_name):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if completed.returncode == 0 and package_name in completed.stdout:
            return True
    return False


def build_share_command(payload_json_text: str) -> list[str]:
    return [
        "am",
        "start",
        "-S",
        "-n",
        ANDROID_COMPONENT,
        "-a",
        "android.intent.action.SEND",
        "-t",
        "text/plain",
        "--es",
        "android.intent.extra.TEXT",
        payload_json_text,
    ]


def launch_operator_packet_to_android(
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    android_root: Path = DEFAULT_ANDROID_ROOT,
    proof_path: Path = DEFAULT_PROOF_PATH,
    live_import_path: Path = DEFAULT_LIVE_IMPORT_PATH,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    export_proof = export_operator_packet(
        signal_path=signal_path,
        android_root=android_root,
        proof_path=proof_path,
        live_import_path=live_import_path,
    )
    payload_json_text = _load_minified_payload(live_import_path)
    installed = app_installed()
    command = build_share_command(payload_json_text)

    attempt = {
        "artifact_type": "sharpedge_android_operator_import_attempt",
        "status": "starting",
        "started_at": _timestamp(),
        "android_package": ANDROID_PACKAGE,
        "android_component": ANDROID_COMPONENT,
        "signal_path": str(signal_path),
        "live_import_path": str(live_import_path),
        "export_proof_path": str(proof_path),
        "export_status": export_proof.get("status"),
        "packet_schema": export_proof.get("schema"),
        "product": export_proof.get("product"),
        "app_installed": installed,
        "payload_bytes": len(payload_json_text.encode("utf-8")),
        "command_preview": command[:-1] + [f"<json:{len(payload_json_text)} chars>"],
        "dry_run": dry_run,
    }
    _write_json(ATTEMPT_PATH, attempt)

    if not installed:
        result = {
            **attempt,
            "artifact_type": "sharpedge_android_operator_import_result",
            "status": "app_not_installed",
            "ended_at": _timestamp(),
            "exit_code": None,
            "stdout": "",
            "stderr": (
                f"Android package {ANDROID_PACKAGE} is not installed on this device yet."
            ),
        }
        _write_json(RESULT_PATH, result)
        return result

    if dry_run:
        result = {
            **attempt,
            "artifact_type": "sharpedge_android_operator_import_result",
            "status": "dry_run",
            "ended_at": _timestamp(),
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
        _write_json(RESULT_PATH, result)
        return result

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result = {
        **attempt,
        "artifact_type": "sharpedge_android_operator_import_result",
        "status": "accepted_by_android"
        if completed.returncode == 0
        else "shell_failed",
        "ended_at": _timestamp(),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _write_json(RESULT_PATH, result)
    return result


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    result = launch_operator_packet_to_android(dry_run=dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"accepted_by_android", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
