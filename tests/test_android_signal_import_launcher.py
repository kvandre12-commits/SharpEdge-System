from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion/launchers"))

import share_signal_to_android_viewer as launcher


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_launch_signal_to_android_viewer_dry_run_writes_result(tmp_path, monkeypatch):
    signal_path = tmp_path / "outputs/signal.json"
    android_root = tmp_path / "SharpEdge-Android"
    proof_path = tmp_path / "phone_companion/views/trading/android_viewer_export.json"
    live_import_path = (
        tmp_path / "phone_companion/views/trading/sharpedge_android_live_import.json"
    )
    attempt_path = (
        tmp_path / "phone_companion/launchers/android_signal_import_attempt.json"
    )
    result_path = (
        tmp_path / "phone_companion/launchers/android_signal_import_result.json"
    )
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

    monkeypatch.setattr(launcher, "ATTEMPT_PATH", attempt_path)
    monkeypatch.setattr(launcher, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        launcher, "app_installed", lambda package_name=launcher.ANDROID_PACKAGE: True
    )

    result = launcher.launch_signal_to_android_viewer(
        signal_path=signal_path,
        android_root=android_root,
        proof_path=proof_path,
        live_import_path=live_import_path,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["app_installed"] is True
    assert result["android_component"] == "com.sharpedge.cockpit/.MainActivity"
    assert live_import_path.exists()
    assert json.loads(result_path.read_text())["status"] == "dry_run"


def test_build_share_command_targets_android_component():
    command = launcher.build_share_command('{"schema":"sharpedge.signal.v1"}')

    assert command[:6] == [
        "am",
        "start",
        "-S",
        "-n",
        "com.sharpedge.cockpit/.MainActivity",
        "-a",
    ]
    assert "android.intent.extra.TEXT" in command


def test_app_installed_falls_back_to_adb_shell(monkeypatch):
    calls = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        if command[:3] == ["adb", "shell", "pm"]:
            return subprocess.CompletedProcess(
                command, 0, "package:com.sharpedge.cockpit\n", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher.app_installed() is True
    assert calls[-1][:3] == ["adb", "shell", "pm"]
