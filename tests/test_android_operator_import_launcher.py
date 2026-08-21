from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion/launchers"))

import share_operator_packet_to_android as launcher  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_launch_operator_packet_to_android_dry_run_writes_result(tmp_path, monkeypatch):
    signal_path = tmp_path / "outputs/signal.json"
    android_root = tmp_path / "SharpEdge-Android"
    proof_path = (
        tmp_path / "phone_companion/views/trading/android_operator_packet_export.json"
    )
    live_import_path = (
        tmp_path
        / "phone_companion/views/trading/sharpedge_android_operator_import.json"
    )
    attempt_path = (
        tmp_path / "phone_companion/launchers/android_operator_import_attempt.json"
    )
    result_path = (
        tmp_path / "phone_companion/launchers/android_operator_import_result.json"
    )
    output_dir = tmp_path / "outputs"

    _write_json(
        signal_path,
        {
            "schema": "sharpedge.signal.v1",
            "ts": "2026-06-25T21:30:00",
            "symbol": "SPY",
            "spot": 734.3,
            "edge_token_position": {
                "suggested_action": "hold",
                "position_state": "open",
                "contracts_held": 1,
            },
            "trade_permission": {
                "trade_gate": "PERMIT",
                "trade_permission_score": 97,
                "bias": "CALLS",
                "supporting_reasons": ["fresh data"],
                "warning_reasons": [],
                "scores": {},
            },
        },
    )
    _write_text(
        tmp_path / "cockpit/cockpit.html",
        '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="2"></head>'
        "<body><h1>SharpEdge Cockpit</h1></body></html>",
    )
    _write_text(
        tmp_path / "cockpit/cockpit_chart.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>",
    )
    _write_json(output_dir / "operator_brief.json", {"headline": "Stand down."})
    _write_json(output_dir / "workflow_state.json", {"state": {"readiness": "blocked"}})
    _write_json(
        output_dir / "approval_decision.json",
        {"decision": "hold", "trade_allowed": False},
    )
    _write_json(
        output_dir / "operator_watchlist.json", {"active_count": 0, "items": []}
    )
    _write_json(
        output_dir / "trade_journal_hints.json",
        {"sample_state": {"closed_trades": 3}, "top_patterns": []},
    )
    _write_json(
        output_dir / "robinhood_beta_execution.json",
        {
            "beta_stage": "position_hold",
            "edge_token_position": {"contracts_held": 1},
            "order_preview": {"token_action": "hold", "position_intent": "hold"},
            "robinhood_beta_handoff": {"bridge_status": {"status": "disabled"}},
        },
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "ATTEMPT_PATH", attempt_path)
    monkeypatch.setattr(launcher, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        launcher, "app_installed", lambda package_name=launcher.ANDROID_PACKAGE: True
    )

    result = launcher.launch_operator_packet_to_android(
        signal_path=signal_path,
        android_root=android_root,
        proof_path=proof_path,
        live_import_path=live_import_path,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["packet_schema"] == "sharpedge.operator_packet.v1"
    assert result["product"] == "SharpEdge Robinhood"
    assert live_import_path.exists()
    assert json.loads(result_path.read_text())["status"] == "dry_run"


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
