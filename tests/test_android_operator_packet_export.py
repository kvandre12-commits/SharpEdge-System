from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion"))

from export_operator_packet_to_android import export_operator_packet
from export_signal_to_android_viewer import ANDROID_VIEWER_BUNDLE_KEY


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_export_operator_packet_writes_live_import(tmp_path, monkeypatch):
    signal_path = tmp_path / "outputs/signal.json"
    android_root = tmp_path / "SharpEdge-Android"
    proof_path = (
        tmp_path / "phone_companion/views/trading/android_operator_packet_export.json"
    )
    live_import_path = (
        tmp_path
        / "phone_companion/views/trading/sharpedge_android_operator_import.json"
    )
    output_dir = tmp_path / "outputs"
    signal = {
        "schema": "sharpedge.signal.v1",
        "ts": "2026-06-25T21:30:00",
        "symbol": "SPY",
        "spot": 734.3,
        "edge_token_position": {
            "suggested_action": "hold",
            "position_state": "open",
            "contracts_held": 1,
            "action_reason": "edge token is still active; keep the single-contract position on.",
            "recommended_actions": ["hold"],
        },
        "trade_permission": {
            "trade_gate": "PERMIT",
            "trade_permission_score": 97,
            "bias": "CALLS",
            "supporting_reasons": ["fresh data"],
            "warning_reasons": [],
            "scores": {},
        },
    }
    operator_brief = {
        "schema_version": "operator_brief.v1",
        "headline": "Stand down.",
        "latest_execution_audit": {
            "available": True,
            "connector_status": "drafted",
            "fill_status": "not_submitted",
            "summary": "Draft prepared and awaiting confirmation.",
        },
        "artifacts": {
            "connector_audit": "outputs/chatgpt_robinhood_connector_audit.json",
            "connector_audit_log": "outputs/robinhood_connector_audit_log.jsonl",
        },
    }
    workflow_state = {
        "state": {"readiness": "blocked", "operator_action": "stand_down"}
    }
    approval_decision = {
        "decision": "hold",
        "trade_allowed": False,
        "broker_order_allowed": False,
    }
    operator_watchlist = {"active_count": 1, "items": [{"symbol": "SPY"}]}
    trade_journal_hints = {
        "sample_state": {"closed_trades": 3},
        "top_patterns": [{"pattern_id": "x"}],
    }
    robinhood_beta_execution = {
        "beta_stage": "position_hold",
        "approval_required": True,
        "edge_token_position": {
            "contracts_held": 1,
        },
        "order_preview": {
            "token_action": "hold",
            "position_intent": "hold",
        },
        "robinhood_beta_handoff": {"bridge_status": {"status": "disabled"}},
    }

    _write_json(signal_path, signal)
    _write_text(
        tmp_path / "cockpit/cockpit.html",
        '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="2"></head>'
        "<body><h1>Imported cockpit</h1></body></html>",
    )
    _write_text(
        tmp_path / "cockpit/cockpit_chart.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'><circle cx='5' cy='5' r='5'/></svg>",
    )
    _write_text(
        tmp_path / "cockpit/cockpit_weekly_context.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'><text>weekly</text></svg>",
    )
    _write_text(
        tmp_path / "cockpit/cockpit_monthly_context.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'><text>monthly</text></svg>",
    )
    _write_json(output_dir / "operator_brief.json", operator_brief)
    _write_json(output_dir / "workflow_state.json", workflow_state)
    _write_json(output_dir / "approval_decision.json", approval_decision)
    _write_json(output_dir / "operator_watchlist.json", operator_watchlist)
    _write_json(output_dir / "trade_journal_hints.json", trade_journal_hints)
    _write_json(output_dir / "robinhood_beta_execution.json", robinhood_beta_execution)

    monkeypatch.chdir(tmp_path)
    proof = export_operator_packet(
        signal_path, android_root, proof_path, live_import_path
    )

    packet = json.loads(live_import_path.read_text())
    assert proof["status"] == "exported"
    assert packet["schema"] == "sharpedge.operator_packet.v1"
    assert packet["product"] == "SharpEdge Robinhood"
    assert packet["approval_decision"]["decision"] == "hold"
    assert packet["operator_watchlist"]["active_count"] == 1
    assert packet["app_sections"] == [
        "cockpit",
        "approvals",
        "agent_status",
        "watchlists",
        "trade_journal",
        "robinhood_actions",
    ]
    assert list(packet["status_summary"]) == [
        "trade_gate",
        "trade_permission_score",
        "workflow_readiness",
        "approval_decision",
        "trade_allowed",
        "watchlist_active_count",
        "journal_closed_trades",
        "bridge_status",
        "connector_audit_available",
        "connector_status",
        "connector_fill_status",
    ]
    assert packet["status_summary"]["bridge_status"] == "disabled"
    assert packet["status_summary"]["connector_status"] == "drafted"
    assert packet["execution_audit"]["available"] is True
    assert packet["robinhood_beta_execution"]["order_preview"]["token_action"] == "hold"
    assert "execution_audit" not in packet["app_sections"]
    assert "connector_audit" not in packet["artifacts"]
    assert ANDROID_VIEWER_BUNDLE_KEY in packet
    assert packet[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_html"].startswith(
        "<!DOCTYPE html>"
    )
    assert (
        'http-equiv="refresh"'
        not in packet[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_html"].lower()
    )
    assert packet[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_weekly_context_svg"].startswith(
        "<svg"
    )
    assert packet[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_monthly_context_svg"].startswith(
        "<svg"
    )
    assert proof["android_viewer_bundle_included"] is True
    assert proof["web_viewer_refresh"]["status"] == "refresh_ready"
    assert proof["web_viewer_refresh"]["cockpit_refresh_seconds"] == 2
