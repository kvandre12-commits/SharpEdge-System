"""Export a unified SharpEdge operator packet for the Android app.

This packages the real operator-facing artifacts into one mobile import contract
so the Android app can act like the platform frontend instead of a signal-only
viewer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from export_signal_to_android_viewer import (
    ANDROID_VIEWER_BUNDLE_KEY,
    DEFAULT_ANDROID_ROOT,
    DEFAULT_SIGNAL_PATH,
    _load_json,
    _validate_signal,
    build_android_viewer_bundle,
    export_signal,
    validate_web_viewer_refresh,
)

DEFAULT_PROOF_PATH = Path(
    "phone_companion/views/trading/android_operator_packet_export.json"
)
DEFAULT_LIVE_IMPORT_PATH = Path(
    "phone_companion/views/trading/sharpedge_android_operator_import.json"
)
OUTPUT_DIR = Path("outputs")
OPERATOR_PACKET_SCHEMA = "sharpedge.operator_packet.v1"
OPERATOR_PACKET_PRODUCT = "SharpEdge Robinhood"
APP_SECTIONS = [
    "cockpit",
    "approvals",
    "agent_status",
    "watchlists",
    "trade_journal",
    "robinhood_actions",
]
ARTIFACT_PATHS = {
    "operator_brief": OUTPUT_DIR / "operator_brief.json",
    "workflow_state": OUTPUT_DIR / "workflow_state.json",
    "approval_decision": OUTPUT_DIR / "approval_decision.json",
    "operator_watchlist": OUTPUT_DIR / "operator_watchlist.json",
    "trade_journal_hints": OUTPUT_DIR / "trade_journal_hints.json",
    "robinhood_beta_execution": OUTPUT_DIR / "robinhood_beta_execution.json",
}
REQUIRED_PACKET_KEYS = [
    "operator_brief",
    "workflow_state",
    "approval_decision",
    "operator_watchlist",
    "trade_journal_hints",
    "robinhood_beta_execution",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_required_artifacts() -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    missing = []
    for key, path in ARTIFACT_PATHS.items():
        if not path.is_file():
            missing.append(str(path))
            continue
        artifacts[key] = _load_json(path)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"missing required operator artifacts: {missing_text}")
    return artifacts


def _status_summary(
    signal: dict,
    approval: dict,
    workflow: dict,
    beta: dict,
    watchlist: dict,
    journal: dict,
    execution_audit: dict,
) -> dict:
    trade_permission = signal.get("trade_permission") or {}
    return {
        "trade_gate": trade_permission.get("trade_gate"),
        "trade_permission_score": trade_permission.get("trade_permission_score"),
        "workflow_readiness": (workflow.get("state") or {}).get("readiness"),
        "approval_decision": approval.get("decision"),
        "trade_allowed": approval.get("trade_allowed"),
        "watchlist_active_count": watchlist.get("active_count"),
        "journal_closed_trades": (journal.get("sample_state") or {}).get(
            "closed_trades"
        ),
        "bridge_status": (beta.get("robinhood_beta_handoff") or {})
        .get("bridge_status", {})
        .get("status"),
        "connector_audit_available": execution_audit.get("available", False),
        "connector_status": execution_audit.get("connector_status"),
        "connector_fill_status": execution_audit.get("fill_status"),
    }


def build_operator_packet(signal_path: Path = DEFAULT_SIGNAL_PATH) -> dict:
    signal = _load_json(signal_path)
    _validate_signal(signal)
    artifacts = _load_required_artifacts()
    approval = artifacts["approval_decision"]
    workflow = artifacts["workflow_state"]
    beta = artifacts["robinhood_beta_execution"]
    watchlist = artifacts["operator_watchlist"]
    journal = artifacts["trade_journal_hints"]
    brief = artifacts["operator_brief"]
    execution_audit = brief.get("latest_execution_audit") or {"available": False}
    packet_artifacts = {
        "signal": str(signal_path),
        **{key: str(path) for key, path in ARTIFACT_PATHS.items()},
    }
    viewer_bundle = build_android_viewer_bundle(signal_path)

    packet = {
        "schema": OPERATOR_PACKET_SCHEMA,
        "created_at": _timestamp(),
        "product": OPERATOR_PACKET_PRODUCT,
        "symbol": signal.get("symbol"),
        "app_sections": list(APP_SECTIONS),
        "artifacts": packet_artifacts,
        "status_summary": _status_summary(
            signal, approval, workflow, beta, watchlist, journal, execution_audit
        ),
        "signal": signal,
        "operator_brief": brief,
        "workflow_state": workflow,
        "approval_decision": approval,
        "operator_watchlist": watchlist,
        "trade_journal_hints": journal,
        "robinhood_beta_execution": beta,
        "execution_audit": execution_audit,
    }
    if viewer_bundle:
        packet[ANDROID_VIEWER_BUNDLE_KEY] = viewer_bundle
    return packet


def export_operator_packet(
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    android_root: Path = DEFAULT_ANDROID_ROOT,
    proof_path: Path = DEFAULT_PROOF_PATH,
    live_import_path: Path = DEFAULT_LIVE_IMPORT_PATH,
) -> dict:
    web_viewer_refresh = validate_web_viewer_refresh(signal_path)
    signal_export = export_signal(signal_path, android_root)
    packet = build_operator_packet(signal_path)
    missing = [key for key in REQUIRED_PACKET_KEYS if key not in packet]
    if missing:
        raise ValueError(
            f"operator packet missing top-level keys: {', '.join(missing)}"
        )

    live_import_path.parent.mkdir(parents=True, exist_ok=True)
    live_import_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")

    approval = packet["approval_decision"]
    workflow = packet["workflow_state"]
    beta = packet["robinhood_beta_execution"]
    watchlist = packet["operator_watchlist"]
    journal = packet["trade_journal_hints"]
    execution_audit = packet.get("execution_audit") or {"available": False}
    proof = {
        "artifact_type": "sharpedge_android_operator_packet_export",
        "status": "exported",
        "exported_at": _timestamp(),
        "android_root": str(android_root),
        "source_signal_path": str(signal_path),
        "signal_live_import_path": signal_export.get("live_import_path"),
        "live_import_path": str(live_import_path),
        "schema": packet["schema"],
        "product": packet["product"],
        "web_viewer_refresh": web_viewer_refresh,
        "symbol": packet.get("symbol"),
        "status_summary": packet["status_summary"],
        "approval": {
            "decision": approval.get("decision"),
            "trade_allowed": approval.get("trade_allowed"),
            "broker_order_allowed": approval.get("broker_order_allowed"),
        },
        "workflow": {
            "readiness": (workflow.get("state") or {}).get("readiness"),
            "operator_action": (workflow.get("state") or {}).get("operator_action"),
            "lifecycle_stage": (workflow.get("state") or {}).get("lifecycle_stage"),
        },
        "watchlist": {
            "active_count": watchlist.get("active_count"),
            "items": len(watchlist.get("items") or []),
        },
        "journal": {
            "closed_trades": (journal.get("sample_state") or {}).get("closed_trades"),
            "top_patterns": len(journal.get("top_patterns") or []),
        },
        "robinhood": {
            "beta_stage": beta.get("beta_stage"),
            "token_action": (beta.get("order_preview") or {}).get("token_action"),
            "position_intent": (beta.get("order_preview") or {}).get("position_intent"),
            "contracts_held": (beta.get("edge_token_position") or {}).get(
                "contracts_held"
            ),
            "bridge_status": (beta.get("robinhood_beta_handoff") or {})
            .get("bridge_status", {})
            .get("status"),
            "approval_required": beta.get("approval_required"),
        },
        "execution_audit": {
            "available": execution_audit.get("available", False),
            "connector_status": execution_audit.get("connector_status"),
            "fill_status": execution_audit.get("fill_status"),
        },
        "android_viewer_bundle_included": ANDROID_VIEWER_BUNDLE_KEY in packet,
        "note": "Live operator packet is ready for import into SharpEdge Android app.",
    }
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main(argv: list[str]) -> int:
    signal_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_SIGNAL_PATH
    android_root = Path(argv[2]) if len(argv) > 2 else DEFAULT_ANDROID_ROOT
    proof_path = Path(argv[3]) if len(argv) > 3 else DEFAULT_PROOF_PATH
    live_import_path = Path(argv[4]) if len(argv) > 4 else DEFAULT_LIVE_IMPORT_PATH
    proof = export_operator_packet(
        signal_path, android_root, proof_path, live_import_path
    )
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
