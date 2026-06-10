#!/usr/bin/env python3
"""Emit SharpEdge Agentic AI v1.0 final decision contract.

This is not a trade executor. It is a safety-oriented contract builder that
normalizes the final pipeline state into explicit permissions, blockers,
freshness checks, and artifact references.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
SYMBOL = os.getenv("SYMBOL", "SPY").upper()
OUTDIR = Path("outputs")

CONTROLLER_JSON = OUTDIR / "agent_controller_decision.json"
MONITOR_JSON = OUTDIR / "robinhood_fvg_monitor.json"
HEALTH_WARNINGS = OUTDIR / "health" / "warnings.log"
OUT_JSON = OUTDIR / "agent_v1_decision.json"
OUT_TXT = OUTDIR / "agent_v1_decision.txt"

MAX_CURRENT_AGE_DAYS = int(os.getenv("AGENT_V1_MAX_CURRENT_AGE_DAYS", "3"))
MAX_OPTIONS_AGE_DAYS = int(os.getenv("AGENT_V1_MAX_OPTIONS_AGE_DAYS", "7"))
MIN_STANDARD_SAMPLE_N = int(os.getenv("AGENT_V1_MIN_STANDARD_SAMPLE_N", "30"))


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_warnings() -> list[str]:
    if not HEALTH_WARNINGS.exists():
        return []
    return [
        line.strip()
        for line in HEALTH_WARNINGS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def monitor_bridge_status(monitor: dict[str, Any]) -> dict[str, Any]:
    handoff = monitor.get("robinhood_mcp_handoff", {})
    bridge = handoff.get("bridge_status", {})
    if not isinstance(bridge, dict) or not bridge:
        return {
            "available": True,
            "status": "legacy_unknown",
            "fallback_mode": handoff.get("fallback_mode", "mcp_quote_monitoring"),
        }
    return {
        "available": bool(bridge.get("available")),
        "status": str(bridge.get("status", "unknown")),
        "fallback_mode": str(
            handoff.get("fallback_mode") or bridge.get("fallback_mode") or "artifact_only_manual_review"
        ),
    }


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def age_days(value: Any) -> int | None:
    parsed = parse_date(value)
    if parsed is None:
        return None
    return (utc_now().date() - parsed).days


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def fetch_latest(con: sqlite3.Connection, table: str, query: str) -> dict[str, Any]:
    if not table_exists(con, table):
        return {}
    con.row_factory = sqlite3.Row
    row = con.execute(query, (SYMBOL,)).fetchone()
    return dict(row) if row else {}


def db_context() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"db_present": False}
    con = sqlite3.connect(DB_PATH)
    try:
        return {
            "db_present": True,
            "risk": fetch_latest(
                con,
                "risk_decision_layer",
                """
                SELECT * FROM risk_decision_layer
                WHERE symbol = ?
                ORDER BY date DESC, decision_ts DESC
                LIMIT 1
                """,
            ),
            "signal": fetch_latest(
                con,
                "signals_daily",
                """
                SELECT * FROM signals_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
                """,
            ),
            "regime": fetch_latest(
                con,
                "regime_daily",
                """
                SELECT * FROM regime_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
                """,
            ),
            "options": fetch_latest(
                con,
                "options_positioning_metrics",
                """
                SELECT * FROM options_positioning_metrics
                WHERE underlying = ?
                ORDER BY session_date DESC, snapshot_ts DESC
                LIMIT 1
                """,
            ),
        }
    finally:
        con.close()


def stale_inputs(monitor: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = [
        ("monitor_gap", monitor.get("latest_gap_event", {}).get("session_date"), MAX_CURRENT_AGE_DAYS),
        ("risk_layer", context.get("risk", {}).get("date"), MAX_CURRENT_AGE_DAYS),
        ("signal", context.get("signal", {}).get("date"), MAX_CURRENT_AGE_DAYS),
        ("regime", context.get("regime", {}).get("date"), MAX_CURRENT_AGE_DAYS),
        ("options_positioning", context.get("options", {}).get("session_date"), MAX_OPTIONS_AGE_DAYS),
    ]
    stale: list[dict[str, Any]] = []
    for name, value, max_age in inputs:
        age = age_days(value)
        if age is None:
            stale.append({"input": name, "date": value, "reason": "missing_or_invalid_date"})
        elif age > max_age:
            stale.append({"input": name, "date": value, "age_days": age, "max_age_days": max_age})
    return stale


def build_contract() -> dict[str, Any]:
    controller = read_json(CONTROLLER_JSON)
    monitor = read_json(MONITOR_JSON)
    context = db_context()
    warnings = read_warnings()
    risk = context.get("risk", {}) or {}
    bridge = monitor_bridge_status(monitor)

    blocking_reasons: list[str] = []
    risk_flags: list[str] = []

    if not context.get("db_present"):
        blocking_reasons.append("database_missing")

    if warnings:
        blocking_reasons.append("pipeline_warnings_present")
        risk_flags.append("pipeline_warnings_present")

    if not bridge["available"]:
        risk_flags.append("broker_integration_unavailable")

    monitor_decision = str(monitor.get("decision", "missing")).lower()
    if monitor_decision != "watch":
        blocking_reasons.append(f"monitor_{monitor_decision}")
        risk_flags.append("monitor_blocks_trade")

    controller_decision = str(controller.get("decision", "missing")).lower()
    if controller_decision != "post":
        blocking_reasons.append(f"controller_{controller_decision}")

    deployment_state = str(risk.get("deployment_state", "missing")).upper()
    if deployment_state in {"NO_TRADE", "WATCH", "MISSING"}:
        blocking_reasons.append(f"risk_state_{deployment_state.lower()}")
        risk_flags.append("risk_layer_blocks_trade")

    sample_n = as_int(risk.get("sample_n"))
    if sample_n < MIN_STANDARD_SAMPLE_N:
        blocking_reasons.append(f"sample_n_below_{MIN_STANDARD_SAMPLE_N}")
        risk_flags.append("low_sample")

    stale = stale_inputs(monitor, context)
    if stale:
        blocking_reasons.append("stale_or_missing_inputs")
        risk_flags.append("freshness_gate_failed")

    trade_edge_confidence = 0.0
    if monitor_decision == "watch" and deployment_state in {"STANDARD", "AGGRESSIVE"}:
        trade_edge_confidence = min(1.0, as_float(risk.get("deployment_confidence")))

    evidence_quality = as_float(controller.get("confidence"))
    broker_allowed = False
    trade_allowed = not blocking_reasons and trade_edge_confidence >= 0.55
    max_risk = as_float(risk.get("capital_risk_pct")) if trade_allowed else 0.0

    decision = "hold"
    required_human_action = "none"
    if trade_allowed:
        decision = "operator_confirm_required"
        required_human_action = "confirm_order"
        broker_allowed = False
    elif monitor_decision == "watch" and not warnings:
        decision = "monitor"

    broker_read_allowed = bridge["available"]

    return {
        "schema_version": "agentic_ai_v1.0",
        "created_ts": utc_now().isoformat(),
        "symbol": SYMBOL,
        "decision": decision,
        "trade_allowed": trade_allowed,
        "broker_order_allowed": broker_allowed,
        "mode": "dry_run_safety_contract",
        "monitoring_mode": bridge["fallback_mode"],
        "broker_integration_status": bridge["status"],
        "confidence_evidence_quality": round(evidence_quality, 4),
        "confidence_trade_edge": round(trade_edge_confidence, 4),
        "risk_state": deployment_state,
        "max_capital_risk_pct": max_risk,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "risk_flags": sorted(set(risk_flags)),
        "required_human_action": required_human_action,
        "freshness": {
            "max_current_age_days": MAX_CURRENT_AGE_DAYS,
            "max_options_age_days": MAX_OPTIONS_AGE_DAYS,
            "stale_inputs": stale,
        },
        "source_decisions": {
            "controller_decision": controller_decision,
            "monitor_decision": monitor_decision,
            "risk_deployment_state": deployment_state,
            "risk_sample_n": sample_n,
        },
        "permissions": {
            "read_market_data": True,
            "read_account_status": broker_read_allowed,
            "read_positions": broker_read_allowed,
            "read_option_chain": broker_read_allowed,
            "monitor_quotes": broker_read_allowed,
            "place_order": False,
            "replace_order": False,
            "cancel_order_without_operator_confirmation": False,
        },
    }


def render_text(contract: dict[str, Any]) -> str:
    stale = contract["freshness"]["stale_inputs"]
    return "\n".join(
        [
            "SHARPEDGE AGENTIC AI V1 DECISION",
            f"Created: {contract['created_ts']}",
            f"Symbol: {contract['symbol']}",
            f"Decision: {contract['decision']}",
            f"Trade allowed: {contract['trade_allowed']}",
            f"Broker order allowed: {contract['broker_order_allowed']}",
            f"Risk state: {contract['risk_state']}",
            f"Max capital risk pct: {contract['max_capital_risk_pct']}",
            f"Evidence quality: {contract['confidence_evidence_quality']}",
            f"Trade edge confidence: {contract['confidence_trade_edge']}",
            f"Blocking reasons: {', '.join(contract['blocking_reasons']) or 'none'}",
            f"Risk flags: {', '.join(contract['risk_flags']) or 'none'}",
            f"Stale inputs: {len(stale)}",
            "Orders remain blocked unless an operator manually confirms outside this contract.",
        ]
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    OUT_JSON.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(contract), encoding="utf-8")
    print(json.dumps(contract, indent=2, sort_keys=True))
    print(f"agent_v1_decision={contract['decision']}")


if __name__ == "__main__":
    main()
