#!/usr/bin/env python3
"""Build four canonical agent-language objects from existing SharpEdge artifacts.

These objects are intended to become the stable handoff language for future
agents. They normalize existing artifacts into four distinct roles:

- workflow_state: what is true right now
- execution_plan: what should happen next if conditions allow it
- approval_decision: what is actually authorized
- journal: what happened and what was learned

Authority rule:
    approval_decision is the only authoritative permission object.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AGENT_ID = os.getenv("AGENT_LANGUAGE_AGENT_ID", "code-puppy-ce5830")
OUTDIR = Path("outputs")
CONTROLLER_JSON = OUTDIR / "agent_controller_decision.json"
MONITOR_JSON = OUTDIR / "robinhood_fvg_monitor.json"
CONTRACT_JSON = OUTDIR / "agent_v1_decision.json"
BRIEF_JSON = OUTDIR / "operator_brief.json"
DASHBOARD_JSON = OUTDIR / "morning_open_dashboard.json"
BETA_JSON = OUTDIR / "robinhood_beta_execution.json"
SESSION_REVIEW_JSON = OUTDIR / "operator_session_review.json"
TRADE_HINTS_JSON = OUTDIR / "trade_journal_hints.json"
JOURNAL_JSONL = OUTDIR / "operator_journal_append.jsonl"
WORKFLOW_STATE_JSON = OUTDIR / "workflow_state.json"
WORKFLOW_STATE_TXT = OUTDIR / "workflow_state.txt"
EXECUTION_PLAN_JSON = OUTDIR / "execution_plan.json"
EXECUTION_PLAN_TXT = OUTDIR / "execution_plan.txt"
APPROVAL_DECISION_JSON = OUTDIR / "approval_decision.json"
APPROVAL_DECISION_TXT = OUTDIR / "approval_decision.txt"
JOURNAL_JSON = OUTDIR / "journal.json"
JOURNAL_TXT = OUTDIR / "journal.txt"
SCHEMA_VERSION = "agent_language.v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def load_inputs() -> dict[str, Any]:
    return {
        "controller": read_json(CONTROLLER_JSON),
        "monitor": read_json(MONITOR_JSON),
        "contract": read_json(CONTRACT_JSON),
        "brief": read_json(BRIEF_JSON),
        "session_review": read_json(SESSION_REVIEW_JSON),
        "trade_hints": read_json(TRADE_HINTS_JSON),
        "journal_entries": read_jsonl(JOURNAL_JSONL),
    }


def first_non_empty(*values: Any, default: str = "SPY") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def lifecycle_stage(inputs: dict[str, Any]) -> str:
    contract = inputs["contract"]
    brief = inputs["brief"]
    decision = str(contract.get("decision", "hold")).lower()
    operator_action = str(brief.get("operator_action", "stand_down")).lower()
    if decision == "operator_confirm_required":
        return "approval_pending_operator"
    if decision == "monitor" or operator_action == "monitor_only":
        return "monitor_only"
    if operator_action == "review_trade_plan":
        return "review_ready"
    return "blocked"


def build_run_id(inputs: dict[str, Any]) -> str:
    contract = inputs["contract"]
    controller = inputs["controller"]
    monitor = inputs["monitor"]
    brief = inputs["brief"]
    latest_entry = (inputs["journal_entries"] or [{}])[-1]
    markers = {
        "controller_ts": controller.get("ts_utc"),
        "monitor_ts": monitor.get("created_ts"),
        "contract_ts": contract.get("created_ts"),
        "brief_ts": brief.get("created_ts"),
        "journal_entry_id": latest_entry.get("entry_id"),
        "symbol": first_non_empty(
            contract.get("symbol"),
            brief.get("symbol"),
            controller.get("symbol"),
        ),
    }
    digest = hashlib.sha1(json.dumps(markers, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def source_artifacts() -> dict[str, str]:
    return {
        "controller": str(CONTROLLER_JSON),
        "monitor": str(MONITOR_JSON),
        "contract": str(CONTRACT_JSON),
        "brief": str(BRIEF_JSON),
        "session_review": str(SESSION_REVIEW_JSON),
        "trade_hints": str(TRADE_HINTS_JSON),
        "journal_append": str(JOURNAL_JSONL),
    }


def base_object(
    object_name: str,
    symbol: str,
    run_id: str,
    authority: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "object_name": object_name,
        "created_ts": utc_now(),
        "run_id": run_id,
        "agent_id": AGENT_ID,
        "symbol": symbol,
        "authority": authority,
        "source_artifacts": source_artifacts(),
    }


def summarize_recent_entries(entries: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    recent = entries[-limit:]
    return [
        {
            "created_ts": entry.get("created_ts"),
            "entry_id": entry.get("entry_id"),
            "operator_action": entry.get("operator_action"),
            "watchlist_status": entry.get("watchlist_status"),
            "headline": entry.get("headline"),
            "blocking_reasons": entry.get("blocking_reasons", []),
            "risk_flags": entry.get("risk_flags", []),
        }
        for entry in recent
    ]


def build_workflow_state(inputs: dict[str, Any], run_id: str) -> dict[str, Any]:
    contract = inputs["contract"]
    brief = inputs["brief"]
    monitor = inputs["monitor"]
    controller = inputs["controller"]
    symbol = first_non_empty(contract.get("symbol"), brief.get("symbol"), controller.get("symbol"))
    payload = base_object("workflow_state", symbol, run_id, authority="informational")
    payload.update(
        {
            "state": {
                "lifecycle_stage": lifecycle_stage(inputs),
                "operator_action": brief.get("operator_action", "stand_down"),
                "readiness": (
                    "review"
                    if brief.get("operator_action") == "review_trade_plan"
                    else "monitor"
                    if brief.get("operator_action") == "monitor_only"
                    else "blocked"
                ),
                "controller_decision": controller.get("decision", "missing"),
                "monitor_decision": monitor.get("decision", "missing"),
                "approval_decision": contract.get("decision", "missing"),
                "risk_state": contract.get("risk_state", "missing"),
                "broker_integration_status": contract.get(
                    "broker_integration_status",
                    brief.get("summary", {}).get("broker_integration_status", "unknown"),
                ),
                "monitoring_mode": contract.get(
                    "monitoring_mode",
                    brief.get("summary", {}).get("monitoring_mode", "unknown"),
                ),
            },
            "market_context": {
                "gap_direction": brief.get("focus", {}).get("gap_direction"),
                "gap_fill_level": brief.get("focus", {}).get("gap_fill_level"),
                "gap_session_date": brief.get("focus", {}).get("gap_session_date"),
                "fill_bias": brief.get("focus", {}).get("fill_bias"),
                "option_side_watch": brief.get("focus", {}).get("option_side_watch"),
                "spot": brief.get("focus", {}).get("spot"),
                "atm_strike": brief.get("focus", {}).get("atm_strike"),
                "dealer_state_hint": brief.get("focus", {}).get("dealer_state_hint"),
            },
            "evidence": {
                "artifact_readiness_confidence": controller.get("confidence", 0.0),
                "evidence_quality": contract.get("confidence_evidence_quality", 0.0),
                "trade_edge_confidence": contract.get("confidence_trade_edge", 0.0),
            },
            "blockers": contract.get("blocking_reasons", []),
            "risk_flags": contract.get("risk_flags", []),
            "freshness": contract.get("freshness", {}),
            "historical_context": brief.get("historical_hints", {"available": False}),
            "constraints": [
                "Informational only; use approval_decision for actual permission authority.",
            ],
        }
    )
    return payload


def build_execution_plan(inputs: dict[str, Any], run_id: str) -> dict[str, Any]:
    brief = inputs["brief"]
    contract = inputs["contract"]
    symbol = first_non_empty(contract.get("symbol"), brief.get("symbol"))
    required_human_action = contract.get("required_human_action", "none")
    required_approvals = [
        "operator_trade_confirmation",
        "price_limit_review",
        "defined_risk_review",
        "risk_budget_review",
    ]
    if required_human_action != "none":
        required_approvals.insert(0, required_human_action)
    payload = base_object("execution_plan", symbol, run_id, authority="advisory")
    payload.update(
        {
            "plan_status": lifecycle_stage(inputs),
            "objective": brief.get("headline", "No execution objective available."),
            "intended_action": brief.get("operator_action", "stand_down"),
            "execution_mode": {
                "monitoring_mode": contract.get("monitoring_mode", "unknown"),
                "fallback_mode": contract.get("monitoring_mode", "unknown"),
                "approval_style": "manual_confirmation_required",
            },
            "steps": brief.get("next_steps", []),
            "prerequisites": {
                "blocking_reasons": contract.get("blocking_reasons", []),
                "stale_inputs": contract.get("freshness", {}).get("stale_inputs", []),
                "required_approvals": required_approvals,
            },
            "reference_context": {
                "option_side_watch": brief.get("focus", {}).get("option_side_watch"),
                "spot": brief.get("focus", {}).get("spot"),
                "atm_strike": brief.get("focus", {}).get("atm_strike"),
                "gap_fill_level": brief.get("focus", {}).get("gap_fill_level"),
                "max_capital_risk_pct": contract.get("max_capital_risk_pct", 0.0),
            },
            "constraints": [
                "Advisory only; never escalate permission from this object.",
                "Re-check approval_decision before any broker-side action or human handoff.",
            ],
        }
    )
    return payload


def build_approval_decision(inputs: dict[str, Any], run_id: str) -> dict[str, Any]:
    contract = inputs["contract"]
    symbol = first_non_empty(contract.get("symbol"), inputs["brief"].get("symbol"))
    payload = base_object(
        "approval_decision",
        symbol,
        run_id,
        authority="authoritative_permission_gate",
    )
    payload.update(
        {
            "decision": contract.get("decision", "missing"),
            "trade_allowed": bool(contract.get("trade_allowed")),
            "broker_order_allowed": bool(contract.get("broker_order_allowed")),
            "required_human_action": contract.get("required_human_action", "none"),
            "permissions": contract.get("permissions", {}),
            "risk_limits": {
                "max_capital_risk_pct": contract.get("max_capital_risk_pct", 0.0),
                "risk_state": contract.get("risk_state", "missing"),
            },
            "blocking_reasons": contract.get("blocking_reasons", []),
            "risk_flags": contract.get("risk_flags", []),
            "freshness": contract.get("freshness", {}),
            "confidence": {
                "evidence_quality": contract.get("confidence_evidence_quality", 0.0),
                "trade_edge_confidence": contract.get("confidence_trade_edge", 0.0),
            },
            "source_decisions": contract.get("source_decisions", {}),
            "authority_policy": {
                "contract_wins": True,
                "manual_confirmation_required": not bool(contract.get("broker_order_allowed")),
                "notes": [
                    "Downstream layers may summarize this object, but must not override it.",
                    "Trade permission remains blocked until this object explicitly allows it.",
                ],
            },
        }
    )
    return payload


def build_journal(inputs: dict[str, Any], run_id: str) -> dict[str, Any]:
    review = inputs["session_review"]
    hints = inputs["trade_hints"]
    entries = inputs["journal_entries"]
    contract = inputs["contract"]
    brief = inputs["brief"]
    symbol = first_non_empty(contract.get("symbol"), brief.get("symbol"))
    latest = entries[-1] if entries else review.get("latest_entry", {})
    payload = base_object("journal", symbol, run_id, authority="historical_non_authoritative")
    payload.update(
        {
            "journal_mode": "append_only_summary",
            "entries_total": len(entries),
            "entries_reviewed": review.get("journal_entries_reviewed", len(entries)),
            "latest_entry": {
                "created_ts": latest.get("created_ts"),
                "entry_id": latest.get("entry_id"),
                "operator_action": latest.get("operator_action"),
                "watchlist_status": latest.get("watchlist_status"),
                "headline": latest.get("headline"),
                "blocking_reasons": latest.get("blocking_reasons", []),
                "risk_flags": latest.get("risk_flags", []),
            },
            "recent_entries": summarize_recent_entries(entries),
            "recurring_patterns": {
                "top_blockers": review.get("top_blockers", []),
                "top_risk_flags": review.get("top_risk_flags", []),
                "historical_top_pattern": (hints.get("top_patterns") or [{}])[0],
            },
            "lessons": [
                item.get("summary")
                for item in hints.get("actionable_hints", [])
                if item.get("summary")
            ][:4],
            "research_backlog": {
                "metric_collection_priorities": hints.get("metric_collection_priorities", []),
                "research_hypotheses": hints.get("research_hypotheses", []),
            },
            "constraints": hints.get(
                "usage_constraints",
                ["Journal is memory, not authority."],
            ),
        }
    )
    return payload


def build_objects() -> dict[str, dict[str, Any]]:
    inputs = load_inputs()
    run_id = build_run_id(inputs)
    return {
        "workflow_state": build_workflow_state(inputs, run_id),
        "execution_plan": build_execution_plan(inputs, run_id),
        "approval_decision": build_approval_decision(inputs, run_id),
        "journal": build_journal(inputs, run_id),
    }


def render_workflow_state(payload: dict[str, Any]) -> str:
    state = payload["state"]
    return "\n".join(
        [
            "WORKFLOW STATE",
            f"Run id: {payload['run_id']}",
            f"Symbol: {payload['symbol']}",
            f"Lifecycle stage: {state['lifecycle_stage']}",
            f"Operator action: {state['operator_action']}",
            f"Readiness: {state['readiness']}",
            f"Approval decision: {state['approval_decision']}",
            f"Blockers: {', '.join(payload['blockers']) or 'none'}",
            f"Risk flags: {', '.join(payload['risk_flags']) or 'none'}",
        ]
    ) + "\n"


def render_execution_plan(payload: dict[str, Any]) -> str:
    prereqs = payload["prerequisites"]
    lines = [
        "EXECUTION PLAN",
        f"Run id: {payload['run_id']}",
        f"Symbol: {payload['symbol']}",
        f"Plan status: {payload['plan_status']}",
        f"Intended action: {payload['intended_action']}",
        "Steps:",
        *[f"- {step}" for step in payload.get("steps", [])],
        f"Required approvals: {', '.join(prereqs['required_approvals']) or 'none'}",
        f"Blocking reasons: {', '.join(prereqs['blocking_reasons']) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def render_approval_decision(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "APPROVAL DECISION",
            f"Run id: {payload['run_id']}",
            f"Symbol: {payload['symbol']}",
            f"Decision: {payload['decision']}",
            f"Trade allowed: {payload['trade_allowed']}",
            f"Broker order allowed: {payload['broker_order_allowed']}",
            f"Required human action: {payload['required_human_action']}",
            f"Blocking reasons: {', '.join(payload['blocking_reasons']) or 'none'}",
            f"Risk flags: {', '.join(payload['risk_flags']) or 'none'}",
        ]
    ) + "\n"


def render_journal(payload: dict[str, Any]) -> str:
    latest = payload["latest_entry"]
    lesson_lines = [f"- {item}" for item in payload.get("lessons", [])] or ["- none"]
    return "\n".join(
        [
            "JOURNAL",
            f"Run id: {payload['run_id']}",
            f"Symbol: {payload['symbol']}",
            f"Entries total: {payload['entries_total']}",
            f"Latest action: {latest.get('operator_action', 'none')}",
            f"Latest headline: {latest.get('headline', 'none')}",
            "Lessons:",
            *lesson_lines,
        ]
    ) + "\n"


def write_object(json_path: Path, txt_path: Path, payload: dict[str, Any], text: str) -> None:
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    objects = build_objects()
    write_object(
        WORKFLOW_STATE_JSON,
        WORKFLOW_STATE_TXT,
        objects["workflow_state"],
        render_workflow_state(objects["workflow_state"]),
    )
    write_object(
        EXECUTION_PLAN_JSON,
        EXECUTION_PLAN_TXT,
        objects["execution_plan"],
        render_execution_plan(objects["execution_plan"]),
    )
    write_object(
        APPROVAL_DECISION_JSON,
        APPROVAL_DECISION_TXT,
        objects["approval_decision"],
        render_approval_decision(objects["approval_decision"]),
    )
    write_object(
        JOURNAL_JSON,
        JOURNAL_TXT,
        objects["journal"],
        render_journal(objects["journal"]),
    )
    print(json.dumps(objects, indent=2, sort_keys=True))
    print(f"agent_language_run_id={objects['approval_decision']['run_id']}")


if __name__ == "__main__":
    main()
