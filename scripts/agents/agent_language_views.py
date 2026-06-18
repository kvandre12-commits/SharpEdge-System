#!/usr/bin/env python3
"""Helpers for consuming canonical agent-language objects with legacy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def derive_readiness(operator_action: str) -> str:
    if operator_action == "review_trade_plan":
        return "review"
    if operator_action == "monitor_only":
        return "monitor"
    return "blocked"


def resolve_workflow_state(
    workflow_state_path: Path,
    brief_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    workflow = read_json(workflow_state_path)
    if workflow:
        return workflow

    brief = read_json(brief_path)
    contract = read_json(contract_path)
    operator_action = str(brief.get("operator_action", "stand_down"))
    return {
        "symbol": brief.get("symbol", contract.get("symbol", "SPY")),
        "state": {
            "lifecycle_stage": (
                "approval_pending_operator"
                if contract.get("decision") == "operator_confirm_required"
                else "monitor_only"
                if contract.get("decision") == "monitor"
                else "blocked"
            ),
            "operator_action": operator_action,
            "readiness": derive_readiness(operator_action),
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
            "option_side_watch": brief.get("focus", {}).get("option_side_watch"),
            "spot": brief.get("focus", {}).get("spot"),
            "atm_strike": brief.get("focus", {}).get("atm_strike"),
            "dealer_state_hint": brief.get("focus", {}).get("dealer_state_hint"),
        },
        "historical_context": brief.get("historical_hints", {"available": False}),
        "blockers": contract.get("blocking_reasons", []),
        "risk_flags": contract.get("risk_flags", []),
        "freshness": contract.get("freshness", {}),
    }


def resolve_approval_decision(
    approval_decision_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    approval = read_json(approval_decision_path)
    if approval:
        return approval

    contract = read_json(contract_path)
    return {
        "symbol": contract.get("symbol", "SPY"),
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
    }


def resolve_execution_plan(
    execution_plan_path: Path,
    brief_path: Path,
    contract_path: Path,
    beta_path: Path,
) -> dict[str, Any]:
    plan = read_json(execution_plan_path)
    if plan:
        return plan

    brief = read_json(brief_path)
    contract = read_json(contract_path)
    beta = read_json(beta_path)
    required_approvals = list(beta.get("required_approvals", []))
    required_human_action = contract.get("required_human_action", "none")
    if required_human_action != "none":
        required_approvals.insert(0, required_human_action)

    return {
        "symbol": brief.get("symbol", contract.get("symbol", "SPY")),
        "plan_status": (
            "approval_pending_operator"
            if contract.get("decision") == "operator_confirm_required"
            else "monitor_only"
            if contract.get("decision") == "monitor"
            else "blocked"
        ),
        "objective": brief.get("headline", "No execution objective available."),
        "intended_action": brief.get("operator_action", "stand_down"),
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
        "capability_context": beta.get("beta_capabilities", {}),
    }


def resolve_journal(
    journal_path: Path,
    session_review_path: Path,
) -> dict[str, Any]:
    journal = read_json(journal_path)
    if journal:
        return journal

    review = read_json(session_review_path)
    latest = review.get("latest_entry", {})
    return {
        "entries_reviewed": review.get("journal_entries_reviewed", 0),
        "latest_entry": {
            "operator_action": latest.get("operator_action"),
            "headline": latest.get("headline"),
        },
        "recurring_patterns": {
            "top_blockers": review.get("top_blockers", []),
        },
        "lessons": [],
    }
