#!/usr/bin/env python3
"""Build a Robinhood beta execution handoff with approval-gated shadow orders.

This layer extends the existing SharpEdge safety contract without granting live
order authority. It converts the current monitor + contract state into a beta
broker handoff that supports:

- read / monitor permissions
- order-draft intent generation
- explicit operator approval gates
- artifact-only fallback when broker integration is unavailable
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.agents.agent_language_views import (
    read_json,
    resolve_approval_decision,
    resolve_execution_plan,
    resolve_workflow_state,
)

OUTDIR = Path("outputs")
MONITOR_JSON = OUTDIR / "robinhood_fvg_monitor.json"
CONTRACT_JSON = OUTDIR / "agent_v1_decision.json"
BRIEF_JSON = OUTDIR / "operator_brief.json"
DASHBOARD_JSON = OUTDIR / "morning_open_dashboard.json"
BETA_JSON = OUTDIR / "robinhood_beta_execution.json"
WORKFLOW_STATE_JSON = OUTDIR / "workflow_state.json"
EXECUTION_PLAN_JSON = OUTDIR / "execution_plan.json"
APPROVAL_DECISION_JSON = OUTDIR / "approval_decision.json"
OUT_JSON = OUTDIR / "robinhood_beta_execution.json"
OUT_TXT = OUTDIR / "robinhood_beta_execution.txt"

BETA_PROFILE = os.getenv("ROBINHOOD_BETA_PROFILE", "approval_gated_shadow")
BETA_MAX_RISK_CAP_PCT = float(os.getenv("ROBINHOOD_BETA_MAX_RISK_CAP_PCT", "0.25"))
BETA_REQUIRE_PRICE_LIMIT = os.getenv("ROBINHOOD_BETA_REQUIRE_PRICE_LIMIT", "1") == "1"
BETA_REQUIRE_DEFINED_RISK = os.getenv("ROBINHOOD_BETA_REQUIRE_DEFINED_RISK", "1") == "1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def monitor_bridge_status(monitor: dict[str, Any]) -> dict[str, Any]:
    handoff = monitor.get("robinhood_mcp_handoff", {})
    bridge = handoff.get("bridge_status", {})
    if not isinstance(bridge, dict):
        return {}
    return bridge


def option_strategy_family(option_side_watch: str) -> str:
    side = str(option_side_watch or "none").lower()
    if "puts" in side:
        return "put_debit_spread" if BETA_REQUIRE_DEFINED_RISK else "long_put"
    if "calls" in side:
        return "call_debit_spread" if BETA_REQUIRE_DEFINED_RISK else "long_call"
    return "no_order_template"


def beta_stage(contract: dict[str, Any], bridge: dict[str, Any]) -> str:
    bridge_available = bool(bridge.get("available"))
    decision = str(contract.get("decision", "hold")).lower()
    trade_allowed = bool(contract.get("trade_allowed"))

    if not bridge_available:
        return "artifact_only"
    if decision == "operator_confirm_required" and trade_allowed:
        return "approval_queue_ready"
    if decision == "monitor":
        return "quote_monitoring"
    return "blocked"


def order_preview(
    workflow: dict[str, Any],
    plan: dict[str, Any],
    approval: dict[str, Any],
    monitor: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    context = plan.get("reference_context", {})
    options = monitor.get("options_context", {})
    max_risk_pct = min(
        as_float(approval.get("risk_limits", {}).get("max_capital_risk_pct")),
        BETA_MAX_RISK_CAP_PCT,
    )
    option_side_watch = context.get("option_side_watch") or workflow.get("market_context", {}).get(
        "option_side_watch"
    )
    strategy_family = option_strategy_family(str(option_side_watch or "none"))
    draft_allowed = stage == "approval_queue_ready" and strategy_family != "no_order_template"

    return {
        "draft_allowed": draft_allowed,
        "symbol": workflow.get("symbol", approval.get("symbol", "SPY")),
        "setup_type": "gap_fill_options_context",
        "headline": plan.get("objective", "No headline available."),
        "option_side_watch": option_side_watch,
        "strategy_family": strategy_family,
        "entry_style": "limit_debit" if BETA_REQUIRE_PRICE_LIMIT else "marketable_limit",
        "defined_risk_required": BETA_REQUIRE_DEFINED_RISK,
        "price_limit_required": BETA_REQUIRE_PRICE_LIMIT,
        "dte_window": {
            "min": options.get("dte_min"),
            "max": options.get("dte_max"),
        },
        "reference_levels": {
            "spot": context.get("spot"),
            "atm_strike": context.get("atm_strike"),
            "gap_fill_level": context.get("gap_fill_level"),
        },
        "risk_limits": {
            "max_capital_risk_pct": round(max_risk_pct, 4),
            "source_contract_risk_pct": as_float(
                approval.get("risk_limits", {}).get("max_capital_risk_pct")
            ),
            "beta_cap_risk_pct": BETA_MAX_RISK_CAP_PCT,
        },
        "leg_selection_rules": [
            "Use near-ATM long leg aligned with option_side_watch.",
            "If defined risk is required, sell further OTM leg to cap loss.",
            "Prefer same-session thesis alignment with contract freshness and monitor context.",
        ],
        "blocked_actions": [
            "submit_order_without_operator_approval",
            "cancel_order_without_operator_approval",
            "remove_defined_risk_when_beta_requires_it",
        ],
    }


def build_payload() -> dict[str, Any]:
    monitor = read_json(MONITOR_JSON)
    workflow = resolve_workflow_state(WORKFLOW_STATE_JSON, BRIEF_JSON, CONTRACT_JSON)
    approval = resolve_approval_decision(APPROVAL_DECISION_JSON, CONTRACT_JSON)
    plan = resolve_execution_plan(EXECUTION_PLAN_JSON, BRIEF_JSON, CONTRACT_JSON, BETA_JSON)
    bridge = monitor_bridge_status(monitor)
    stage = beta_stage(approval, bridge)
    preview = order_preview(workflow, plan, approval, monitor, stage)

    bridge_available = bool(bridge.get("available"))
    approval_required = True
    create_order_draft_allowed = bridge_available and preview["draft_allowed"]
    state = workflow.get("state", {})
    required_approvals = plan.get("prerequisites", {}).get("required_approvals") or [
        "operator_trade_confirmation",
        "price_limit_review",
        "defined_risk_review",
        "risk_budget_review",
    ]

    return {
        "schema_version": "robinhood_beta_execution.v1",
        "created_ts": utc_now(),
        "beta_profile": BETA_PROFILE,
        "beta_stage": stage,
        "symbol": workflow.get("symbol", approval.get("symbol", "SPY")),
        "headline": plan.get("objective", "No headline available."),
        "operator_action": plan.get("intended_action", state.get("operator_action", "stand_down")),
        "readiness": state.get("readiness", "blocked"),
        "broker_integration_status": state.get(
            "broker_integration_status",
            bridge.get("status", "unknown"),
        ),
        "monitoring_mode": state.get(
            "monitoring_mode",
            bridge.get("fallback_mode", "artifact_only_manual_review"),
        ),
        "trade_allowed": bool(approval.get("trade_allowed")),
        "approval_required": approval_required,
        "blocking_reasons": approval.get("blocking_reasons", []),
        "risk_flags": approval.get("risk_flags", []),
        "beta_capabilities": {
            "read_account_status": bridge_available,
            "read_positions": bridge_available,
            "read_option_chain": bridge_available,
            "monitor_quotes": bridge_available,
            "create_order_draft": create_order_draft_allowed,
            "submit_order": False,
            "replace_order": False,
            "cancel_order_without_operator_approval": False,
        },
        "required_approvals": required_approvals,
        "order_preview": preview,
        "robinhood_beta_handoff": {
            "server": bridge.get("server", "robinhood-trading"),
            "agent": bridge.get("agent", "code-puppy"),
            "bridge_available": bridge_available,
            "bridge_status": bridge,
            "intent": "approval_gated_shadow_order_management",
            "mode": "beta_shadow_execution",
            "manual_review_required": True,
            "fallback_mode": (
                "artifact_only_shadow_review" if not bridge_available else "approval_queue_shadow_draft"
            ),
            "permitted_actions": [
                "read_account_status",
                "read_positions",
                "read_option_chain",
                "monitor_underlying_quote",
                "monitor_option_quotes",
                "create_order_draft" if create_order_draft_allowed else "artifact_only_order_preview",
            ],
            "blocked_actions": [
                "submit_order_without_operator_approval",
                "replace_order_without_operator_approval",
                "cancel_order_without_operator_approval",
            ],
        },
    }


def render_text(payload: dict[str, Any]) -> str:
    preview = payload["order_preview"]
    caps = payload["beta_capabilities"]
    return "\n".join(
        [
            "ROBINHOOD BETA EXECUTION HANDOFF",
            f"Created: {payload['created_ts']}",
            f"Symbol: {payload['symbol']}",
            f"Beta profile: {payload['beta_profile']}",
            f"Beta stage: {payload['beta_stage']}",
            f"Headline: {payload['headline']}",
            f"Readiness: {payload['readiness']}",
            f"Broker integration: {payload['broker_integration_status']}",
            "",
            f"Trade allowed by contract: {payload['trade_allowed']}",
            f"Create order draft allowed: {caps['create_order_draft']}",
            f"Submit order allowed: {caps['submit_order']}",
            f"Approval required: {payload['approval_required']}",
            "",
            f"Strategy family: {preview['strategy_family']}",
            f"Option side watch: {preview['option_side_watch']}",
            f"Entry style: {preview['entry_style']}",
            f"DTE window: {preview['dte_window']['min']} to {preview['dte_window']['max']}",
            f"Max capital risk pct: {preview['risk_limits']['max_capital_risk_pct']}",
            "",
            f"Blocking reasons: {', '.join(payload['blocking_reasons']) or 'none'}",
            f"Risk flags: {', '.join(payload['risk_flags']) or 'none'}",
            "Orders remain approval-gated in beta. No autonomous live submission.",
        ]
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"robinhood_beta_stage={payload['beta_stage']}")


if __name__ == "__main__":
    main()
