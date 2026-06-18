#!/usr/bin/env python3
"""Build a morning-open dashboard artifact from operator-facing outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.agents.agent_language_views import (
    read_json,
    resolve_approval_decision,
    resolve_execution_plan,
    resolve_journal,
    resolve_workflow_state,
)

OUTDIR = Path("outputs")
BRIEF_JSON = OUTDIR / "operator_brief.json"
WATCHLIST_JSON = OUTDIR / "operator_watchlist.json"
CONTRACT_JSON = OUTDIR / "agent_v1_decision.json"
SESSION_REVIEW_JSON = OUTDIR / "operator_session_review.json"
WORKFLOW_STATE_JSON = OUTDIR / "workflow_state.json"
EXECUTION_PLAN_JSON = OUTDIR / "execution_plan.json"
APPROVAL_DECISION_JSON = OUTDIR / "approval_decision.json"
JOURNAL_JSON = OUTDIR / "journal.json"
BETA_JSON = OUTDIR / "robinhood_beta_execution.json"
OUT_JSON = OUTDIR / "morning_open_dashboard.json"
OUT_TXT = OUTDIR / "morning_open_dashboard.txt"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def checklist_item(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def build_dashboard() -> dict[str, Any]:
    watchlist = read_json(WATCHLIST_JSON)
    workflow = resolve_workflow_state(WORKFLOW_STATE_JSON, BRIEF_JSON, CONTRACT_JSON)
    approval = resolve_approval_decision(APPROVAL_DECISION_JSON, CONTRACT_JSON)
    plan = resolve_execution_plan(EXECUTION_PLAN_JSON, BRIEF_JSON, CONTRACT_JSON, BETA_JSON)
    journal = resolve_journal(JOURNAL_JSON, SESSION_REVIEW_JSON)

    stale_inputs = approval.get("freshness", {}).get("stale_inputs", [])
    state = workflow.get("state", {})
    focus = workflow.get("market_context", {})
    broker_status = str(state.get("broker_integration_status", "unknown"))
    operator_action = str(state.get("operator_action", plan.get("intended_action", "stand_down")))
    watch_items = watchlist.get("items", [])
    top_item = watch_items[0] if watch_items else {}
    historical = workflow.get("historical_context", {"available": False})
    readiness = state.get("readiness", "blocked")

    checklist = [
        checklist_item(
            "fresh_inputs",
            not stale_inputs,
            "No stale inputs detected." if not stale_inputs else f"{len(stale_inputs)} stale input(s) present.",
        ),
        checklist_item(
            "broker_integration",
            broker_status == "ready",
            f"Broker integration status={broker_status}.",
        ),
        checklist_item(
            "watchlist_focus",
            bool(watch_items),
            f"Watchlist items available={len(watch_items)}.",
        ),
        checklist_item(
            "trade_permission",
            bool(approval.get("trade_allowed")),
            f"Approval decision={approval.get('decision', 'missing')}.",
        ),
    ]

    return {
        "schema_version": "morning_open_dashboard.v1",
        "created_ts": utc_now(),
        "readiness": readiness,
        "operator_action": operator_action,
        "headline": plan.get("objective", "No headline available."),
        "market_focus": {
            "symbol": workflow.get("symbol", approval.get("symbol", "SPY")),
            "gap_direction": focus.get("gap_direction"),
            "gap_fill_level": focus.get("gap_fill_level"),
            "option_side_watch": focus.get("option_side_watch"),
            "spot": focus.get("spot"),
            "atm_strike": focus.get("atm_strike"),
            "dealer_state_hint": focus.get("dealer_state_hint"),
        },
        "permissions": approval.get("permissions", {}),
        "blocking_reasons": approval.get("blocking_reasons", []),
        "risk_flags": approval.get("risk_flags", []),
        "stale_inputs": stale_inputs,
        "checklist": checklist,
        "watchlist_snapshot": {
            "active_count": watchlist.get("active_count", 0),
            "top_item_status": top_item.get("status"),
            "top_item_priority": top_item.get("priority"),
            "top_item_headline": top_item.get("headline"),
        },
        "recent_review": {
            "journal_entries_reviewed": journal.get("entries_reviewed", 0),
            "latest_action": journal.get("latest_entry", {}).get("operator_action"),
            "top_blockers": journal.get("recurring_patterns", {}).get("top_blockers", []),
        },
        "historical_hints": {
            "available": historical.get("available", False),
            "top_pattern_summary": historical.get("top_pattern_summary"),
            "low_sample": historical.get("low_sample"),
            "metric_collection_priorities": historical.get("metric_collection_priorities", []),
        },
        "next_steps": plan.get("steps", []),
    }


def render_text(dashboard: dict[str, Any]) -> str:
    focus = dashboard["market_focus"]
    checklist = dashboard["checklist"]
    historical = dashboard.get("historical_hints", {})
    lines = [
        "SHARPEDGE MORNING OPEN DASHBOARD",
        f"Created: {dashboard['created_ts']}",
        f"Readiness: {dashboard['readiness']}",
        f"Operator action: {dashboard['operator_action']}",
        f"Headline: {dashboard['headline']}",
        "",
        f"Symbol: {focus['symbol']}",
        f"Gap direction / fill level: {focus['gap_direction']} / {focus['gap_fill_level']}",
        f"Option side watch: {focus['option_side_watch']}",
        f"Spot / ATM: {focus['spot']} / {focus['atm_strike']}",
        f"Dealer state: {focus['dealer_state_hint']}",
        "",
        "Checklist:",
        *[
            f"- [{'OK' if item['ok'] else 'BLOCKED'}] {item['name']}: {item['detail']}"
            for item in checklist
        ],
        "",
        f"Blocking reasons: {', '.join(dashboard['blocking_reasons']) or 'none'}",
        f"Risk flags: {', '.join(dashboard['risk_flags']) or 'none'}",
    ]
    if historical.get("available"):
        lines.extend(
            [
                "",
                f"Historical top pattern: {historical.get('top_pattern_summary')}",
                f"Historical low sample: {historical.get('low_sample')}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dashboard = build_dashboard()
    OUT_JSON.write_text(json.dumps(dashboard, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(dashboard), encoding="utf-8")
    print(json.dumps(dashboard, indent=2, sort_keys=True))
    print(f"morning_open_readiness={dashboard['readiness']}")


if __name__ == "__main__":
    main()
