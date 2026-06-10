#!/usr/bin/env python3
"""Build a morning-open dashboard artifact from operator-facing outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTDIR = Path("outputs")
BRIEF_JSON = OUTDIR / "operator_brief.json"
WATCHLIST_JSON = OUTDIR / "operator_watchlist.json"
CONTRACT_JSON = OUTDIR / "agent_v1_decision.json"
SESSION_REVIEW_JSON = OUTDIR / "operator_session_review.json"
OUT_JSON = OUTDIR / "morning_open_dashboard.json"
OUT_TXT = OUTDIR / "morning_open_dashboard.txt"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def checklist_item(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def build_dashboard() -> dict[str, Any]:
    brief = read_json(BRIEF_JSON)
    watchlist = read_json(WATCHLIST_JSON)
    contract = read_json(CONTRACT_JSON)
    review = read_json(SESSION_REVIEW_JSON)

    stale_inputs = contract.get("freshness", {}).get("stale_inputs", [])
    broker_status = str(brief.get("summary", {}).get("broker_integration_status", "unknown"))
    operator_action = str(brief.get("operator_action", "stand_down"))
    watch_items = watchlist.get("items", [])
    top_item = watch_items[0] if watch_items else {}

    readiness = "blocked"
    if operator_action == "review_trade_plan":
        readiness = "review"
    elif operator_action == "monitor_only":
        readiness = "monitor"

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
            bool(contract.get("trade_allowed")),
            f"Contract decision={contract.get('decision', 'missing')}.",
        ),
    ]

    return {
        "schema_version": "morning_open_dashboard.v1",
        "created_ts": utc_now(),
        "readiness": readiness,
        "operator_action": operator_action,
        "headline": brief.get("headline", "No headline available."),
        "market_focus": {
            "symbol": brief.get("symbol", "SPY"),
            "gap_direction": brief.get("focus", {}).get("gap_direction"),
            "gap_fill_level": brief.get("focus", {}).get("gap_fill_level"),
            "option_side_watch": brief.get("focus", {}).get("option_side_watch"),
            "spot": brief.get("focus", {}).get("spot"),
            "atm_strike": brief.get("focus", {}).get("atm_strike"),
            "dealer_state_hint": brief.get("focus", {}).get("dealer_state_hint"),
        },
        "permissions": contract.get("permissions", {}),
        "blocking_reasons": contract.get("blocking_reasons", []),
        "risk_flags": contract.get("risk_flags", []),
        "stale_inputs": stale_inputs,
        "checklist": checklist,
        "watchlist_snapshot": {
            "active_count": watchlist.get("active_count", 0),
            "top_item_status": top_item.get("status"),
            "top_item_priority": top_item.get("priority"),
            "top_item_headline": top_item.get("headline"),
        },
        "recent_review": {
            "journal_entries_reviewed": review.get("journal_entries_reviewed", 0),
            "latest_action": review.get("latest_entry", {}).get("operator_action"),
            "top_blockers": review.get("top_blockers", []),
        },
        "next_steps": brief.get("next_steps", []),
    }


def render_text(dashboard: dict[str, Any]) -> str:
    focus = dashboard["market_focus"]
    checklist = dashboard["checklist"]
    return "\n".join(
        [
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
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dashboard = build_dashboard()
    OUT_JSON.write_text(json.dumps(dashboard, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(dashboard), encoding="utf-8")
    print(json.dumps(dashboard, indent=2, sort_keys=True))
    print(f"morning_open_readiness={dashboard['readiness']}")


if __name__ == "__main__":
    main()
