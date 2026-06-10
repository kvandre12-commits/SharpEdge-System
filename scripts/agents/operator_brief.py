#!/usr/bin/env python3
"""Build concise operator-facing artifacts from SharpEdge outputs.

This script is intentionally simple: it reads the existing local contracts and
monitor artifacts, then emits an operator brief, a watchlist snapshot, and an
append-only journal line for the human operator.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SYMBOL = os.getenv("SYMBOL", "SPY").upper()
OUTDIR = Path("outputs")

CONTROLLER_JSON = OUTDIR / "agent_controller_decision.json"
MONITOR_JSON = OUTDIR / "robinhood_fvg_monitor.json"
AGENT_V1_JSON = OUTDIR / "agent_v1_decision.json"
HEALTH_WARNINGS = OUTDIR / "health" / "warnings.log"
OUT_JSON = OUTDIR / "operator_brief.json"
OUT_TXT = OUTDIR / "operator_brief.txt"
OUT_WATCHLIST_JSON = OUTDIR / "operator_watchlist.json"
OUT_JOURNAL_JSONL = OUTDIR / "operator_journal_append.jsonl"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    return (
        read_json(CONTROLLER_JSON),
        read_json(MONITOR_JSON),
        read_json(AGENT_V1_JSON),
        read_warnings(),
    )


def choose_operator_action(contract: dict[str, Any]) -> str:
    decision = str(contract.get("decision", "hold")).lower()
    if decision == "operator_confirm_required":
        return "review_trade_plan"
    if decision == "monitor":
        return "monitor_only"
    return "stand_down"


def watchlist_status(operator_action: str) -> str:
    if operator_action == "review_trade_plan":
        return "ready_for_review"
    if operator_action == "monitor_only":
        return "monitor_only"
    return "blocked"


def watchlist_priority(operator_action: str) -> str:
    if operator_action == "review_trade_plan":
        return "high"
    if operator_action == "monitor_only":
        return "medium"
    return "low"


def build_headline(operator_action: str, monitor: dict[str, Any]) -> str:
    gap = monitor.get("latest_gap_event", {})
    direction = str(gap.get("gap_direction", "NA")).upper()
    fill_level = gap.get("gap_fill_level", "NA")
    if operator_action == "review_trade_plan":
        return f"Review {direction} gap-fill setup near {fill_level}; manual confirmation still required."
    if operator_action == "monitor_only":
        return f"Monitor {direction} gap-fill behavior near {fill_level}; no order path is open."
    return "Stand down. Preserve context, but do not act on this setup."


def build_next_steps(
    operator_action: str,
    contract: dict[str, Any],
    monitor: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    steps: list[str] = []
    hypothesis = monitor.get("directional_hypothesis", {})
    bridge_status = str(contract.get("broker_integration_status", "unknown"))
    fill_level = monitor.get("latest_gap_event", {}).get("gap_fill_level")
    option_side = hypothesis.get("option_side_watch", "none")

    if operator_action == "review_trade_plan":
        steps.append(
            f"Review the {option_side} thesis against live price behavior near gap fill level {fill_level}."
        )
        steps.append("Confirm sample quality, freshness, and risk budget before any manual action.")
    elif operator_action == "monitor_only":
        steps.append(
            f"Watch whether price moves toward gap fill level {fill_level} and whether the thesis stays intact."
        )
        steps.append("Do not place orders; use this run as structured observation only.")
    else:
        blockers = contract.get("blocking_reasons", [])
        steps.append(f"Do nothing until blockers clear: {', '.join(blockers) or 'unknown blocker'}.")

    if bridge_status != "ready":
        steps.append("Broker integration is not live; rely on artifact review and manual platform checks.")
    if warnings:
        steps.append(f"Pipeline emitted {len(warnings)} warning(s); inspect outputs/health/warnings.log.")
    steps.append("Orders remain blocked by design unless manually confirmed outside the automation loop.")
    return steps[:5]


def build_brief_payload(
    controller: dict[str, Any],
    monitor: dict[str, Any],
    contract: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    operator_action = choose_operator_action(contract)
    gap = monitor.get("latest_gap_event", {})
    hypothesis = monitor.get("directional_hypothesis", {})
    options = monitor.get("options_context", {})
    risk = monitor.get("risk_context", {})
    stale = contract.get("freshness", {}).get("stale_inputs", [])

    return {
        "schema_version": "operator_brief.v1",
        "created_ts": utc_now(),
        "symbol": SYMBOL,
        "operator_action": operator_action,
        "headline": build_headline(operator_action, monitor),
        "summary": {
            "controller_decision": controller.get("decision", "missing"),
            "monitor_decision": monitor.get("decision", "missing"),
            "contract_decision": contract.get("decision", "missing"),
            "risk_state": contract.get(
                "risk_state", str(risk.get("deployment_state", "missing"))
            ),
            "broker_integration_status": contract.get(
                "broker_integration_status", "unknown"
            ),
            "monitoring_mode": contract.get("monitoring_mode", "unknown"),
        },
        "focus": {
            "fill_bias": hypothesis.get("fill_bias", "unknown"),
            "option_side_watch": hypothesis.get("option_side_watch", "none"),
            "gap_direction": gap.get("gap_direction", "NA"),
            "gap_fill_level": gap.get("gap_fill_level", "NA"),
            "gap_session_date": gap.get("session_date"),
            "spot": options.get("spot"),
            "atm_strike": options.get("atm_strike"),
            "dealer_state_hint": options.get("dealer_state_hint"),
        },
        "confidence": {
            "evidence_quality": contract.get("confidence_evidence_quality", 0.0),
            "trade_edge": contract.get("confidence_trade_edge", 0.0),
            "controller_confidence": controller.get("confidence", 0.0),
        },
        "risk": {
            "max_capital_risk_pct": contract.get("max_capital_risk_pct", 0.0),
            "blocking_reasons": contract.get("blocking_reasons", []),
            "risk_flags": contract.get("risk_flags", []),
            "stale_inputs": stale,
            "sample_n": risk.get("sample_n"),
        },
        "next_steps": build_next_steps(operator_action, contract, monitor, warnings),
        "artifacts": {
            "controller": str(CONTROLLER_JSON),
            "monitor": str(MONITOR_JSON),
            "contract": str(AGENT_V1_JSON),
        },
    }


def build_brief() -> dict[str, Any]:
    controller, monitor, contract, warnings = load_inputs()
    return build_brief_payload(controller, monitor, contract, warnings)


def build_watchlist_payload(brief: dict[str, Any]) -> dict[str, Any]:
    status = watchlist_status(brief["operator_action"])
    item = {
        "item_id": (
            f"{brief['symbol']}-gap-fill-"
            f"{brief['focus'].get('gap_session_date') or 'na'}-"
            f"{str(brief['focus'].get('gap_direction', 'na')).lower()}"
        ),
        "symbol": brief["symbol"],
        "setup_type": "gap_fill_options_context",
        "status": status,
        "priority": watchlist_priority(brief["operator_action"]),
        "operator_action": brief["operator_action"],
        "headline": brief["headline"],
        "gap_session_date": brief["focus"].get("gap_session_date"),
        "gap_direction": brief["focus"].get("gap_direction"),
        "gap_fill_level": brief["focus"].get("gap_fill_level"),
        "fill_bias": brief["focus"].get("fill_bias"),
        "option_side_watch": brief["focus"].get("option_side_watch"),
        "spot": brief["focus"].get("spot"),
        "atm_strike": brief["focus"].get("atm_strike"),
        "dealer_state_hint": brief["focus"].get("dealer_state_hint"),
        "broker_integration_status": brief["summary"].get("broker_integration_status"),
        "monitoring_mode": brief["summary"].get("monitoring_mode"),
        "blocking_reasons": brief["risk"].get("blocking_reasons", []),
        "risk_flags": brief["risk"].get("risk_flags", []),
        "stale_inputs_count": len(brief["risk"].get("stale_inputs", [])),
    }
    return {
        "schema_version": "operator_watchlist.v1",
        "created_ts": utc_now(),
        "symbol": brief["symbol"],
        "active_count": 0 if status == "blocked" else 1,
        "items": [item],
    }


def build_watchlist() -> dict[str, Any]:
    return build_watchlist_payload(build_brief())


def build_journal_entry_payload(
    brief: dict[str, Any],
    controller: dict[str, Any],
    monitor: dict[str, Any],
    contract: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    watchlist = build_watchlist_payload(brief)
    status = watchlist["items"][0]["status"]
    identity_payload = {
        "controller_ts": controller.get("ts_utc"),
        "monitor_ts": monitor.get("created_ts"),
        "contract_ts": contract.get("created_ts"),
        "symbol": brief["symbol"],
        "operator_action": brief["operator_action"],
        "headline": brief["headline"],
        "status": status,
    }
    entry_id = hashlib.sha1(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "entry_id": entry_id,
        "created_ts": brief["created_ts"],
        "symbol": brief["symbol"],
        "operator_action": brief["operator_action"],
        "watchlist_status": status,
        "headline": brief["headline"],
        "controller_decision": brief["summary"].get("controller_decision"),
        "monitor_decision": brief["summary"].get("monitor_decision"),
        "contract_decision": brief["summary"].get("contract_decision"),
        "risk_state": brief["summary"].get("risk_state"),
        "broker_integration_status": brief["summary"].get(
            "broker_integration_status"
        ),
        "monitoring_mode": brief["summary"].get("monitoring_mode"),
        "gap_direction": brief["focus"].get("gap_direction"),
        "gap_fill_level": brief["focus"].get("gap_fill_level"),
        "option_side_watch": brief["focus"].get("option_side_watch"),
        "spot": brief["focus"].get("spot"),
        "atm_strike": brief["focus"].get("atm_strike"),
        "blocking_reasons": brief["risk"].get("blocking_reasons", []),
        "risk_flags": brief["risk"].get("risk_flags", []),
        "stale_inputs_count": len(brief["risk"].get("stale_inputs", [])),
        "warnings_count": len(warnings),
        "artifacts": brief["artifacts"],
    }


def build_journal_entry() -> dict[str, Any]:
    controller, monitor, contract, warnings = load_inputs()
    brief = build_brief_payload(controller, monitor, contract, warnings)
    return build_journal_entry_payload(brief, controller, monitor, contract, warnings)


def append_journal_entry(entry: dict[str, Any]) -> bool:
    OUT_JOURNAL_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JOURNAL_JSONL.exists():
        lines = [
            line.strip()
            for line in OUT_JOURNAL_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            try:
                last_entry = json.loads(lines[-1])
            except json.JSONDecodeError:
                last_entry = {}
            if last_entry.get("entry_id") == entry.get("entry_id"):
                return False
    with OUT_JOURNAL_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return True


def render_text(brief: dict[str, Any]) -> str:
    summary = brief["summary"]
    focus = brief["focus"]
    risk = brief["risk"]
    return "\n".join(
        [
            "SHARPEDGE OPERATOR BRIEF",
            f"Created: {brief['created_ts']}",
            f"Symbol: {brief['symbol']}",
            f"Action: {brief['operator_action']}",
            f"Headline: {brief['headline']}",
            "",
            f"Contract decision: {summary['contract_decision']}",
            f"Controller decision: {summary['controller_decision']}",
            f"Monitor decision: {summary['monitor_decision']}",
            f"Risk state: {summary['risk_state']}",
            f"Broker integration: {summary['broker_integration_status']} ({summary['monitoring_mode']})",
            "",
            f"Gap direction: {focus['gap_direction']}",
            f"Gap fill level: {focus['gap_fill_level']}",
            f"Option side watch: {focus['option_side_watch']}",
            f"Spot / ATM: {focus['spot']} / {focus['atm_strike']}",
            f"Dealer state: {focus['dealer_state_hint']}",
            "",
            f"Blocking reasons: {', '.join(risk['blocking_reasons']) or 'none'}",
            f"Risk flags: {', '.join(risk['risk_flags']) or 'none'}",
            "",
            "Next steps:",
            *[f"- {step}" for step in brief["next_steps"]],
        ]
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    controller, monitor, contract, warnings = load_inputs()
    brief = build_brief_payload(controller, monitor, contract, warnings)
    watchlist = build_watchlist_payload(brief)
    journal_entry = build_journal_entry_payload(
        brief, controller, monitor, contract, warnings
    )
    journal_appended = append_journal_entry(journal_entry)

    OUT_JSON.write_text(json.dumps(brief, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(brief), encoding="utf-8")
    OUT_WATCHLIST_JSON.write_text(
        json.dumps(watchlist, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(brief, indent=2, sort_keys=True))
    print(f"operator_brief_action={brief['operator_action']}")
    print(f"operator_watchlist_items={len(watchlist['items'])}")
    print(f"operator_journal_appended={journal_appended}")


if __name__ == "__main__":
    main()
