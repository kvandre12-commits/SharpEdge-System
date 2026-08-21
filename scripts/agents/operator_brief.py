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

from scripts.agents.operator_watchlist_logic import build_watchlist_derivatives

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
TRADE_HINTS_JSON = OUTDIR / "trade_journal_hints.json"
SIGNAL_JSON = OUTDIR / "signal.json"
NERV_CURATOR_JSON = OUTDIR / "nerv_curator.json"
CONNECTOR_AUDIT_JSON = OUTDIR / "chatgpt_robinhood_connector_audit.json"
CONNECTOR_AUDIT_LOG_JSONL = OUTDIR / "robinhood_connector_audit_log.jsonl"


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


def load_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    return (
        read_json(CONTROLLER_JSON),
        read_json(MONITOR_JSON),
        read_json(AGENT_V1_JSON),
        read_warnings(),
        read_json(TRADE_HINTS_JSON),
        read_json(CONNECTOR_AUDIT_JSON),
        read_json(SIGNAL_JSON),
        read_json(NERV_CURATOR_JSON),
    )


def summarize_historical_hints(hints: dict[str, Any]) -> dict[str, Any]:
    if not hints:
        return {"available": False}

    top_pattern = (hints.get("top_patterns") or [{}])[0]
    top_condition = top_pattern.get("condition", {})
    primary_hint = (hints.get("actionable_hints") or [{}])[0]
    sample_state = hints.get("sample_state", {})
    return {
        "available": True,
        "total_trades": sample_state.get("total_trades", 0),
        "closed_trades": sample_state.get("closed_trades", 0),
        "low_sample": sample_state.get("low_sample", True),
        "minimum_pattern_sample_n": sample_state.get("minimum_pattern_sample_n"),
        "top_pattern_summary": primary_hint.get("summary"),
        "top_pattern_condition": top_condition,
        "top_pattern_confidence": top_pattern.get("confidence_label"),
        "metric_collection_priorities": hints.get("metric_collection_priorities", [])[
            :3
        ],
        "usage_constraints": hints.get("usage_constraints", [])[:2],
    }


def summarize_latest_execution_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if not audit or audit.get("schema") != "sharpedge.robinhood_connector_audit.v1":
        return {"available": False}

    requested = audit.get("requested_action") or {}
    observed = audit.get("connector_observation") or {}
    follow_up = audit.get("operator_follow_up") or {}
    return {
        "available": True,
        "created_at": audit.get("created_at"),
        "task_type": requested.get("task_type"),
        "symbol": requested.get("symbol"),
        "connector_status": observed.get("connector_status"),
        "fill_status": observed.get("fill_status"),
        "broker_order_id": observed.get("broker_order_id"),
        "summary": observed.get("summary"),
        "blockers": (observed.get("blockers") or [])[:3],
        "questions": (observed.get("questions") or [])[:3],
        "follow_up_prompts": (follow_up.get("prompts") or [])[:3],
        "source_handoff_path": audit.get("source_handoff_path"),
        "source_response_path": audit.get("source_response_path"),
    }


def summarize_execution_logic(signal: dict[str, Any]) -> dict[str, Any]:
    permission = signal.get("trade_permission") or {}
    setup_conviction = permission.get("setup_conviction") or {}
    if not permission:
        return {"available": False}
    return {
        "available": True,
        "trade_gate": permission.get("trade_gate"),
        "trade_permission_score": permission.get("trade_permission_score"),
        "execution_permission_score": permission.get("execution_permission_score"),
        "bias": permission.get("bias"),
        "setup_gate": setup_conviction.get("setup_gate"),
        "setup_bias": setup_conviction.get("bias"),
        "setup_tag": setup_conviction.get("setup_tag"),
        "entry_workflow": ((setup_conviction.get("entry_gate") or {}).get("workflow")),
    }


def summarize_liquidity_read(curator: dict[str, Any]) -> dict[str, Any]:
    if curator.get("schema") != "sharpedge.nerv_curator.v1":
        return {"available": False}

    summary = curator.get("hey_guy_summary") or {}
    return {
        "available": True,
        "generated_at_utc": curator.get("generated_at_utc"),
        "symbol": curator.get("symbol"),
        "headline": curator.get("headline"),
        "stance": curator.get("stance"),
        "plain_english": summary.get("plain_english"),
        "liquidity_spot": summary.get("liquidity_spot"),
        "flow_balance": summary.get("flow_balance"),
        "bias_alignment": summary.get("bias_alignment"),
        "quote_quality_context": summary.get("quote_quality_context"),
        "put_pressure_score": summary.get("put_pressure_score"),
        "call_pressure_score": summary.get("call_pressure_score"),
        "put_pressure_pct": summary.get("put_pressure_pct"),
        "call_pressure_pct": summary.get("call_pressure_pct"),
        "dominant_side": summary.get("dominant_side"),
        "call_side_summary": summary.get("call_side_summary"),
        "put_side_summary": summary.get("put_side_summary"),
        "call_flow": list(summary.get("call_flow") or [])[:2],
        "put_flow": list(summary.get("put_flow") or [])[:2],
        "near_money_tape": list(summary.get("near_money_tape") or [])[:4],
        "supporting_flow": list(summary.get("supporting_flow") or [])[:2],
        "opposing_flow": list(summary.get("opposing_flow") or [])[:2],
        "confirms": list(summary.get("confirms") or [])[:2],
        "invalidates": list(summary.get("invalidates") or [])[:2],
        "watch_next": list(curator.get("watch_next") or [])[:3],
        "warnings": list(curator.get("warnings") or [])[:2],
    }


def summarize_permission_score_trend(signal: dict[str, Any]) -> dict[str, Any]:
    trend = signal.get("permission_score_trend") or {}
    if not trend or trend.get("schema") != "sharpedge.permission_score_trend.v1":
        return {"available": False}
    return {
        "available": True,
        "current": trend.get("current"),
        "previous": trend.get("previous"),
        "delta": trend.get("delta"),
        "direction": trend.get("direction"),
        "largest_changes_since_last_update": (
            trend.get("largest_changes_since_last_update") or []
        )[:3],
        "setup_transitions_since_last_update": (
            trend.get("setup_transitions_since_last_update") or []
        )[:3],
    }


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
    execution_audit: dict[str, Any],
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
        steps.append(
            "Confirm sample quality, freshness, and risk budget before any manual action."
        )
    elif operator_action == "monitor_only":
        steps.append(
            f"Watch whether price moves toward gap fill level {fill_level} and whether the thesis stays intact."
        )
        steps.append(
            "Do not place orders; use this run as structured observation only."
        )
    else:
        blockers = contract.get("blocking_reasons", [])
        steps.append(
            f"Do nothing until blockers clear: {', '.join(blockers) or 'unknown blocker'}."
        )

    if execution_audit.get("available"):
        connector_status = execution_audit.get("connector_status") or "unknown"
        fill_status = execution_audit.get("fill_status") or "unknown"
        if connector_status == "drafted":
            steps.append(
                "Latest connector result is a draft; confirm contracts, price, and sizing before any manual submit."
            )
        elif connector_status in {"submitted", "replaced"}:
            steps.append(
                f"Latest connector result says the order was {connector_status}; verify live broker status and remaining quantity ({fill_status})."
            )
        elif connector_status == "filled":
            steps.append(
                f"Latest connector result says the order filled; capture actual fill details and slippage ({fill_status})."
            )
        elif connector_status == "blocked":
            blockers = execution_audit.get("blockers") or []
            steps.append(
                f"Latest connector attempt was blocked: {', '.join(blockers) or 'review the connector audit artifact for blockers'}."
            )
        elif execution_audit.get("summary"):
            steps.append(f"Latest connector outcome: {execution_audit['summary']}")

        prompts = execution_audit.get("follow_up_prompts") or []
        if prompts:
            steps.append(prompts[0])

    if bridge_status != "ready":
        steps.append(
            "Broker integration is not live; rely on artifact review and manual platform checks."
        )
    if warnings:
        steps.append(
            f"Pipeline emitted {len(warnings)} warning(s); inspect outputs/health/warnings.log."
        )
    steps.append(
        "Orders remain blocked by design unless manually confirmed outside the automation loop."
    )
    return steps[:6]


def build_brief_payload(
    controller: dict[str, Any],
    monitor: dict[str, Any],
    contract: dict[str, Any],
    warnings: list[str],
    hints: dict[str, Any],
    connector_audit: dict[str, Any],
    signal: dict[str, Any],
    curator: dict[str, Any],
) -> dict[str, Any]:
    operator_action = choose_operator_action(contract)
    gap = monitor.get("latest_gap_event", {})
    hypothesis = monitor.get("directional_hypothesis", {})
    options = monitor.get("options_context", {})
    risk = monitor.get("risk_context", {})
    stale = contract.get("freshness", {}).get("stale_inputs", [])
    latest_execution_audit = summarize_latest_execution_audit(connector_audit)
    execution_logic = summarize_execution_logic(signal)
    permission_score_trend = summarize_permission_score_trend(signal)
    liquidity_read = summarize_liquidity_read(curator)
    artifacts = {
        "controller": str(CONTROLLER_JSON),
        "monitor": str(MONITOR_JSON),
        "contract": str(AGENT_V1_JSON),
    }
    if signal:
        artifacts["signal"] = str(SIGNAL_JSON)
    if latest_execution_audit.get("available"):
        artifacts["connector_audit"] = str(CONNECTOR_AUDIT_JSON)
        if CONNECTOR_AUDIT_LOG_JSONL.exists():
            artifacts["connector_audit_log"] = str(CONNECTOR_AUDIT_LOG_JSONL)
    if liquidity_read.get("available"):
        artifacts["nerv_curator"] = str(NERV_CURATOR_JSON)

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
        "historical_hints": summarize_historical_hints(hints),
        "execution_logic": execution_logic,
        "permission_score_trend": permission_score_trend,
        "options_liquidity_read": liquidity_read,
        "latest_execution_audit": latest_execution_audit,
        "next_steps": build_next_steps(
            operator_action,
            contract,
            monitor,
            warnings,
            latest_execution_audit,
        ),
        "artifacts": artifacts,
    }


def build_brief() -> dict[str, Any]:
    controller, monitor, contract, warnings, hints, connector_audit, signal, curator = (
        load_inputs()
    )
    return build_brief_payload(
        controller,
        monitor,
        contract,
        warnings,
        hints,
        connector_audit,
        signal,
        curator,
    )


def build_watchlist_payload(brief: dict[str, Any]) -> dict[str, Any]:
    status = watchlist_status(brief["operator_action"])
    priority = watchlist_priority(brief["operator_action"])
    item = {
        "item_id": (
            f"{brief['symbol']}-gap-fill-"
            f"{brief['focus'].get('gap_session_date') or 'na'}-"
            f"{str(brief['focus'].get('gap_direction', 'na')).lower()}"
        ),
        "symbol": brief["symbol"],
        "setup_type": "gap_fill_options_context",
        "status": status,
        "priority": priority,
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
        "trade_permission_score": (brief.get("execution_logic") or {}).get(
            "trade_permission_score"
        ),
        "execution_permission_score": (brief.get("execution_logic") or {}).get(
            "execution_permission_score"
        ),
        "permission_trend_direction": (brief.get("permission_score_trend") or {}).get(
            "direction"
        ),
        "permission_trend_delta": (brief.get("permission_score_trend") or {}).get(
            "delta"
        ),
    }
    derivatives, omitted = build_watchlist_derivatives(
        brief,
        base_status=status,
        base_priority=priority,
    )
    items = [item, *derivatives]
    active_count = sum(1 for candidate in items if candidate.get("status") != "blocked")
    return {
        "schema_version": "operator_watchlist.v1",
        "created_ts": utc_now(),
        "symbol": brief["symbol"],
        "active_count": active_count,
        "items": items,
        "omitted_candidates": omitted,
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
    execution_audit = brief.get("latest_execution_audit", {})
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
        "broker_integration_status": brief["summary"].get("broker_integration_status"),
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
        "connector_audit_available": execution_audit.get("available", False),
        "connector_status": execution_audit.get("connector_status"),
        "connector_fill_status": execution_audit.get("fill_status"),
        "connector_broker_order_id": execution_audit.get("broker_order_id"),
        "artifacts": brief["artifacts"],
    }


def build_journal_entry() -> dict[str, Any]:
    controller, monitor, contract, warnings, hints, connector_audit, signal, curator = (
        load_inputs()
    )
    brief = build_brief_payload(
        controller,
        monitor,
        contract,
        warnings,
        hints,
        connector_audit,
        signal,
        curator,
    )
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
    historical = brief.get("historical_hints", {})
    execution_logic = brief.get("execution_logic", {})
    permission_trend = brief.get("permission_score_trend", {})
    liquidity_read = brief.get("options_liquidity_read", {})
    execution_audit = brief.get("latest_execution_audit", {})
    lines = [
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
    ]
    if execution_logic.get("available"):
        lines.extend(
            [
                "",
                "Execution logic:",
                f"- Trade gate / score: {execution_logic.get('trade_gate') or 'unknown'} / {execution_logic.get('execution_permission_score')}",
                f"- Setup gate / tag: {execution_logic.get('setup_gate') or 'unknown'} / {execution_logic.get('setup_tag') or 'none'}",
            ]
        )
    if permission_trend.get("available"):
        lines.extend(
            [
                "",
                "Permission score trend:",
                f"- Direction / delta: {permission_trend.get('direction') or 'unknown'} / {permission_trend.get('delta')}",
                f"- Current / previous: {permission_trend.get('current')} / {permission_trend.get('previous')}",
            ]
        )
    if historical.get("available"):
        lines.extend(
            [
                "",
                "Historical hints:",
                f"- Top pattern: {historical.get('top_pattern_summary') or 'none'}",
                f"- Low sample: {historical.get('low_sample')}",
            ]
        )
    if liquidity_read.get("available"):
        lines.extend(
            [
                "",
                "Options liquidity read:",
                f"- Stance: {liquidity_read.get('stance') or 'unknown'}",
                f"- Read: {liquidity_read.get('plain_english') or 'none'}",
                f"- Liquidity spot: {liquidity_read.get('liquidity_spot') or 'none'}",
                f"- Flow balance: {liquidity_read.get('flow_balance') or 'none'}",
                f"- Bias alignment: {liquidity_read.get('bias_alignment') or 'unknown'}",
                f"- Quote quality: {liquidity_read.get('quote_quality_context') or 'unknown'}",
            ]
        )
        put_flow = liquidity_read.get("put_flow") or []
        if put_flow:
            lines.append(f"- Put side: {' | '.join(put_flow)}")
        call_flow = liquidity_read.get("call_flow") or []
        if call_flow:
            lines.append(f"- Call side: {' | '.join(call_flow)}")
        if liquidity_read.get("put_side_summary"):
            lines.append(f"- Put summary: {liquidity_read['put_side_summary']}")
        if liquidity_read.get("call_side_summary"):
            lines.append(f"- Call summary: {liquidity_read['call_side_summary']}")
        watch_next = liquidity_read.get("watch_next") or []
        if watch_next:
            lines.append(f"- Watch next: {' | '.join(watch_next[:2])}")
    if execution_audit.get("available"):
        lines.extend(
            [
                "",
                "Latest execution audit:",
                f"- Connector status: {execution_audit.get('connector_status') or 'unknown'}",
                f"- Fill status: {execution_audit.get('fill_status') or 'unknown'}",
                f"- Broker order id: {execution_audit.get('broker_order_id') or 'none'}",
                f"- Summary: {execution_audit.get('summary') or 'none'}",
            ]
        )
    lines.extend(["", "Next steps:", *[f"- {step}" for step in brief["next_steps"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    controller, monitor, contract, warnings, hints, connector_audit, signal, curator = (
        load_inputs()
    )
    brief = build_brief_payload(
        controller,
        monitor,
        contract,
        warnings,
        hints,
        connector_audit,
        signal,
        curator,
    )
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
