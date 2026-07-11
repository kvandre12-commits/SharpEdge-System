from __future__ import annotations

from typing import Any


def canonical_option_side(*candidates: Any) -> str:
    for candidate in candidates:
        text = str(candidate or "").strip().upper()
        if not text:
            continue
        if "PUT" in text:
            return "PUTS"
        if "CALL" in text:
            return "CALLS"
    return "NEUTRAL"


def tactical_invalidation_reason(brief: dict[str, Any]) -> str:
    execution = brief.get("execution_logic") or {}
    trend = brief.get("permission_score_trend") or {}

    trade_gate = str(execution.get("trade_gate") or "").upper()
    execution_score = execution.get("execution_permission_score")
    setup_gate = str(execution.get("setup_gate") or "").upper()
    setup_side = canonical_option_side(
        execution.get("setup_bias"),
        execution.get("bias"),
        brief.get("focus", {}).get("option_side_watch"),
    )
    direction = str(trend.get("direction") or "").lower()
    delta = trend.get("delta")
    transitions = trend.get("setup_transitions_since_last_update") or []

    if brief.get("operator_action") == "stand_down":
        return "operator_action_stand_down"
    if trade_gate == "BLOCK":
        return "trade_gate_blocked"
    if setup_gate and setup_gate != "ACTIONABLE":
        return f"setup_gate_{setup_gate.lower()}"
    if not isinstance(execution_score, (int, float)):
        return "execution_permission_missing"
    if execution_score < 60:
        return f"execution_permission_below_threshold_{int(execution_score)}"
    if setup_side == "NEUTRAL":
        return "setup_side_unclear"
    if direction == "weakening" and isinstance(delta, (int, float)) and delta <= -5:
        return f"permission_score_trend_weakening_{int(delta)}"
    for transition in transitions:
        label = str(transition.get("label") or "").upper()
        if "EXPIRED" in label:
            return "setup_transition_expired"
    return ""


def build_watchlist_derivatives(
    brief: dict[str, Any],
    *,
    base_status: str,
    base_priority: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    focus = brief.get("focus") or {}
    execution = brief.get("execution_logic") or {}
    trend = brief.get("permission_score_trend") or {}

    if base_status == "blocked":
        return [], []

    derivatives: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    thesis_side = canonical_option_side(
        focus.get("option_side_watch"),
        execution.get("setup_bias"),
        execution.get("bias"),
    )
    if thesis_side != "NEUTRAL":
        derivatives.append(
            {
                "item_id": (
                    f"{brief['symbol']}-atm-{thesis_side.lower()}-21dte-"
                    f"{focus.get('gap_session_date') or 'na'}"
                ),
                "symbol": brief["symbol"],
                "setup_type": "atm_options_thesis",
                "watchlist_family": "swing",
                "status": base_status,
                "priority": base_priority,
                "operator_action": brief["operator_action"],
                "headline": (
                    f"21 DTE ATM {thesis_side} watch for "
                    f"{focus.get('fill_bias') or focus.get('option_side_watch') or 'current thesis'}"
                ),
                "option_side": thesis_side,
                "dte_target": 21,
                "strike_selection": "ATM",
                "gap_session_date": focus.get("gap_session_date"),
                "gap_direction": focus.get("gap_direction"),
                "gap_fill_level": focus.get("gap_fill_level"),
                "fill_bias": focus.get("fill_bias"),
                "option_side_watch": focus.get("option_side_watch"),
                "spot": focus.get("spot"),
                "atm_strike": focus.get("atm_strike"),
                "dealer_state_hint": focus.get("dealer_state_hint"),
                "broker_integration_status": brief.get("summary", {}).get(
                    "broker_integration_status"
                ),
                "trade_permission_score": execution.get("trade_permission_score"),
                "execution_permission_score": execution.get(
                    "execution_permission_score"
                ),
                "permission_trend_direction": trend.get("direction"),
                "permission_trend_delta": trend.get("delta"),
                "setup_tag": execution.get("setup_tag"),
                "trade_gate": execution.get("trade_gate"),
                "blocking_reasons": brief.get("risk", {}).get("blocking_reasons", []),
                "risk_flags": brief.get("risk", {}).get("risk_flags", []),
                "stale_inputs_count": len(
                    brief.get("risk", {}).get("stale_inputs", [])
                ),
            }
        )

    tactical_side = canonical_option_side(
        execution.get("setup_bias"),
        execution.get("bias"),
        focus.get("option_side_watch"),
    )
    tactical_reason = tactical_invalidation_reason(brief)
    tactical_candidate = {
        "item_id": (
            f"{brief['symbol']}-atm-{tactical_side.lower()}-1dte-"
            f"{focus.get('gap_session_date') or 'na'}"
        ),
        "symbol": brief["symbol"],
        "setup_type": "atm_options_execution",
        "watchlist_family": "tactical",
        "status": base_status,
        "priority": "high" if base_priority == "high" else "medium",
        "operator_action": brief["operator_action"],
        "headline": (
            f"1 DTE ATM {tactical_side} execution watch for "
            f"{execution.get('setup_tag') or focus.get('option_side_watch') or 'current setup'}"
        ),
        "option_side": tactical_side,
        "dte_target": 1,
        "strike_selection": "ATM",
        "gap_session_date": focus.get("gap_session_date"),
        "gap_direction": focus.get("gap_direction"),
        "gap_fill_level": focus.get("gap_fill_level"),
        "fill_bias": focus.get("fill_bias"),
        "spot": focus.get("spot"),
        "atm_strike": focus.get("atm_strike"),
        "dealer_state_hint": focus.get("dealer_state_hint"),
        "trade_permission_score": execution.get("trade_permission_score"),
        "execution_permission_score": execution.get("execution_permission_score"),
        "permission_trend_direction": trend.get("direction"),
        "permission_trend_delta": trend.get("delta"),
        "trade_gate": execution.get("trade_gate"),
        "setup_gate": execution.get("setup_gate"),
        "setup_tag": execution.get("setup_tag"),
        "blocking_reasons": brief.get("risk", {}).get("blocking_reasons", []),
        "risk_flags": brief.get("risk", {}).get("risk_flags", []),
        "stale_inputs_count": len(brief.get("risk", {}).get("stale_inputs", [])),
    }
    if tactical_reason:
        omitted.append(
            {
                **tactical_candidate,
                "status": "removed",
                "invalidation_reason": tactical_reason,
            }
        )
    else:
        derivatives.append(tactical_candidate)
    return derivatives, omitted
