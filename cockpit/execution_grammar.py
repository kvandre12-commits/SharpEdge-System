"""Execution interpretation + permission governance for SharpEdge.

Current seam:
- Interpretation: break/dealer states resolve into thesis, bias, and authority.
- Governance: live trigger readiness constrains permission with caps/floors.

Keep both here while they evolve together. Split only when interpreter doctrine
or permission-governor doctrine starts changing independently.
"""

from __future__ import annotations

from typing import Any

from accepted_break_interpreter import (
    accepted_break_break_state,
    best_accepted_break_event,
)
from dealer_state_engine import build_dealer_state
from execution_state_scores import score_dealer_state
from failed_break_facts import RESISTANCE_LEVEL_NAMES, SUPPORT_LEVEL_NAMES
from failed_break_interpreter import best_failed_break_event, failed_break_break_state
from level_state_engine import build_level_state_map
from live_trigger_check import live_trigger_check
from trade_permission_context import BEARISH, BULLISH

ACTIVE_RESISTANCE_LEVELS = RESISTANCE_LEVEL_NAMES
ACTIVE_SUPPORT_LEVELS = SUPPORT_LEVEL_NAMES
RECENT_BARS = 6
ACCEPTANCE_CLOSES = 3
WALL_PROXIMITY_PCT = 0.20
PIN_PROXIMITY_PCT = 0.25


def _bias_label(bias: int) -> str:
    if bias == BULLISH:
        return "CALLS"
    if bias == BEARISH:
        return "PUTS"
    return "NEUTRAL"


def _active_levels(levels: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in (levels or {}).items()
        if name in {*ACTIVE_RESISTANCE_LEVELS, *ACTIVE_SUPPORT_LEVELS}
        and isinstance(value, (int, float))
    }


def _accepted_break(level_states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    event = best_accepted_break_event(
        level_states,
        level_order=tuple((level_states or {}).keys()),
        acceptance_closes=ACCEPTANCE_CLOSES,
    )
    return accepted_break_break_state(event) if event else {}


def _failed_break(level_states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Infer failed-break state from the shared level-state engine."""
    event = best_failed_break_event(
        level_states,
        level_order=tuple((level_states or {}).keys()),
        recent_bars=RECENT_BARS,
    )
    return failed_break_break_state(event) if event else {}


def _setup_break_state(setups: list[dict[str, Any]] | None) -> dict[str, Any]:
    for setup in setups or []:
        tag = str(setup.get("tag") or "").upper()
        if tag == "FAILED BREAKOUT":
            return {
                "state": "failed_breakout",
                "bias": "PUTS",
                "level_name": setup.get("level_name"),
                "level_price": setup.get("level_price"),
                "trigger_price": setup.get("trigger_price"),
                "score": 88,
                "reason": setup.get("detail") or "failed breakout pressure point",
                "source": "setup_card",
            }
        if tag == "FAILED BREAKDOWN":
            return {
                "state": "failed_breakdown",
                "bias": "CALLS",
                "level_name": setup.get("level_name"),
                "level_price": setup.get("level_price"),
                "trigger_price": setup.get("trigger_price"),
                "score": 88,
                "reason": setup.get("detail") or "failed breakdown pressure point",
                "source": "setup_card",
            }
    return {}


def build_break_state(
    bars: list[tuple],
    levels: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    setup_state = _setup_break_state(setups)
    if setup_state:
        return setup_state
    clean_levels = _active_levels(levels)
    if not bars or not clean_levels:
        return {
            "state": "no_break_read",
            "bias": "NEUTRAL",
            "score": 35,
            "reason": "no active break levels available",
        }
    level_states = build_level_state_map(
        bars,
        clean_levels,
        level_names=tuple(clean_levels.keys()),
        recent_window=RECENT_BARS,
        acceptance_window=ACCEPTANCE_CLOSES,
    )
    failed = _failed_break(level_states)
    accepted = _accepted_break(level_states)
    chosen = failed or accepted
    if chosen:
        bias = chosen.get("bias", "NEUTRAL")
        if isinstance(bias, int):
            bias = _bias_label(bias)
        return {**chosen, "bias": bias}
    spot = float(bars[-1][4])
    name, level = min(clean_levels.items(), key=lambda item: abs(spot - item[1]))
    return {
        "state": "no_active_break",
        "bias": "NEUTRAL",
        "level_name": name,
        "level_price": level,
        "score": 42,
        "reason": f"no accepted or failed break; nearest pressure level {name} {level:.2f}",
    }


def build_dealer_gamma_state(
    pa: dict[str, Any], op: dict[str, Any], gp: dict[str, Any]
) -> dict[str, Any]:
    dealer = build_dealer_state(pa, op, gp)
    dealer_score = score_dealer_state(dealer)
    pin_state = dealer.get("pin_state") or {}
    wall_state = dealer.get("wall_state") or {}
    gamma_state = dealer.get("gamma_state") or {}
    return {
        "state": str(dealer.get("state") or "dealer_unknown"),
        "bias": str(dealer.get("bias") or "NEUTRAL"),
        "score": int(dealer_score.score),
        "regime": gamma_state.get("regime"),
        "pin": pin_state.get("pin"),
        "pin_dist_pct": pin_state.get("pin_dist_pct"),
        "call_wall": wall_state.get("call_wall"),
        "put_wall": wall_state.get("put_wall"),
        "reason": str(dealer_score.reason),
    }


def _proof_state(parts: dict[str, Any]) -> dict[str, Any]:
    volume = parts.get("volume_score")
    pressure = parts.get("pressure_score")
    evidence = []
    if volume:
        evidence.append(
            f"volume {volume.score}/{_bias_label(volume.bias)}: {volume.reason}"
        )
    if pressure:
        evidence.append(
            f"pressure {pressure.score}/{_bias_label(pressure.bias)}: {pressure.reason}"
        )
    return {"evidence": evidence}


def build_execution_grammar(
    bars: list[tuple],
    pa: dict[str, Any],
    levels: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
    parts: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
    day_bucket: dict[str, Any] | None = None,
) -> dict[str, Any]:
    break_state = build_break_state(bars, levels, setups)
    dealer = build_dealer_gamma_state(pa, op, gp)
    proof = _proof_state(parts)
    break_name = break_state["state"]
    dealer_name = dealer["state"]
    adjustment: dict[str, Any] = {
        "cap": None,
        "floor": None,
        "reason": "no grammar adjustment",
    }
    thesis = "wait_for_pressure_point"
    bias = break_state.get("bias", "NEUTRAL")
    authority = "context"

    if break_name == "failed_breakout":
        thesis = "failed_breakout_reversal"
        bias = "PUTS"
        authority = "primary"
    elif break_name == "failed_breakdown":
        thesis = "failed_breakdown_reclaim"
        bias = "CALLS"
        authority = "primary"
    elif (
        break_name == "accepted_breakout" and dealer_name == "negative_gamma_expansion"
    ):
        thesis = "accepted_breakout_runner"
        bias = "CALLS"
        authority = "primary"
        adjustment = {
            "cap": None,
            "floor": 72,
            "reason": "negative gamma lets accepted breakout run",
        }
    elif (
        break_name == "accepted_breakdown" and dealer_name == "negative_gamma_expansion"
    ):
        thesis = "accepted_breakdown_runner"
        bias = "PUTS"
        authority = "primary"
        adjustment = {
            "cap": None,
            "floor": 72,
            "reason": "negative gamma lets accepted breakdown run",
        }
    elif break_name == "accepted_breakout" and dealer_name == "positive_gamma_gravity":
        thesis = "breakout_into_dealer_resistance"
        bias = "NEUTRAL"
        authority = "governor"
        adjustment = {
            "cap": 68,
            "floor": None,
            "reason": "accepted breakout is pressing into positive-gamma pin/wall gravity",
        }
    elif break_name == "accepted_breakdown" and dealer_name == "positive_gamma_gravity":
        thesis = "breakdown_into_dealer_support"
        bias = "NEUTRAL"
        authority = "governor"
        adjustment = {
            "cap": 68,
            "floor": None,
            "reason": "accepted breakdown is pressing into positive-gamma pin/wall gravity",
        }
    elif dealer_name == "positive_gamma_gravity":
        thesis = "pin_chop_wait_for_failed_break"
        bias = dealer.get("bias", "NEUTRAL")
        authority = "governor"
        adjustment = {
            "cap": 70,
            "floor": None,
            "reason": "positive gamma gravity requires a pressure-point trigger",
        }

    live_trigger = live_trigger_check(thesis, day_bucket, pa, dealer, levels)
    if live_trigger["status"] == "WAIT":
        existing_cap = adjustment.get("cap")
        adjustment = {
            **adjustment,
            "cap": min(existing_cap, 68) if isinstance(existing_cap, int) else 68,
            "floor": None,
            "reason": live_trigger["reason"],
        }
    elif live_trigger["status"] == "CONTEXT_MATCH":
        existing_cap = adjustment.get("cap")
        adjustment = {
            **adjustment,
            "cap": min(existing_cap, 70) if isinstance(existing_cap, int) else 70,
            "floor": None,
            "reason": live_trigger["reason"],
        }

    return {
        "schema": "sharpedge.execution_grammar.v1",
        "thesis": thesis,
        "bias": bias,
        "authority": authority,
        "day_bucket": day_bucket or {},
        "break_state": break_state,
        "dealer_gamma_state": dealer,
        "proof_state": proof,
        "live_trigger_check": live_trigger,
        "permission_adjustment": adjustment,
    }


def apply_permission_adjustment(permission: int, grammar: dict[str, Any]) -> int:
    adjustment = grammar.get("permission_adjustment") or {}
    cap = adjustment.get("cap")
    floor = adjustment.get("floor")
    result = permission
    if isinstance(cap, int):
        result = min(result, cap)
    if isinstance(floor, int):
        result = max(result, floor)
    return max(0, min(100, int(result)))


__all__ = [
    "apply_permission_adjustment",
    "build_break_state",
    "build_dealer_gamma_state",
    "build_execution_grammar",
]
