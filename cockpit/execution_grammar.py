"""Execution interpretation for SharpEdge.

Resolves raw bars/levels/setups into break and dealer-gamma states. Permission
governance now lives in the bucket-conditioned spine, so this module is
interpretation only.
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
from trade_permission_context import BEARISH, BULLISH

ACTIVE_RESISTANCE_LEVELS = RESISTANCE_LEVEL_NAMES
ACTIVE_SUPPORT_LEVELS = SUPPORT_LEVEL_NAMES
RECENT_BARS = 6
ACCEPTANCE_CLOSES = 3


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


__all__ = [
    "build_break_state",
    "build_dealer_gamma_state",
]
