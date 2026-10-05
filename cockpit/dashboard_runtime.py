"""Local-first runtime helpers for cockpit dashboards.

These pages should keep rendering even when the Robinhood Bridge repo is absent.
Use local SharpEdge artifacts first, then optionally enrich with Bridge logic when
that repo is available on the machine.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
_BRIDGE_SRC = Path(os.path.expanduser("~/SharpEdge-Robinhood-Bridge/src"))
if _BRIDGE_SRC.exists() and str(_BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_SRC))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def local_signal() -> dict[str, Any]:
    return _load_json(OUTPUTS / "signal.json")


def local_scoreboard() -> dict[str, Any]:
    return _load_json(OUTPUTS / "reconcile_summary.json")


def unavailable_context(note: str) -> SimpleNamespace:
    return SimpleNamespace(
        available=False,
        note=note,
        final_bias=None,
        fresh=False,
        age_days=None,
        prob_trend=None,
        prob_range=None,
        execution_score=None,
    )


def resolve_execution_context(symbol: str = "SPY") -> SimpleNamespace:
    try:
        from sharpedge_robinhood_bridge.analytics_context import load_execution_state
    except ImportError:
        return unavailable_context("Robinhood Bridge unavailable; using local artifacts only")
    try:
        return load_execution_state(symbol=symbol)
    except Exception as exc:  # pragma: no cover - defensive dashboard fallback
        return unavailable_context(f"Bridge analytics unavailable: {exc}")


def resolve_decide() -> Callable[[dict[str, Any], Any], dict[str, Any]] | None:
    try:
        from sharpedge_robinhood_bridge.trade_intent import decide
    except ImportError:
        return None
    return decide


def resolve_decision(signal: dict[str, Any], ctx: Any) -> dict[str, Any]:
    if not signal:
        return {"action": "stand_down", "reason": "no signal", "intent": None}
    decide = resolve_decide()
    if decide is None:
        reason = signal.get("setup_bias") or "bridge unavailable; signal-only dashboard mode"
        return {"action": "stand_down", "reason": reason, "intent": None}
    try:
        return decide(signal, analytics=ctx)
    except Exception as exc:  # pragma: no cover - defensive dashboard fallback
        return {"action": "stand_down", "reason": f"decision unavailable: {exc}", "intent": None}


__all__ = [
    "local_scoreboard",
    "local_signal",
    "resolve_decision",
    "resolve_execution_context",
    "unavailable_context",
]
