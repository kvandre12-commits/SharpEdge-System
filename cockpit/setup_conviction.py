from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim
from gate_workflows import gate_metadata, primary_context_setup, primary_trade_setup


def _clamp(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def _bias_label(setup: dict[str, Any] | None) -> str:
    text = str((setup or {}).get("bias", "")).upper()
    if "CALL" in text or "BULL" in text or "RECLAIM" in text:
        return "CALLS"
    if "PUT" in text or "BEAR" in text or "REJECT" in text:
        return "PUTS"
    return "NEUTRAL"


def _base_score(tag: str, actionable: bool) -> int:
    mapping = {
        "FAILED BREAKDOWN": 88,
        "FAILED BREAKOUT": 88,
        "DOWNSIDE EXHAUSTION": 68,
        "UPSIDE EXHAUSTION": 68,
        "EXHAUSTION -> RUNNER HANDOFF": 78,
        "POST-SELLOFF COIL": 74,
        "STICKY DAY (CALM/CHOP)": 52,
        "RUNNER DAY (WHEEE)": 52,
    }
    if tag in mapping:
        return mapping[tag]
    return 58 if actionable else 45


def _score_setup(setup: dict[str, Any] | None) -> int:
    if not setup:
        return 22
    meta = gate_metadata(setup)
    tag = str(meta.get("tag") or "").upper()
    score = float(_base_score(tag, bool(meta.get("actionable"))))
    bars_ago = setup.get("bars_ago")
    if isinstance(bars_ago, int):
        score -= min(max(bars_ago, 0) * 3, 15)
    if setup.get("level_name") and isinstance(setup.get("level_price"), (int, float)):
        score += 4
    if setup.get("trigger_price"):
        score += 2
    return _clamp(score)


def _gate_for(score: int, actionable: bool, has_setup: bool) -> str:
    if not has_setup:
        return "NONE"
    if actionable and score >= 72:
        return "ACTIONABLE"
    if actionable and score >= 58:
        return "EMERGING"
    if not actionable:
        return "CONTEXT"
    return "WATCH"


def _fresh_setup_evidence(
    entry_setup: dict[str, Any] | None,
    context_setup: dict[str, Any] | None,
) -> dict[str, Any]:
    chosen = entry_setup or context_setup or {}
    meta = gate_metadata(chosen)
    status = (
        "fresh_actionable_setup"
        if chosen and bool(meta.get("actionable"))
        else "fresh_context_setup"
        if chosen
        else "none"
    )
    return {
        "source": "current_setups",
        "status": status,
        "setup_tag": chosen.get("tag"),
        "gate_id": meta.get("gate_id"),
        "actionable": bool(meta.get("actionable")),
        "bars_ago": chosen.get("bars_ago"),
        "level_name": chosen.get("level_name"),
        "level_price": chosen.get("level_price"),
    }


def _persisted_setup_thesis() -> dict[str, Any]:
    return {
        "source": "setup_event_lifecycle",
        "active": False,
        "setup_tag": None,
        "event_status": None,
        "persisted_without_fresh_trigger": False,
        "first_seen_ts": None,
        "last_seen_ts": None,
        "last_confirmed_ts": None,
        "observation_count": None,
    }


def _live_trap_corroboration(parts: dict[str, Any] | None = None) -> dict[str, Any]:
    parts = parts or {}
    trap = parts.get("trap_score")
    rejection = parts.get("rejection_score")
    return {
        "source": "execution_vectors",
        "trap_score": int(getattr(trap, "score", 0) or 0),
        "trap_bias": prim.bias_label(getattr(trap, "bias", 0) or 0),
        "trap_reason": str(getattr(trap, "reason", "") or ""),
        "rejection_score": int(getattr(rejection, "score", 0) or 0),
        "rejection_bias": prim.bias_label(getattr(rejection, "bias", 0) or 0),
        "rejection_reason": str(getattr(rejection, "reason", "") or ""),
    }


def sync_setup_evidence_fields(card: dict[str, Any]) -> dict[str, Any]:
    setup_conviction = (card or {}).get("setup_conviction") or {}
    card["fresh_setup_evidence"] = dict(
        setup_conviction.get("fresh_setup_evidence") or {}
    )
    card["persisted_setup_thesis"] = dict(
        setup_conviction.get("persisted_setup_thesis") or {}
    )
    card["live_trap_corroboration"] = dict(
        setup_conviction.get("live_trap_corroboration") or {}
    )
    return card


def build_setup_conviction(
    setups: list[dict[str, Any]] | None = None,
    corroboration_parts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return setup identity metadata without re-scoring authority.

    Contract doctrine:
    - `fresh_setup_evidence` is the canonical current setup-card identity from
      `detect_failed_breaks()` and other setup detectors.
    - `persisted_setup_thesis` is lifecycle metadata and may describe a carried
      setup thesis after fresh setup evidence has gone stale.
    - `live_trap_corroboration` is present-tense vector corroboration from
      `_score_trap()` / `_score_rejection()`, not canonical setup identity.
    """
    entry_setup = primary_trade_setup(setups)
    context_setup = primary_context_setup(setups)
    has_setup = bool(entry_setup or context_setup)
    chosen = entry_setup or context_setup or {}
    meta = gate_metadata(chosen)
    actionable = bool(meta.get("actionable"))
    score = _score_setup(chosen)
    reason = ""
    if chosen:
        reason = str(
            chosen.get("detail") or chosen.get("bias") or chosen.get("tag") or ""
        )
    if not reason:
        reason = "no active setup card"
    return {
        "schema": "sharpedge.setup_conviction.v1",
        "setup_conviction_score": score,
        "setup_gate": _gate_for(score, actionable, has_setup),
        "bias": _bias_label(chosen),
        "setup_tag": chosen.get("tag"),
        "reason": reason,
        "entry_gate": gate_metadata(entry_setup),
        "context_gate": gate_metadata(context_setup),
        "fresh_setup_evidence": _fresh_setup_evidence(entry_setup, context_setup),
        "persisted_setup_thesis": _persisted_setup_thesis(),
        "live_trap_corroboration": _live_trap_corroboration(corroboration_parts),
    }


__all__ = ["build_setup_conviction", "sync_setup_evidence_fields"]
