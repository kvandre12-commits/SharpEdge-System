"""Public trade-permission wrapper for the SharpEdge cockpit."""

from __future__ import annotations

import os
from typing import Any

from ace_authority import build_ace_authority_card, normalize_authority_engine
from execution_vector_engine import ExecutionVectorEngine

AUTHORITY_ENGINE_ENV_VAR = "SHARPEDGE_AUTHORITY_ENGINE"


def resolve_authority_engine(authority_engine: str | None = None) -> str:
    return normalize_authority_engine(
        authority_engine or os.getenv(AUTHORITY_ENGINE_ENV_VAR)
    )


def _ace_advisory_voice(card: dict[str, Any]) -> dict[str, Any]:
    authority = card.get("authority_summary") or {}
    adjudication = card.get("authority_adjudication") or {}
    doing = adjudication.get("we_are_doing_this") or {}
    stance = (
        f"{doing.get('gate') or authority.get('gate') or 'BLOCK'} / "
        f"{doing.get('bias') or authority.get('bias') or 'NEUTRAL'} / "
        f"{doing.get('action') or authority.get('recommended_action') or 'watch_only'}"
    )
    return {
        "voice_id": "ace_advisory",
        "label": "Ace advisory lane",
        "stance": stance,
        "summary": str(
            adjudication.get("summary")
            or authority.get("bucket")
            or "Ace advisory packet unavailable"
        ),
        "score": authority.get("score"),
        "bias": authority.get("bias") or doing.get("bias") or "NEUTRAL",
        "source": "ace_authority",
        "advisory_only": True,
        "engine": "ace",
    }


def _attach_ace_advisory_voice(
    legacy_card: dict[str, Any],
    ace_card: dict[str, Any],
) -> dict[str, Any]:
    adjudication = legacy_card.setdefault("authority_adjudication", {})
    voices = list(adjudication.get("competing_voices") or [])
    voices.append(_ace_advisory_voice(ace_card))
    adjudication["competing_voices"] = voices
    advisory_engines = list(adjudication.get("advisory_engines") or [])
    advisory_engines.append(
        {
            "engine": "ace",
            "mode": str(ace_card.get("authority_mode") or "core_spine_only"),
            "score": (ace_card.get("authority_summary") or {}).get("score"),
            "bias": (ace_card.get("authority_summary") or {}).get("bias"),
            "action": (ace_card.get("authority_summary") or {}).get(
                "recommended_action"
            ),
        }
    )
    adjudication["advisory_engines"] = advisory_engines
    return legacy_card


def score_trade_permission(
    bars,
    pa,
    levels,
    setups=None,
    op=None,
    gp=None,
    magnitude=None,
    volatility_structure=None,
    authority_engine=None,
):
    """Return an explainable trade-permission card for cockpit + signal.json."""
    resolved_engine = resolve_authority_engine(authority_engine)
    if resolved_engine == "ace":
        return build_ace_authority_card(
            bars,
            pa,
            levels,
            setups=setups,
            op=op,
            gp=gp,
            magnitude=magnitude,
            volatility_structure=volatility_structure,
        )
    engine = ExecutionVectorEngine()
    legacy_card = engine.build_card(
        bars,
        pa,
        levels,
        setups=setups,
        op=op,
        gp=gp,
        magnitude=magnitude,
        volatility_structure=volatility_structure,
    )
    try:
        ace_card = build_ace_authority_card(
            bars,
            pa,
            levels,
            setups=setups,
            op=op,
            gp=gp,
            magnitude=magnitude,
            volatility_structure=volatility_structure,
        )
    except Exception:
        return legacy_card
    return _attach_ace_advisory_voice(legacy_card, ace_card)


__all__ = [
    "AUTHORITY_ENGINE_ENV_VAR",
    "ExecutionVectorEngine",
    "resolve_authority_engine",
    "score_trade_permission",
]
