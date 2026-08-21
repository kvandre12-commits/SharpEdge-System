"""Public trade-permission wrapper for the SharpEdge cockpit."""

from __future__ import annotations

import os
from typing import Any

from ace_authority import build_ace_authority_card, normalize_authority_engine
from execution_vector_engine import ExecutionVectorEngine

AUTHORITY_ENGINE_ENV_VAR = "SHARPEDGE_AUTHORITY_ENGINE"
MARKET_DATA_GUARD_SCHEMA = "sharpedge.market_data_guard.v1"


def _date_prefix(value: Any) -> str | None:
    text = str(value or "")
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return None


def _build_market_data_guard(
    pa: dict[str, Any],
    gp: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a fail-closed eligibility verdict for live permission output.

    No provenance means an older/internal caller that has not opted into the
    live guard. The cockpit always supplies provenance before publishing a
    signal, so missing or stale live evidence cannot become PERMIT.
    """
    if provenance is None:
        return {
            "schema": MARKET_DATA_GUARD_SCHEMA,
            "status": "unchecked",
            "eligible": True,
            "blockers": [],
        }

    blockers: list[str] = []
    price_authority = pa.get("price_authority") or {}
    analysis_stale = price_authority.get("analysis_bar_stale")
    if analysis_stale is True:
        blockers.append("analysis_bars_stale")
    elif analysis_stale is not False:
        blockers.append("analysis_bar_freshness_unknown")

    options_source = provenance.get("options") or {}
    expected_session = str(provenance.get("price_session_date") or "") or None
    observed_session = _date_prefix(
        options_source.get("latest_option_trade_time_raw")
        or options_source.get("last_trade_time_raw")
    )
    if expected_session and observed_session != expected_session:
        blockers.append(
            "options_session_missing"
            if observed_session is None
            else "options_session_mismatch"
        )
    elif expected_session is None:
        blockers.append("price_session_missing")

    gamma = gp or {}
    gamma_quality = str(gamma.get("gamma_data_quality") or "missing").lower()
    if gamma_quality != "ok":
        blockers.append(f"gamma_quality_{gamma_quality}")
    dte = gamma.get("dte")
    if isinstance(dte, (int, float)) and dte < 0:
        blockers.append("gamma_expired")

    return {
        "schema": MARKET_DATA_GUARD_SCHEMA,
        "status": "eligible" if not blockers else "blocked",
        "eligible": not blockers,
        "blockers": blockers,
        "analysis_bar_stale": analysis_stale,
        "analysis_bar_lag_minutes": price_authority.get("analysis_bar_lag_minutes"),
        "price_session_date": expected_session,
        "options_session_date": observed_session,
        "gamma_data_quality": gamma_quality,
        "gamma_dte": dte,
    }


def _apply_market_data_guard(
    card: dict[str, Any],
    pa: dict[str, Any],
    gp: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    guard = _build_market_data_guard(pa, gp, provenance)
    card["market_data_guard"] = guard
    if guard["eligible"]:
        return card

    card["trade_gate"] = "BLOCK"
    warnings = list(card.get("warning_reasons") or [])
    card["warning_reasons"] = [
        *(f"market data guard: {reason}" for reason in guard["blockers"]),
        *warnings,
    ]
    execution_permission = (card.get("execution_flow") or {}).get(
        "execution_permission"
    )
    if isinstance(execution_permission, dict):
        execution_permission["gate"] = "BLOCK"
        execution_permission["data_guard_override"] = True

    adjudication = card.get("authority_adjudication") or {}
    for key in ("cockpit_read", "we_are_doing_this"):
        read = adjudication.get(key)
        if isinstance(read, dict):
            read["gate"] = "BLOCK"
            read["action"] = "stand_down_stale_or_invalid_data"
            read["data_guard_override"] = True
    return card


def resolve_authority_engine(authority_engine: str | None = None) -> str:
    return normalize_authority_engine(
        authority_engine or os.getenv(AUTHORITY_ENGINE_ENV_VAR)
    )


def _ace_advisory_voice(card: dict[str, Any]) -> dict[str, Any]:
    authority = card.get("authority_summary") or {}
    adjudication = card.get("authority_adjudication") or {}
    doing = (
        adjudication.get("cockpit_read") or adjudication.get("we_are_doing_this") or {}
    )
    stance = (
        f"{doing.get('gate') or authority.get('gate') or 'BLOCK'} / "
        f"{doing.get('bias') or authority.get('bias') or 'NEUTRAL'} / "
        f"{doing.get('action') or authority.get('diagnostic_posture') or 'watch_only_context'}"
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
                "diagnostic_posture"
            )
            or "watch_only_context",
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
    data_provenance=None,
):
    """Return an explainable trade-permission card for cockpit + signal.json."""
    resolved_engine = resolve_authority_engine(authority_engine)
    if resolved_engine == "ace":
        card = build_ace_authority_card(
            bars,
            pa,
            levels,
            setups=setups,
            op=op,
            gp=gp,
            magnitude=magnitude,
            volatility_structure=volatility_structure,
        )
        return _apply_market_data_guard(card, pa, gp, data_provenance)

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
        card = legacy_card
    else:
        card = _attach_ace_advisory_voice(legacy_card, ace_card)
    return _apply_market_data_guard(card, pa, gp, data_provenance)


__all__ = [
    "AUTHORITY_ENGINE_ENV_VAR",
    "MARKET_DATA_GUARD_SCHEMA",
    "ExecutionVectorEngine",
    "resolve_authority_engine",
    "score_trade_permission",
]
