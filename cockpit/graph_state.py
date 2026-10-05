"""Canonical graph-state packet for SharpEdge cockpit execution reads.

The graph is the operator's primary visual context.  This module does not grant
execution authority; it makes every diagnostic spine vertical state whether it
agrees with, defers to, or conflicts with that visual context.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim

GRAPH_STATE_SCHEMA = "sharpedge.graph_state.v1"
GRAPH_AGREEMENT_SCHEMA = "sharpedge.graph_agreement.v1"
GRAPH_CANON_ROLE = "operator_visual_canon"
FINAL_AUTHORITY_SOURCE = "approval_decision_plus_operator"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_close(bars: list[Any]) -> float | None:
    if not bars:
        return None
    try:
        return float(bars[-1][4])
    except (TypeError, ValueError, IndexError):
        return None


def _level_rows(levels: dict[str, Any], spot: float | None) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(levels):
        value = _float_or_none(levels.get(name))
        if value is None:
            continue
        distance_pct = None
        if spot:
            distance_pct = round((value - spot) / spot * 100.0, 3)
        rows.append({"name": name, "price": value, "distance_pct": distance_pct})
    return rows


def _setup_bias(setups: list[dict[str, Any]]) -> tuple[str, str]:
    for setup in setups or []:
        bias = str(setup.get("bias") or "").upper()
        tag = str(setup.get("tag") or setup.get("kind") or "setup")
        if bias in {"CALLS", "BULLISH"}:
            return "CALLS", f"fresh setup marker favors calls: {tag}"
        if bias in {"PUTS", "BEARISH"}:
            return "PUTS", f"fresh setup marker favors puts: {tag}"
    return "NEUTRAL", "no fresh directional setup marker on graph"


def _price_bias(pa: dict[str, Any]) -> tuple[str, str]:
    vs_vwap = _float_or_none(pa.get("vs_vwap")) or 0.0
    mom15 = _float_or_none(pa.get("mom15")) or 0.0
    rng_pos = _float_or_none(pa.get("rng_pos"))
    if vs_vwap >= 0.10 and mom15 >= 0.05:
        return (
            "CALLS",
            f"price is above VWAP ({vs_vwap:+.2f}%) with positive 15m momentum",
        )
    if vs_vwap <= -0.10 and mom15 <= -0.05:
        return (
            "PUTS",
            f"price is below VWAP ({vs_vwap:+.2f}%) with negative 15m momentum",
        )
    if rng_pos is not None and rng_pos >= 80 and mom15 < 0:
        return "PUTS", f"graph is high in range ({rng_pos:.0f}%) with fading momentum"
    if rng_pos is not None and rng_pos <= 20 and mom15 > 0:
        return (
            "CALLS",
            f"graph is low in range ({rng_pos:.0f}%) with reclaiming momentum",
        )
    return (
        "NEUTRAL",
        "graph shows rotation/balance rather than clean directional control",
    )


def _graph_bias(pa: dict[str, Any], setups: list[dict[str, Any]]) -> tuple[str, str]:
    setup_bias, setup_reason = _setup_bias(setups)
    if setup_bias != "NEUTRAL":
        return setup_bias, setup_reason
    return _price_bias(pa)


def build_graph_state(
    bars: list[Any],
    pa: dict[str, Any],
    levels: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the graph-canon packet consumed by spine diagnostics."""
    spot = _float_or_none(pa.get("spot")) or _latest_close(bars)
    day_open = _float_or_none(pa.get("day_open"))
    vwap = _float_or_none(pa.get("vwap"))
    bias, reason = _graph_bias(pa, setups or [])
    last_minute = None
    if bars:
        try:
            last_minute = int(bars[-1][0])
        except (TypeError, ValueError, IndexError):
            last_minute = None
    return {
        "schema": GRAPH_STATE_SCHEMA,
        "authority_role": GRAPH_CANON_ROLE,
        "final_authority_source": FINAL_AUTHORITY_SOURCE,
        "graph_bias": bias,
        "graph_reason": reason,
        "spot": spot,
        "last_close": _latest_close(bars),
        "last_bar_minute": last_minute,
        "day_open": day_open,
        "vwap": vwap,
        "vs_vwap": _float_or_none(pa.get("vs_vwap")),
        "mom15": _float_or_none(pa.get("mom15")),
        "rng_pos": _float_or_none(pa.get("rng_pos")),
        "session_position_in_range": _float_or_none(
            pa.get("session_position_in_range")
        ),
        "visible_references": _level_rows(levels, spot),
        "setup_marker_count": len(setups or []),
        "contract": (
            "Spine verticals must respect this visual graph context, carry "
            "agreement/conflict evidence, and remain diagnostic until operator approval."
        ),
    }


def graph_agreement_for_part(
    name: str, part: Any, graph_state: dict[str, Any]
) -> dict[str, Any]:
    """Describe how one spine vertical relates to graph canon."""
    graph_bias = str(graph_state.get("graph_bias") or "NEUTRAL")
    part_bias = prim.bias_label(getattr(part, "bias", 0.0))
    if graph_bias == "NEUTRAL" and part_bias == "NEUTRAL":
        relation = "aligned"
        action = "observe"
    elif graph_bias == "NEUTRAL":
        relation = "graph_neutral_part_directional"
        action = "explain_without_overriding_graph"
    elif part_bias == "NEUTRAL":
        relation = "deferred_to_graph"
        action = "respect_graph_context"
    elif part_bias == graph_bias:
        relation = "aligned"
        action = "respect_graph_context"
    else:
        relation = "conflict"
        action = "demote_or_explain_before_trusting"
    return {
        "schema": GRAPH_AGREEMENT_SCHEMA,
        "part": name,
        "relation": relation,
        "action": action,
        "graph_bias": graph_bias,
        "part_bias": part_bias,
        "graph_reason": str(graph_state.get("graph_reason") or ""),
        "authority_role": graph_state.get("authority_role") or GRAPH_CANON_ROLE,
    }


def attach_graph_agreement(
    row: dict[str, Any], part: Any, graph_state: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a row copy annotated with graph agreement when graph_state exists."""
    if not graph_state:
        return row
    annotated = dict(row)
    annotated["graph_agreement"] = graph_agreement_for_part(
        str(row.get("name") or "unknown_score"), part, graph_state
    )
    return annotated


__all__ = [
    "FINAL_AUTHORITY_SOURCE",
    "GRAPH_AGREEMENT_SCHEMA",
    "GRAPH_CANON_ROLE",
    "GRAPH_STATE_SCHEMA",
    "attach_graph_agreement",
    "build_graph_state",
    "graph_agreement_for_part",
]
