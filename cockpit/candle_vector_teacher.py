"""Teach how Candle Coach events map into SharpEdge execution vectors.

This is education-only connective tissue.  Candles identify auction events;
execution vectors explain what must corroborate those events; graph canon shows
whether the visual battlefield agrees, conflicts, or stays neutral.
"""

from __future__ import annotations

from typing import Any

from execution_vector_taxonomy import VECTOR_PART_TAXONOMY

CANDLE_VECTOR_TEACHER_SCHEMA = "sharpedge.candle_vector_teacher.v1"

DEFAULT_VECTOR_PARTS = (
    "location_score",
    "acceptance_score",
    "volume_score",
    "rejection_score",
    "trap_score",
)

PATTERN_VECTOR_MAP: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        (
            "dragonfly",
            "hammer",
            "demand tail",
            "gravestone",
            "shooting star",
            "supply tail",
        ),
        (
            "location_score",
            "rejection_score",
            "acceptance_score",
            "volume_score",
            "trap_score",
        ),
        "Tail candles teach rejection only after location says the wick hit something that matters.",
    ),
    (
        ("doji", "spinning top", "inside", "harami", "compression", "coil"),
        ("compression_score", "acceptance_score", "volume_score", "time_of_day_score"),
        "Indecision candles teach compression; direction belongs to the expansion break, not the pause candle.",
    ),
    (
        (
            "engulfing",
            "morning star",
            "evening star",
            "outside",
            "kicker",
            "piercing",
            "dark cloud",
        ),
        (
            "trap_score",
            "rejection_score",
            "pressure_score",
            "acceptance_score",
            "location_score",
            "volume_score",
        ),
        "Reversal patterns teach control-transfer; the spine asks whether trapped inventory and acceptance confirm it.",
    ),
    (
        (
            "marubozu",
            "conviction",
            "strong bull",
            "strong bear",
            "soldiers",
            "crows",
            "stair-step",
        ),
        (
            "trend_score",
            "pressure_score",
            "volume_score",
            "acceptance_score",
            "dealer_gamma_score",
        ),
        "Conviction candles teach pressure; continuation still needs participation, acceptance, and no pin/gravity veto.",
    ),
    (
        ("flag", "triangle", "methods"),
        (
            "structure_score",
            "compression_score",
            "acceptance_score",
            "volume_score",
            "balance_context_score",
        ),
        "Multi-candle structures teach setup geometry; the trigger is boundary acceptance with participation.",
    ),
)

VECTOR_TEACHING_NOTES = {
    "structure_score": "Are swings/sequence supporting the candle story, or is the pattern fighting the larger structure?",
    "acceptance_score": "Did price actually stay beyond the level/body/wick, or was it just a poke?",
    "rejection_score": "Did the latest candle reject one side clearly enough to matter right now?",
    "trend_score": "Does VWAP/momentum alignment support follow-through, or is the candle countertrend noise?",
    "volume_score": "Did participation arrive with the candle, or did the shape print on thin air?",
    "location_score": "Did the candle happen at a reference level that matters: VWAP, OR, prior day, wall, gap, swing?",
    "pressure_score": "Are recent closes one-sided after the candle, or already mixed/fading?",
    "time_of_day_score": "Is the session window friendly to continuation, or is it chop/lunch/late trap time?",
    "volatility_score": "Is premium/IV context friendly to follow-through, or likely to overprice the move?",
    "opening_auction_score": "Is the candle part of the opening auction, and has that evidence decayed?",
    "exhaustion_score": "Is the candle appearing after stretch where chasing gets punished?",
    "trap_score": "Did a breakout/breakdown fail and trap one side of the auction?",
    "dealer_gamma_score": "Do pin/call-wall/put-wall/gamma forces support travel or pull price back?",
    "regime_score": "Does the broader day type agree with trend, balance, or fade behavior?",
    "compression_score": "Is energy coiling for expansion, and where are the real boundaries?",
    "balance_context_score": "Is price accepted in value/balance or rejecting away from it?",
}


def _score_row(permission: dict[str, Any], part_name: str) -> dict[str, Any]:
    scores = (
        permission.get("scores") if isinstance(permission.get("scores"), dict) else {}
    )
    row = scores.get(part_name) if isinstance(scores.get(part_name), dict) else {}
    taxonomy = VECTOR_PART_TAXONOMY.get(part_name) or {}
    return {
        "part": part_name,
        "score": row.get("score"),
        "bias": row.get("bias") or "NEUTRAL",
        "reason": str(row.get("reason") or "live vector evidence unavailable"),
        "phase": row.get("phase"),
        "phase_reason": row.get("phase_reason"),
        "category": taxonomy.get("category") or "unknown",
        "correlation_family": taxonomy.get("correlation_family") or "unknown",
        "overlap_families": list(taxonomy.get("overlap_families") or ()),
        "taxonomy_note": taxonomy.get("note") or "",
        "teaching_question": VECTOR_TEACHING_NOTES.get(
            part_name, "What evidence would make this candle matter?"
        ),
    }


def _pattern_names(patterns: list[dict[str, Any]]) -> list[str]:
    return [
        str(pattern.get("name") or "") for pattern in patterns if pattern.get("name")
    ]


def _select_vector_parts(names: list[str]) -> tuple[tuple[str, ...], str]:
    haystack = " ".join(names).lower()
    for needles, parts, lesson in PATTERN_VECTOR_MAP:
        if any(needle in haystack for needle in needles):
            return parts, lesson
    return DEFAULT_VECTOR_PARTS, (
        "Candle shape is only the event label; execution vectors decide whether the event has usable evidence."
    )


def _graph_bridge(permission: dict[str, Any]) -> dict[str, Any]:
    graph = (
        permission.get("graph_state")
        if isinstance(permission.get("graph_state"), dict)
        else {}
    )
    return {
        "graph_bias": graph.get("graph_bias") or "NEUTRAL",
        "graph_reason": graph.get("graph_reason") or "graph context unavailable",
        "authority_role": graph.get("authority_role") or "operator_visual_canon",
        "final_authority_source": graph.get("final_authority_source")
        or "approval_decision_plus_operator",
        "teaching": (
            "The graph is visual canon: vector rows must agree, defer, or explain conflict. "
            "A candle does not overrule what the graph says about battlefield context."
        ),
    }


def _relation_to_graph(row: dict[str, Any], graph_bias: str) -> str:
    bias = str(row.get("bias") or "NEUTRAL")
    if graph_bias == "NEUTRAL" and bias == "NEUTRAL":
        return "aligned_neutral"
    if graph_bias == "NEUTRAL":
        return "graph_neutral_vector_directional"
    if bias == "NEUTRAL":
        return "vector_defers_to_graph"
    if bias == graph_bias:
        return "aligned_with_graph"
    return "conflicts_with_graph"


def build_candle_vector_lesson(
    *,
    patterns: list[dict[str, Any]],
    framework: dict[str, Any],
    sharpedge_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an education packet connecting candle events to execution vectors."""
    context = sharpedge_context if isinstance(sharpedge_context, dict) else {}
    permission = context.get("permission") or context.get("trade_permission") or {}
    permission = permission if isinstance(permission, dict) else {}
    names = _pattern_names(patterns)
    parts, lesson = _select_vector_parts(names)
    graph = _graph_bridge(permission)
    rows = [_score_row(permission, part) for part in parts]
    graph_bias = str(graph.get("graph_bias") or "NEUTRAL")
    for row in rows:
        row["graph_relation"] = _relation_to_graph(row, graph_bias)
    return {
        "schema": CANDLE_VECTOR_TEACHER_SCHEMA,
        "authority": "education_only_not_trade_permission",
        "headline": "Candle → execution vectors → graph canon",
        "pattern_stack": names,
        "pattern_lesson": lesson,
        "graph_bridge": graph,
        "vector_rows": rows,
        "output_state": framework.get("output") or "Watch",
        "doctrine": (
            "Candles name auction behavior. Vectors ask what must prove it. "
            "Graph canon teaches whether that proof fits the visible battlefield."
        ),
    }


__all__ = [
    "CANDLE_VECTOR_TEACHER_SCHEMA",
    "build_candle_vector_lesson",
]
