"""Execution hierarchy packaging for SharpEdge permission vectors."""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim
from graph_state import attach_graph_agreement

CORE_EXECUTION_SPINE_PART_NAMES = (
    "structure_score",
    "acceptance_score",
    "trend_score",
    "pressure_score",
    "location_score",
    "balance_context_score",
    "volume_score",
    "time_of_day_score",
    "dealer_gamma_score",
)

SECONDARY_CONFIRMATION_PART_NAMES = (
    "trap_score",
    "rejection_score",
)

CONTEXT_GOVERNOR_PART_NAMES = (
    "opening_auction_score",
    "exhaustion_score",
    "volatility_score",
    "compression_score",
)

SUSPECT_DRIFT_VOICE_PART_NAMES = ("regime_score",)

ADVISORY_SURFACE_PART_NAMES = ("expansion_fuel_score", "line_authority_score")

EXECUTION_HIERARCHY_LABEL_OVERRIDES = {
    "acceptance_score": "Auction Acceptance",
    "volume_score": "Participation",
    "expansion_fuel_score": "Expansion Fuel",
    "line_authority_score": "Line Authority",
}


def part_label(name: str) -> str:
    return EXECUTION_HIERARCHY_LABEL_OVERRIDES.get(
        name, name.replace("_score", "").replace("_", " ").title()
    )


def _hierarchy_rows(
    names: tuple[str, ...],
    parts: dict[str, Any],
    weights: dict[str, float],
    graph_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        part = parts.get(name)
        if part is None:
            continue
        row = {
            "name": name,
            "label": part_label(name),
            "score": int(part.score),
            "bias": prim.bias_label(part.bias),
            "reason": part.reason,
            "weight": float(weights.get(name, 0.0)),
        }
        rows.append(attach_graph_agreement(row, part, graph_state))
    return rows


def _normalized_weighted_score(rows: list[dict[str, Any]]) -> float:
    total_weight = sum(float(row.get("weight", 0.0)) for row in rows)
    if total_weight <= 0:
        return 0.0
    weighted = sum(float(row["score"]) * float(row["weight"]) for row in rows)
    return weighted / total_weight


def build_execution_hierarchy(
    parts: dict[str, Any],
    score_weights: dict[str, float],
    graph_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    core_rows = _hierarchy_rows(
        CORE_EXECUTION_SPINE_PART_NAMES,
        parts,
        score_weights,
        graph_state,
    )
    ranked_core = sorted(core_rows, key=lambda row: row["score"], reverse=True)
    return {
        "schema": "sharpedge.execution_hierarchy.v1",
        "core_spine": {
            "normalized_weighted_score": round(
                _normalized_weighted_score(core_rows), 2
            ),
            "features": core_rows,
            "best": ranked_core[:3],
            "worst": list(reversed(ranked_core[-3:])),
        },
        "secondary_confirmations": _hierarchy_rows(
            SECONDARY_CONFIRMATION_PART_NAMES,
            parts,
            score_weights,
            graph_state,
        ),
        "context_governors": _hierarchy_rows(
            CONTEXT_GOVERNOR_PART_NAMES,
            parts,
            score_weights,
            graph_state,
        ),
        "suspect_drift_voices": _hierarchy_rows(
            SUSPECT_DRIFT_VOICE_PART_NAMES,
            parts,
            score_weights,
            graph_state,
        ),
        "advisory_surfaces": _hierarchy_rows(
            ADVISORY_SURFACE_PART_NAMES,
            parts,
            score_weights,
            graph_state,
        ),
    }


__all__ = [
    "ADVISORY_SURFACE_PART_NAMES",
    "CONTEXT_GOVERNOR_PART_NAMES",
    "CORE_EXECUTION_SPINE_PART_NAMES",
    "EXECUTION_HIERARCHY_LABEL_OVERRIDES",
    "SECONDARY_CONFIRMATION_PART_NAMES",
    "SUSPECT_DRIFT_VOICE_PART_NAMES",
    "build_execution_hierarchy",
    "part_label",
]
