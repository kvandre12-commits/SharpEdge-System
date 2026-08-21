"""Pure reference-geometry helpers for SharpEdge.

This module owns buffer-aware spatial relations and distance calculations only.
It must not emit semantic level-state labels, setup tags, or authority calls.
"""

from __future__ import annotations

from typing import Any

import execution_vector_primitives as prim


def numeric_reference_map(references: dict[str, Any] | None) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in (references or {}).items()
        if isinstance(value, (int, float))
    }


def distance_pct(
    spot: float | None, reference_price: float | None
) -> float | None:
    if not isinstance(spot, (int, float)) or not isinstance(
        reference_price, (int, float)
    ):
        return None
    if not float(spot):
        return None
    return abs(float(spot) - float(reference_price)) / float(spot) * 100


def relation_to_reference(
    value: float | None,
    reference_price: float,
    *,
    at_label: str = "at_reference",
    buffer: float | None = None,
) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    price = float(reference_price)
    used_buffer = (
        float(buffer)
        if isinstance(buffer, (int, float))
        else prim.buffer_for_price(price)
    )
    value = float(value)
    if value > price + used_buffer:
        return "above"
    if value < price - used_buffer:
        return "below"
    return at_label


def reference_row(
    spot: float,
    name: str,
    price: float,
    *,
    at_label: str = "at_reference",
) -> dict[str, Any]:
    return {
        "reference_name": name,
        "reference_price": price,
        "relation": relation_to_reference(spot, price, at_label=at_label),
        "distance": abs(spot - price),
        "distance_pct": distance_pct(spot, price),
    }


__all__ = [
    "distance_pct",
    "numeric_reference_map",
    "reference_row",
    "relation_to_reference",
]
