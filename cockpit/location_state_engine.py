"""Pure location-state engine for SharpEdge.

Location owns spatial facts only.
It does not own balance, stretch, or composite edge interpretation.
"""

from __future__ import annotations

from typing import Any

from reference_geometry import numeric_reference_map, reference_row

AT_REFERENCE_PCT = 0.08
NEAR_REFERENCE_PCT = 0.20


def build_location_state(
    current_price: float | None,
    references: dict[str, Any] | None,
) -> dict[str, Any]:
    clean_references = numeric_reference_map(references)
    spot = float(current_price) if isinstance(current_price, (int, float)) else None
    packet = {
        "schema": "sharpedge.location_state.v1",
        "state": "insufficient_references",
        "bias": "NEUTRAL",
        "reason": "no location reference map available",
        "spot": spot,
        "reference_count": len(clean_references),
        "reference_names": sorted(clean_references.keys()),
        "nearest_reference": {},
        "nearest_above_reference": {},
        "nearest_below_reference": {},
        "reference_relations": {},
    }
    if spot is None or not clean_references:
        return packet

    rows = [
        reference_row(spot, name, price) for name, price in clean_references.items()
    ]
    rows.sort(
        key=lambda item: (float(item["distance_pct"] or 0.0), item["reference_name"])
    )
    nearest = rows[0]
    above = [row for row in rows if row["reference_price"] > spot]
    below = [row for row in rows if row["reference_price"] < spot]
    nearest_above = min(above, key=lambda item: item["reference_price"], default={})
    nearest_below = max(below, key=lambda item: item["reference_price"], default={})
    relations = {
        row["reference_name"]: row["relation"]
        for row in sorted(rows, key=lambda item: item["reference_name"])
    }
    base = {
        **packet,
        "nearest_reference": nearest,
        "nearest_above_reference": nearest_above,
        "nearest_below_reference": nearest_below,
        "reference_relations": relations,
    }
    dist = float(nearest.get("distance_pct") or 0.0)
    name = str(nearest.get("reference_name") or "reference")
    price = float(nearest.get("reference_price") or 0.0)
    if dist <= AT_REFERENCE_PCT:
        return {
            **base,
            "state": "at_reference",
            "reason": f"at decision reference {name} {price:.2f}",
        }
    if dist <= NEAR_REFERENCE_PCT:
        return {
            **base,
            "state": "near_reference",
            "reason": f"near {name} {price:.2f} ({dist:.2f}% away)",
        }
    if not above:
        anchor = str(nearest.get("reference_name") or "reference")
        anchor_price = float(nearest.get("reference_price") or 0.0)
        return {
            **base,
            "state": "above_all_references",
            "bias": "CALLS",
            "reason": f"above all tracked references; nearest below is {anchor} {anchor_price:.2f}",
        }
    if not below:
        anchor = str(nearest.get("reference_name") or "reference")
        anchor_price = float(nearest.get("reference_price") or 0.0)
        return {
            **base,
            "state": "below_all_references",
            "bias": "PUTS",
            "reason": f"below all tracked references; nearest above is {anchor} {anchor_price:.2f}",
        }
    lower_name = str(nearest_below.get("reference_name") or "lower")
    lower_price = float(nearest_below.get("reference_price") or 0.0)
    upper_name = str(nearest_above.get("reference_name") or "upper")
    upper_price = float(nearest_above.get("reference_price") or 0.0)
    return {
        **base,
        "state": "between_references",
        "reason": f"between {lower_name} {lower_price:.2f} and {upper_name} {upper_price:.2f}; no nearby reference edge",
    }


__all__ = [
    "AT_REFERENCE_PCT",
    "NEAR_REFERENCE_PCT",
    "build_location_state",
]
