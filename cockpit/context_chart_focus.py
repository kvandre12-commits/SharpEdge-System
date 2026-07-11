from __future__ import annotations

from typing import Any


def build_focus_window(
    values: list[float],
    levels: list[dict[str, Any]],
    *,
    anchor: float | None = None,
    recent_values: list[float] | None = None,
    min_span_pct: float = 1.0,
    nearby_level_pct: float = 1.0,
    outer_level_pct: float = 2.0,
    pad_ratio: float = 0.14,
) -> tuple[float, float, set[str]]:
    clean_values = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean_values:
        return 0.0, 1.0, set()

    clean_recent = [
        float(value)
        for value in (recent_values or clean_values)
        if isinstance(value, (int, float))
    ]
    anchor_price = (
        float(anchor)
        if isinstance(anchor, (int, float))
        else clean_recent[-1]
        if clean_recent
        else clean_values[-1]
    )

    visible_names: set[str] = set()
    focus_values = list(clean_recent)
    above_candidates: list[tuple[float, str]] = []
    below_candidates: list[tuple[float, str]] = []

    for level in levels:
        price = level.get("price")
        name = str(level.get("name") or "")
        if not isinstance(price, (int, float)):
            continue
        price = float(price)
        dist_pct = abs(price - anchor_price) / max(abs(anchor_price), 1e-9) * 100
        if dist_pct <= nearby_level_pct:
            focus_values.append(price)
            visible_names.add(name)
            continue
        if price >= anchor_price and dist_pct <= outer_level_pct:
            above_candidates.append((dist_pct, name))
        if price <= anchor_price and dist_pct <= outer_level_pct:
            below_candidates.append((dist_pct, name))

    if above_candidates and not any(
        name in visible_names for _dist, name in above_candidates
    ):
        _dist, name = min(above_candidates)
        visible_names.add(name)
        focus_values.extend(
            float(level["price"])
            for level in levels
            if str(level.get("name") or "") == name
        )
    if below_candidates and not any(
        name in visible_names for _dist, name in below_candidates
    ):
        _dist, name = min(below_candidates)
        visible_names.add(name)
        focus_values.extend(
            float(level["price"])
            for level in levels
            if str(level.get("name") or "") == name
        )

    lo = min(focus_values)
    hi = max(focus_values)
    min_span = max(abs(anchor_price) * (min_span_pct / 100), 1e-9)
    span = max(hi - lo, min_span)
    center = (hi + lo) / 2
    lo = center - span / 2
    hi = center + span / 2
    pad = span * pad_ratio
    return lo - pad, hi + pad, visible_names


__all__ = ["build_focus_window"]
