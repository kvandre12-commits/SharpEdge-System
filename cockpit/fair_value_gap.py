"""Simple fair value gap detection for SharpEdge.

This module is intentionally mechanical. It detects classic three-candle FVGs,
tracks whether they have been filled, and exposes the nearest still-open gaps.
It does not decide whether a gap should be traded.
"""

from __future__ import annotations

from typing import Any


def _fill_state_for_bullish(
    gap_low: float,
    gap_high: float,
    subsequent_bars: list[tuple[Any, ...]] | list[list[Any]],
) -> tuple[str, float]:
    if not subsequent_bars:
        return "open", 0.0
    lowest_low = min(float(bar[3]) for bar in subsequent_bars)
    if lowest_low <= gap_low:
        return "filled", 100.0
    if lowest_low < gap_high:
        pct = (gap_high - lowest_low) / max(gap_high - gap_low, 1e-9) * 100
        return "partial", min(100.0, max(0.0, pct))
    return "open", 0.0


def _fill_state_for_bearish(
    gap_low: float,
    gap_high: float,
    subsequent_bars: list[tuple[Any, ...]] | list[list[Any]],
) -> tuple[str, float]:
    if not subsequent_bars:
        return "open", 0.0
    highest_high = max(float(bar[2]) for bar in subsequent_bars)
    if highest_high >= gap_high:
        return "filled", 100.0
    if highest_high > gap_low:
        pct = (highest_high - gap_low) / max(gap_high - gap_low, 1e-9) * 100
        return "partial", min(100.0, max(0.0, pct))
    return "open", 0.0


def _gap_packet(
    *,
    direction: str,
    start_index: int,
    created_index: int,
    minute: int,
    gap_low: float,
    gap_high: float,
    spot: float | None,
    subsequent_bars: list[tuple[Any, ...]] | list[list[Any]],
    total_bars: int,
) -> dict[str, Any]:
    fill_state, fill_pct = (
        _fill_state_for_bullish(gap_low, gap_high, subsequent_bars)
        if direction == "bullish"
        else _fill_state_for_bearish(gap_low, gap_high, subsequent_bars)
    )
    midpoint = (gap_low + gap_high) / 2
    return {
        "direction": direction,
        "start_index": start_index,
        "created_index": created_index,
        "minute": minute,
        "gap_low": round(gap_low, 2),
        "gap_high": round(gap_high, 2),
        "midpoint": round(midpoint, 2),
        "size": round(gap_high - gap_low, 2),
        "size_pct": round((gap_high - gap_low) / max(midpoint, 1e-9) * 100, 3),
        "fill_state": fill_state,
        "fill_pct": round(fill_pct, 1),
        "age_bars": max(0, total_bars - 1 - created_index),
        "distance_from_spot": (
            round(abs(float(spot) - midpoint), 2)
            if isinstance(spot, (int, float))
            else None
        ),
        "position_vs_spot": (
            "below"
            if isinstance(spot, (int, float)) and midpoint < float(spot)
            else "above"
            if isinstance(spot, (int, float)) and midpoint > float(spot)
            else "at"
            if isinstance(spot, (int, float))
            else "unknown"
        ),
        "fill_direction": "down" if direction == "bullish" else "up",
    }


def _nearest_gap(
    gaps: list[dict[str, Any]],
    *,
    position_vs_spot: str,
) -> dict[str, Any]:
    matched = [gap for gap in gaps if gap.get("position_vs_spot") == position_vs_spot]
    if not matched:
        return {}
    return min(
        matched,
        key=lambda gap: (
            float(gap.get("distance_from_spot") or 1e9),
            int(gap.get("age_bars") or 0),
        ),
    )


def build_fair_value_gap_map(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    *,
    spot: float | None = None,
    max_gaps: int = 12,
) -> dict[str, Any]:
    rows = list(bars or [])
    packet = {
        "schema": "sharpedge.fair_value_gaps.v1",
        "gap_count": 0,
        "open_gap_count": 0,
        "gaps": [],
        "open_gaps": [],
        "latest_bullish_gap": {},
        "latest_bearish_gap": {},
        "nearest_open_gap": {},
        "nearest_open_gap_above": {},
        "nearest_open_gap_below": {},
    }
    if len(rows) < 3:
        return packet

    gaps: list[dict[str, Any]] = []
    for index in range(2, len(rows)):
        first = rows[index - 2]
        third = rows[index]
        first_high = float(first[2])
        first_low = float(first[3])
        third_high = float(third[2])
        third_low = float(third[3])
        subsequent = rows[index + 1 :]
        minute = int(third[0])
        if third_low > first_high:
            gaps.append(
                _gap_packet(
                    direction="bullish",
                    start_index=index - 2,
                    created_index=index,
                    minute=minute,
                    gap_low=first_high,
                    gap_high=third_low,
                    spot=spot,
                    subsequent_bars=subsequent,
                    total_bars=len(rows),
                )
            )
        if third_high < first_low:
            gaps.append(
                _gap_packet(
                    direction="bearish",
                    start_index=index - 2,
                    created_index=index,
                    minute=minute,
                    gap_low=third_high,
                    gap_high=first_low,
                    spot=spot,
                    subsequent_bars=subsequent,
                    total_bars=len(rows),
                )
            )

    kept = gaps[-max(int(max_gaps or 0), 1) :]
    open_gaps = [gap for gap in kept if gap.get("fill_state") != "filled"]
    latest_bullish = next(
        (gap for gap in reversed(kept) if gap.get("direction") == "bullish"), {}
    )
    latest_bearish = next(
        (gap for gap in reversed(kept) if gap.get("direction") == "bearish"), {}
    )
    nearest_any = min(
        open_gaps,
        key=lambda gap: (
            float(gap.get("distance_from_spot") or 1e9),
            int(gap.get("age_bars") or 0),
        ),
        default={},
    )
    return {
        **packet,
        "gap_count": len(kept),
        "open_gap_count": len(open_gaps),
        "gaps": kept,
        "open_gaps": open_gaps,
        "latest_bullish_gap": latest_bullish,
        "latest_bearish_gap": latest_bearish,
        "nearest_open_gap": nearest_any,
        "nearest_open_gap_above": _nearest_gap(open_gaps, position_vs_spot="above"),
        "nearest_open_gap_below": _nearest_gap(open_gaps, position_vs_spot="below"),
    }


__all__ = ["build_fair_value_gap_map"]
