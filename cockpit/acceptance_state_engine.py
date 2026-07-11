"""Pure acceptance-state engine for SharpEdge.

Acceptance owns repeated close-based acceptance at explicit levels only.
It does not own trend, VWAP drift, failed breaks, or broader auction role state.
"""

from __future__ import annotations

from typing import Any

import execution_vector_context as ctx
import execution_vector_primitives as prim

ACCEPTANCE_WINDOW = 3


def _numeric_levels(levels: dict[str, Any] | None) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in (levels or {}).items()
        if isinstance(value, (int, float))
    }


def _acceptance_for_level(
    closes: list[float],
    level_name: str,
    level_price: float,
    *,
    acceptance_window: int,
) -> dict[str, Any] | None:
    buffer = prim.buffer_for_price(level_price)
    if all(close > level_price + buffer for close in closes):
        return {
            "level_name": level_name,
            "level_price": level_price,
            "acceptance": "accepted_above",
            "buffer": buffer,
            "reason": f"{acceptance_window} closes accepted above {level_name} {level_price:.2f}",
        }
    if all(close < level_price - buffer for close in closes):
        return {
            "level_name": level_name,
            "level_price": level_price,
            "acceptance": "accepted_below",
            "buffer": buffer,
            "reason": f"{acceptance_window} closes accepted below {level_name} {level_price:.2f}",
        }
    return None


def build_acceptance_state(
    bars: list[tuple[Any, ...]] | list[list[Any]],
    levels: dict[str, Any] | None,
    *,
    acceptance_window: int = ACCEPTANCE_WINDOW,
) -> dict[str, Any]:
    closes = [
        float(close) for close in ctx.recent_closes(list(bars), acceptance_window)
    ]
    latest_close = closes[-1] if closes else None
    clean_levels = _numeric_levels(levels)
    packet = {
        "schema": "sharpedge.acceptance_state.v1",
        "state": "insufficient_data",
        "bias": "NEUTRAL",
        "reason": f"need {int(acceptance_window)} closes for acceptance",
        "acceptance_window": int(acceptance_window),
        "recent_close_count": len(closes),
        "latest_close": latest_close,
        "evaluated_level_count": len(clean_levels),
        "evaluated_levels": sorted(clean_levels.keys()),
        "accepted_level_count": 0,
        "accepted_levels": [],
        "representative_level": {},
    }
    if len(closes) < int(acceptance_window):
        return packet
    accepted_levels = []
    for level_name, level_price in clean_levels.items():
        acceptance = _acceptance_for_level(
            closes,
            level_name,
            level_price,
            acceptance_window=int(acceptance_window),
        )
        if not acceptance:
            continue
        accepted_levels.append(
            {
                **acceptance,
                "distance_from_latest_close": abs(level_price - latest_close),
            }
        )
    accepted_levels.sort(
        key=lambda item: (float(item["distance_from_latest_close"]), item["level_name"])
    )
    if not accepted_levels:
        return {
            **packet,
            "state": "no_acceptance",
            "reason": "no clean level acceptance",
        }
    representative = accepted_levels[0]
    acceptance = str(representative.get("acceptance") or "")
    state = (
        "accepted_above_level"
        if acceptance == "accepted_above"
        else "accepted_below_level"
    )
    bias = "CALLS" if acceptance == "accepted_above" else "PUTS"
    return {
        **packet,
        "state": state,
        "bias": bias,
        "reason": str(representative.get("reason") or ""),
        "accepted_level_count": len(accepted_levels),
        "accepted_levels": accepted_levels,
        "representative_level": representative,
    }


__all__ = ["ACCEPTANCE_WINDOW", "build_acceptance_state"]
