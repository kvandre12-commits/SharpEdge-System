"""Session timing state engine for SharpEdge."""

from __future__ import annotations

from typing import Any

import execution_vector_context as ctx
from session_doctrine import (
    REGULAR_SESSION_MINUTES,
    classify_session_window,
    clock_label,
)


def build_time_state(bars: list[tuple] | None) -> dict[str, Any]:
    packet = {
        "schema": "sharpedge.time_state.v1",
        "state": "closed_or_unknown",
        "bias": "NEUTRAL",
        "reason": "session_unavailable",
        "detail": "no session timing context available",
        "minutes_since_open": None,
        "clock": None,
        "within_regular_session": False,
    }
    clean_bars = bars or []
    if not clean_bars:
        return packet

    current_time = ctx.session_datetime(clean_bars)
    minutes_since_open = ctx.last_minute(clean_bars)
    clock = clock_label(current_time)
    session = classify_session_window(minutes_since_open, clock=clock)
    return {
        **packet,
        **session,
        "minutes_since_open": minutes_since_open,
        "clock": clock,
    }


__all__ = ["build_time_state", "REGULAR_SESSION_MINUTES"]
