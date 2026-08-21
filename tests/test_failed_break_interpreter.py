from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from failed_break_interpreter import (
    best_failed_break_event,
    collect_failed_break_events,
    failed_break_break_state,
    failed_break_setup_card,
)
from level_state_engine import build_level_state_map


def _failed_breakdown_bars() -> list[tuple]:
    return [
        (0, 100.02, 100.08, 99.96, 100.04, 1000),
        (1, 100.04, 100.06, 99.86, 99.94, 1100),
        (2, 99.94, 99.98, 99.74, 99.82, 1200),
        (3, 99.82, 100.08, 99.92, 100.04, 1300),
        (4, 100.04, 100.09, 99.98, 100.06, 1250),
    ]


def test_failed_break_interpreter_renders_setup_and_grammar_views_from_same_event():
    level_states = build_level_state_map(
        _failed_breakdown_bars(),
        {"ORL": 100.0},
        level_names=("ORL",),
        recent_window=6,
    )

    event = best_failed_break_event(level_states, level_order=("ORL",), recent_bars=6)

    assert event == {
        "state": "failed_breakdown",
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS",
        "setup_bias": "CALLS (bullish)",
        "setup_kind": "ok",
        "level_name": "ORL",
        "level_price": 100.0,
        "trigger_price": 99.74,
        "bars_ago": 1,
        "event_age_bars": 1,
        "event_detected": True,
        "entry_window_open": True,
        "magnitude_pct": 0.2600000000000051,
        "score": 5.260000000000005,
    }
    assert failed_break_setup_card(event) == {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": "reclaimed ORL $100.00 1m ago after stabbing $99.74 (-0.26% below) - bear trap; entry window open",
        "score": 5.260000000000005,
        "level_name": "ORL",
        "level_price": 100.0,
        "trigger_price": 99.74,
        "bars_ago": 1,
        "event_age_bars": 1,
        "event_detected": True,
        "entry_window_open": True,
    }
    assert failed_break_break_state(event) == {
        "state": "failed_breakdown",
        "bias": "CALLS",
        "level_name": "ORL",
        "level_price": 100.0,
        "trigger_price": 99.74,
        "score": 88,
        "reason": "sellers trapped below ORL 100.00; reclaimed from 99.74",
    }


def test_failed_break_observation_outlives_fresh_entry_window():
    bars = [*_failed_breakdown_bars()]
    bars.extend(
        [
            (5, 100.06, 100.10, 100.01, 100.05, 1100),
            (6, 100.05, 100.09, 100.02, 100.04, 1100),
            (7, 100.04, 100.08, 100.01, 100.03, 1100),
            (8, 100.03, 100.07, 100.01, 100.04, 1100),
            (9, 100.04, 100.08, 100.02, 100.05, 1100),
            (10, 100.05, 100.09, 100.02, 100.06, 1100),
            (11, 100.06, 100.10, 100.03, 100.07, 1100),
        ]
    )
    level_states = build_level_state_map(
        bars,
        {"ORL": 100.0},
        level_names=("ORL",),
        recent_window=6,
    )

    assert level_states["ORL"]["event_detected"] is True
    assert level_states["ORL"]["entry_window_open"] is False
    assert level_states["ORL"]["event_age_bars"] == 8
    assert (
        best_failed_break_event(level_states, level_order=("ORL",), recent_bars=6) == {}
    )

    observed = collect_failed_break_events(
        level_states,
        level_order=("ORL",),
        recent_bars=6,
        entry_window_only=False,
    )
    assert len(observed) == 1
    assert observed[0]["event_detected"] is True
    assert observed[0]["entry_window_open"] is False
    assert failed_break_setup_card(observed[0])["detail"].endswith("entry window stale")
