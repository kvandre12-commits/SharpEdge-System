from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from accepted_break_interpreter import (
    accepted_break_break_state,
    best_accepted_break_event,
)
from level_state_engine import build_level_state_map


def _breakout_bars() -> list[tuple]:
    return [
        (0, 99.70, 99.90, 99.60, 99.82, 1000),
        (1, 99.82, 100.05, 99.76, 99.98, 1040),
        (2, 99.98, 100.45, 99.97, 100.32, 1100),
        (3, 100.32, 100.36, 100.00, 100.12, 1200),
        (4, 100.12, 100.18, 99.96, 100.02, 1180),
        (5, 100.02, 100.55, 100.08, 100.36, 1300),
        (6, 100.36, 100.72, 100.34, 100.60, 1400),
        (7, 100.60, 100.62, 100.38, 100.44, 1350),
        (8, 100.44, 100.48, 100.30, 100.36, 1320),
        (9, 100.36, 100.78, 100.40, 100.62, 1500),
        (10, 100.62, 100.95, 100.54, 100.76, 1600),
        (11, 100.76, 100.82, 100.64, 100.70, 1550),
        (12, 100.70, 101.02, 100.66, 100.92, 1700),
    ]


def test_accepted_break_interpreter_renders_grammar_view_from_shared_event():
    level_states = build_level_state_map(
        _breakout_bars(),
        {"PDH": 100.0, "ORH": 99.8, "ORL": 98.8, "PDL": 98.5},
        level_names=("PDH", "ORH", "ORL", "PDL"),
        recent_window=6,
        acceptance_window=3,
    )

    event = best_accepted_break_event(
        level_states,
        level_order=("PDH", "ORH", "ORL", "PDL"),
        acceptance_closes=3,
    )

    assert event == {
        "state": "accepted_breakout",
        "bias": "CALLS",
        "level_name": "PDH",
        "level_price": 100.0,
        "score": 72,
        "reason": "3 closes accepted above PDH 100.00",
    }
    assert accepted_break_break_state(event) == {
        "state": "accepted_breakout",
        "bias": "CALLS",
        "level_name": "PDH",
        "level_price": 100.0,
        "score": 72,
        "reason": "3 closes accepted above PDH 100.00",
    }
