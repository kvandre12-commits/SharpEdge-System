from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from time_state_engine import build_time_state
from trade_permission import ExecutionVectorEngine
from trade_permission_context import NEUTRAL


def _bars(start_minute: int = 0, count: int = 8) -> list[tuple]:
    bars = []
    price = 100.0
    for minute in range(count):
        open_ = price
        close = price + 0.05
        high = close + 0.03
        low = open_ - 0.02
        bars.append(
            (start_minute + minute, open_, high, low, close, 1000 + minute * 10)
        )
        price = close
    return bars


def test_time_state_detects_opening_window():
    packet = build_time_state(_bars(start_minute=5))

    assert packet["schema"] == "sharpedge.time_state.v1"
    assert packet["state"] == "opening"
    assert packet["reason"] == "opening_auction"
    assert packet["within_regular_session"] is True


def test_time_state_detects_midday_window():
    packet = build_time_state(_bars(start_minute=180))

    assert packet["state"] == "midday"
    assert packet["reason"] == "midday_chop"


def test_time_state_detects_power_hour_window():
    packet = build_time_state(_bars(start_minute=340))

    assert packet["state"] == "power_hour"
    assert packet["reason"] == "power_hour_positioning"


def test_time_state_marks_outside_regular_hours_as_closed_or_unknown():
    packet = build_time_state(_bars(start_minute=410))

    assert packet["state"] == "closed_or_unknown"
    assert packet["reason"] == "outside_regular_hours"
    assert packet["within_regular_session"] is False


def test_execution_vector_engine_time_score_uses_time_state_engine():
    engine = ExecutionVectorEngine()
    morning = engine.build_parts(
        _bars(start_minute=45),
        {"spot": 100.4, "vs_vwap": 0.1, "mom15": 0.1, "vwap": 100.2},
        {"ORH": 100.5, "ORL": 99.8, "PDC": 100.0},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["time_of_day_score"]
    afternoon = engine.build_parts(
        _bars(start_minute=270),
        {"spot": 100.4, "vs_vwap": 0.1, "mom15": 0.1, "vwap": 100.2},
        {"ORH": 100.5, "ORL": 99.8, "PDC": 100.0},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["time_of_day_score"]

    assert morning.score == 74
    assert morning.bias == NEUTRAL
    assert "morning continuation window" in morning.reason
    assert afternoon.score == 58
    assert afternoon.bias == NEUTRAL
