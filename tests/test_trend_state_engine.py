from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from trade_permission import ExecutionVectorEngine  # noqa: E402
from trend_state_engine import build_trend_state  # noqa: E402
from trade_permission_context import BULLISH, NEUTRAL  # noqa: E402


def _bars(step: float = 0.08, count: int = 8) -> list[tuple]:
    bars = []
    price = 100.0
    for minute in range(count):
        open_ = price
        close = price + step
        high = max(open_, close) + 0.03
        low = min(open_, close) - 0.02
        bars.append((minute, open_, high, low, close, 1000 + minute * 10))
        price = close
    return bars


def test_trend_state_detects_aligned_up():
    packet = build_trend_state(_bars(), {"vs_vwap": 0.18, "mom15": 0.22})

    assert packet["schema"] == "sharpedge.trend_state.v1"
    assert packet["state"] == "aligned_up"
    assert packet["bias"] == "CALLS"
    assert packet["reason"] == "full_alignment"
    assert packet["component_states"] == {
        "slope": "up",
        "vwap": "up",
        "momentum": "up",
    }


def test_trend_state_detects_conflict_with_vwap_rotation_reason():
    packet = build_trend_state(_bars(step=0.08), {"vs_vwap": -0.18, "mom15": 0.22})

    assert packet["state"] == "conflict"
    assert packet["bias"] == "NEUTRAL"
    assert packet["reason"] == "vwap_rotation"
    assert packet["component_states"]["slope"] == "up"
    assert packet["component_states"]["vwap"] == "down"
    assert packet["component_states"]["momentum"] == "up"


def test_trend_state_uses_neutral_state_for_vwap_chop_rotation():
    packet = build_trend_state(_bars(step=0.0), {"vs_vwap": 0.01, "mom15": 0.01})

    assert packet["state"] == "neutral"
    assert packet["reason"] == "vwap_chop"
    assert packet["bias"] == "NEUTRAL"


def test_trend_state_marks_insufficient_without_enough_bars():
    packet = build_trend_state(_bars(count=4), {"vs_vwap": 0.18, "mom15": 0.22})

    assert packet["state"] == "insufficient"
    assert packet["reason"] == "insufficient_bars"


def test_execution_vector_engine_trend_score_uses_trend_state_engine():
    engine = ExecutionVectorEngine()
    aligned = engine.build_parts(
        _bars(),
        {"spot": 100.64, "vs_vwap": 0.18, "mom15": 0.22, "vwap": 100.2},
        {"ORH": 100.4, "ORL": 99.8, "PDC": 99.9},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["trend_score"]
    conflict = engine.build_parts(
        _bars(),
        {"spot": 100.64, "vs_vwap": -0.18, "mom15": 0.22, "vwap": 100.8},
        {"ORH": 100.4, "ORL": 99.8, "PDC": 99.9},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["trend_score"]

    assert aligned.score == 82
    assert aligned.bias == BULLISH
    assert "aligned up" in aligned.reason
    assert conflict.score == 42
    assert conflict.bias == NEUTRAL
