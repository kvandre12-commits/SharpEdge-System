from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from structure_state_engine import build_structure_state
from trade_permission import ExecutionVectorEngine


def _bullish_sequence_bars() -> list[tuple]:
    rows = [
        (0, 100.0, 100.4, 99.8, 100.2, 1000),
        (1, 100.2, 101.0, 100.0, 100.8, 1010),
        (2, 100.8, 102.2, 100.6, 101.9, 1020),
        (3, 101.9, 102.1, 100.9, 101.2, 1030),
        (4, 101.2, 101.4, 100.2, 100.5, 1040),
        (5, 100.5, 102.0, 100.6, 101.7, 1050),
        (6, 101.7, 103.1, 101.3, 102.8, 1060),
        (7, 102.8, 103.0, 101.7, 102.0, 1070),
        (8, 102.0, 102.2, 101.1, 101.4, 1080),
        (9, 101.4, 103.0, 101.5, 102.6, 1090),
        (10, 102.6, 104.2, 102.1, 103.8, 1100),
        (11, 103.8, 104.0, 102.8, 103.1, 1110),
        (12, 103.1, 103.3, 102.4, 102.7, 1120),
    ]
    return rows


def _sparse_drift_bars() -> list[tuple]:
    bars = []
    price = 100.0
    for minute in range(12):
        open_ = price
        close = price + 0.10
        high = close + 0.03
        low = open_ - 0.02
        bars.append((minute, open_, high, low, close, 1000 + minute * 5))
        price = close
    return bars


def _weak_bullish_sequence_bars() -> list[tuple]:
    return [
        (0, 100.0, 100.2, 99.9, 100.1, 1000),
        (1, 100.1, 100.8, 100.0, 100.7, 1010),
        (2, 100.7, 101.3, 100.6, 101.1, 1020),
        (3, 101.1, 101.0, 100.4, 100.5, 1030),
        (4, 100.5, 100.6, 100.0, 100.1, 1040),
        (5, 100.1, 101.35, 100.05, 101.0, 1050),
        (6, 101.0, 101.0, 100.4, 100.6, 1060),
        (7, 100.6, 100.65, 100.05, 100.2, 1070),
        (8, 100.2, 101.40, 100.1, 101.1, 1080),
        (9, 101.1, 101.0, 100.5, 100.7, 1090),
        (10, 100.7, 100.8, 100.4, 100.6, 1100),
    ]


def _stale_bullish_sequence_bars() -> list[tuple]:
    rows = list(_bullish_sequence_bars())
    price = rows[-1][4]
    for minute in range(13, 20):
        open_ = price
        close = price + 0.12
        high = close + 0.02
        low = open_ - 0.02
        rows.append((minute, open_, high, low, close, 1120 + minute * 5))
        price = close
    return rows


def test_structure_state_detects_bullish_sequence_from_swings():
    packet = build_structure_state(_bullish_sequence_bars())

    assert packet["state"] == "bullish_sequence"
    assert packet["bias"] == "CALLS"
    assert packet["reason"] == "HH/HL structure intact"
    assert packet["sequence_quality"] == "confirmed"
    assert packet["spacing_ok"] is True
    assert packet["amplitude_ok"] is True
    assert packet["freshness_ok"] is True
    assert packet["swing_high_count"] >= 2
    assert packet["swing_low_count"] >= 2


def test_structure_state_flags_weak_sequence_when_pivots_are_tight_and_small():
    packet = build_structure_state(_weak_bullish_sequence_bars())

    assert packet["state"] == "bullish_sequence"
    assert packet["sequence_quality"] == "weak"
    assert packet["spacing_ok"] is False
    assert packet["amplitude_ok"] is False
    assert "pivot_spacing_tight" in packet["quality_issues"]
    assert "swing_amplitude_small" in packet["quality_issues"]
    assert "pivot spacing is tight" in packet["reason"]


def test_structure_state_flags_stale_sequence_when_latest_pivots_are_old():
    packet = build_structure_state(_stale_bullish_sequence_bars())

    assert packet["state"] == "bullish_sequence"
    assert packet["sequence_quality"] == "weak"
    assert packet["freshness_ok"] is False
    assert "pivot_freshness_stale" in packet["quality_issues"]
    assert "latest pivots are getting stale" in packet["reason"]


def test_structure_state_stays_neutral_when_sequence_is_sparse():
    packet = build_structure_state(_sparse_drift_bars())

    assert packet["state"] == "insufficient_sequence"
    assert packet["bias"] == "NEUTRAL"
    assert "not enough confirmed swing points" in packet["reason"]


def test_execution_vector_engine_structure_score_uses_structure_state_engine():
    engine = ExecutionVectorEngine()

    bullish = engine.build_parts(
        _bullish_sequence_bars(),
        {"spot": 102.2, "vwap": 101.3, "vs_vwap": 0.2, "mom15": 0.4},
        {"ORH": 102.0, "ORL": 99.8, "PDC": 100.5},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["structure_score"]
    weak = engine.build_parts(
        _weak_bullish_sequence_bars(),
        {"spot": 100.6, "vwap": 100.5, "vs_vwap": 0.08, "mom15": 0.12},
        {"ORH": 101.0, "ORL": 100.0, "PDC": 100.4},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["structure_score"]
    sparse = engine.build_parts(
        _sparse_drift_bars(),
        {"spot": 101.2, "vwap": 100.8, "vs_vwap": 0.15, "mom15": 0.3},
        {"ORH": 101.0, "ORL": 99.8, "PDC": 100.5},
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )["structure_score"]

    assert bullish.score == 82
    assert bullish.reason == "HH/HL structure intact"
    assert weak.score == 68
    assert "pivot spacing is tight" in weak.reason
    assert sparse.score == 40
    assert sparse.bias == 0
