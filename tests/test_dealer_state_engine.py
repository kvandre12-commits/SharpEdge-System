from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from dealer_state_engine import build_dealer_state
from trade_permission import ExecutionVectorEngine


def test_dealer_state_detects_positive_gamma_gravity_without_pin():
    packet = build_dealer_state(
        {"spot": 100.0},
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "positive", "gamma_data_quality": "ok", "dte": 0},
    )

    assert packet["state"] == "positive_gamma_gravity"
    assert packet["pin_state"]["state"] == "pin_unavailable"
    assert packet["wall_state"]["state"] == "near_call_wall"
    assert "pin unavailable" in packet["reason"]
    assert "near call wall" in packet["reason"]


def test_dealer_state_detects_negative_gamma_expansion_without_premium_read():
    packet = build_dealer_state(
        {"spot": 100.0},
        {"call_wall": 102.0, "put_wall": 98.0},
        {"regime": "negative", "pin": 99.0, "gamma_data_quality": "ok", "dte": 0},
    )

    assert packet["state"] == "negative_gamma_expansion"
    assert packet["gamma_state"]["state"] == "gamma_expansion"
    assert "negative gamma/OI proxy may support expansion" in packet["reason"]


def test_dealer_state_emits_explicit_unknown_when_gamma_quality_is_weak():
    packet = build_dealer_state(
        {"spot": 100.0},
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "unknown", "gamma_data_quality": "missing"},
    )

    assert packet["state"] == "dealer_unknown"
    assert packet["bias"] == "NEUTRAL"
    assert packet["gamma_state"]["state"] == "gamma_unknown"
    assert "dealer unknown" in packet["reason"]


def test_execution_vector_engine_dealer_score_uses_dealer_state_engine():
    engine = ExecutionVectorEngine()

    gravity = engine.build_parts(
        [
            (0, 100.0, 100.2, 99.9, 100.0, 1000),
            (1, 100.0, 100.2, 99.9, 100.0, 1000),
            (2, 100.0, 100.2, 99.9, 100.0, 1000),
        ],
        {"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.0, "mom15": 0.0},
        {"ORH": 101.0, "ORL": 99.0, "PDC": 100.0},
        [],
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "positive", "gamma_data_quality": "ok", "dte": 0},
        {},
    )["dealer_gamma_score"]
    unknown = engine.build_parts(
        [
            (0, 100.0, 100.2, 99.9, 100.0, 1000),
            (1, 100.0, 100.2, 99.9, 100.0, 1000),
            (2, 100.0, 100.2, 99.9, 100.0, 1000),
        ],
        {"spot": 100.0, "vwap": 100.0, "vs_vwap": 0.0, "mom15": 0.0},
        {"ORH": 101.0, "ORL": 99.0, "PDC": 100.0},
        [],
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "unknown", "gamma_data_quality": "missing"},
        {},
    )["dealer_gamma_score"]

    assert gravity.score == 38
    assert "pinning" in gravity.reason
    assert unknown.score == 40
    assert "dealer unknown" in unknown.reason


def test_dealer_state_trusts_explicit_regime_when_quality_missing():
    # Policy: an explicit positive/negative regime is trusted even when the
    # gamma feed omits a quality field. Only expired contracts, an unknown
    # regime, or an explicitly unusable quality read fall back to unknown.
    packet = build_dealer_state(
        {"spot": 100.0},
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "positive", "dte": 0},
    )

    assert packet["state"] == "positive_gamma_gravity"
    assert packet["gamma_state"]["quality"] == "missing"


def test_dealer_state_rejects_expired_gamma_packet():
    packet = build_dealer_state(
        {"spot": 100.0},
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "positive", "gamma_data_quality": "ok", "dte": -1},
    )

    assert packet["state"] == "dealer_unknown"
    assert packet["gamma_state"]["state"] == "gamma_unknown"
    assert packet["gamma_state"]["reason"] == "gamma contract is expired"
