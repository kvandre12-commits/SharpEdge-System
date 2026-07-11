from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_grammar import build_break_state, build_dealer_gamma_state  # noqa: E402
from trade_permission import score_trade_permission  # noqa: E402


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


def _failed_breakout_bars() -> list[tuple]:
    return [
        (0, 99.60, 99.78, 99.55, 99.70, 1000),
        (1, 99.70, 99.92, 99.68, 99.88, 1100),
        (2, 99.88, 100.05, 99.82, 99.96, 1200),
        (3, 99.96, 100.35, 99.90, 100.18, 1600),
        (4, 100.18, 100.42, 99.92, 100.02, 1900),
        (5, 100.02, 100.08, 99.78, 99.84, 2100),
    ]


def _pa(bars: list[tuple], **overrides) -> dict:
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "rng_pos": 90.0,
        "day_chg": 0.8,
        "vwap": closes[-1] - 0.15,
        "vs_vwap": 0.16,
        "mom15": 0.4,
        "vol_mult": 1.3,
        "balance_high": closes[-1],
        "balance_low": closes[0],
        "position_in_balance": 1.0,
        "balance_state": "above",
        "balance_label": "ABOVE",
        "balance_width_pct": 0.35,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "test",
        "dominant_balance_reason": "test",
        "dominant_balance_previous_name": "test",
        "dominant_balance_flip": {},
        "balance_models": {},
        "balance_confluence": {},
        "balance_disagreement": {},
    }
    pa.update(overrides)
    return pa


def test_build_break_state_reads_level_state_engine_for_accepted_breakout():
    bars = _breakout_bars()
    state = build_break_state(
        bars,
        {"PDH": 100.0, "ORH": 99.8, "ORL": 98.8, "PDL": 98.5},
    )

    assert state == {
        "state": "accepted_breakout",
        "bias": "CALLS",
        "level_name": "PDH",
        "level_price": 100.0,
        "score": 72,
        "reason": "3 closes accepted above PDH 100.00",
    }


def test_positive_gamma_wall_gravity_does_not_require_pin():
    dealer = build_dealer_gamma_state(
        {"spot": 100.0},
        {"call_wall": 100.1, "put_wall": 95.0},
        {"regime": "positive"},
    )

    assert dealer["state"] == "positive_gamma_gravity"
    assert dealer["score"] == 38
    assert dealer["pin"] is None
    assert "pin unavailable" in dealer["reason"]
    assert "near call wall" in dealer["reason"]


def test_bucket_conditioned_spine_replaces_live_trigger_wait_lane():
    bars = _breakout_bars()
    levels = {"PDH": 105.0, "ORH": 104.0, "ORL": 98.8, "PDL": 98.5}

    card = score_trade_permission(
        bars,
        _pa(bars),
        levels,
        [],
        {"atm_iv": 0.18, "call_wall": 103.0, "put_wall": 98.0},
        {"regime": "positive", "pin": 100.5},
        {"premium_read": "rich"},
    )

    spine = card["bucket_conditioned_spine"]
    assert card["market_day"]["bucket"] == "range_balance_day"
    assert card["execution_flow"]["day_bucket"] == "range_balance_day"
    assert card["execution_grammar"] == {
        "mode": "full_stack",
        "authority_engine": "legacy",
    }
    assert spine["gate"] == card["trade_gate"]
    assert spine["score"] == card["trade_permission_score"]
    assert spine["recommended_action"] == "watch_edges"


def test_positive_gamma_breakout_no_longer_gets_capped_to_68():
    bars = _breakout_bars()
    levels = {"PDH": 100.0, "ORH": 99.8, "ORL": 98.8, "PDL": 98.5}

    card = score_trade_permission(
        bars,
        _pa(bars),
        levels,
        [],
        {"atm_iv": 0.18, "call_wall": 100.55, "put_wall": 98.0},
        {"regime": "positive", "pin": 100.55},
        {"premium_read": "rich"},
    )

    assert card["trade_permission_score"] == 71
    assert card["raw_vector_permission_score"] == 71
    assert card["trade_permission_score"] >= card["raw_vector_permission_score"]
    assert card["execution_grammar"] == {
        "mode": "full_stack",
        "authority_engine": "legacy",
    }


def test_negative_gamma_trend_day_promotes_candidate_calls_directly_from_spine():
    bars = _breakout_bars()
    levels = {"PDH": 100.0, "ORH": 99.8, "ORL": 98.8, "PDL": 98.5}

    card = score_trade_permission(
        bars,
        _pa(bars, vol_mult=1.6),
        levels,
        [],
        {"atm_iv": 0.18, "call_wall": 102.0, "put_wall": 98.0},
        {"regime": "negative", "pin": 99.0},
        {"premium_read": "cheap"},
    )

    spine = card["bucket_conditioned_spine"]
    assert card["market_day"]["bucket"] == "a_plus_trend_day"
    assert card["trade_permission_score"] == 84
    assert card["trade_gate"] == "PERMIT"
    assert card["bias"] == "CALLS"
    assert spine["recommended_action"] == "candidate_calls"


def test_failed_breakout_setup_card_drives_puts_bias_through_bucket_spine():
    bars = _breakout_bars()
    levels = {"PDH": 100.0, "ORH": 99.8, "ORL": 98.8, "PDL": 98.5}

    card = score_trade_permission(
        bars,
        _pa(bars),
        levels,
        [
            {
                "tag": "FAILED BREAKOUT",
                "bias": "PUTS (bearish)",
                "kind": "bad",
                "level_name": "PDH",
                "level_price": 100.0,
                "trigger_price": 100.5,
                "detail": "setup detector saw the bull trap",
            }
        ],
        {"atm_iv": 0.18, "call_wall": 100.55, "put_wall": 98.0},
        {"regime": "positive", "pin": 100.55},
        {"premium_read": "rich"},
    )

    spine = card["bucket_conditioned_spine"]
    assert card["market_day"]["bucket"] == "failed_breakout_short_day"
    assert card["trade_permission_score"] == 82
    assert card["trade_gate"] == "PERMIT"
    assert card["bias"] == "PUTS"
    assert spine["recommended_action"] == "candidate_puts"


def test_failed_breakout_bars_still_degrade_to_watch_edges_when_bucket_stays_rangey():
    bars = _failed_breakout_bars()
    levels = {"PDH": 100.0, "ORH": 100.0, "ORL": 98.8, "PDL": 98.5}

    card = score_trade_permission(
        bars,
        _pa(bars, spot=bars[-1][4], vs_vwap=-0.08, mom15=-0.12, rng_pos=78.0),
        levels,
        [],
        {"atm_iv": 0.18, "call_wall": 101.5, "put_wall": 98.0},
        {"regime": "positive", "pin": 100.0},
        {"premium_read": "rich"},
    )

    assert card["market_day"]["bucket"] == "range_balance_day"
    assert card["trade_permission_score"] == 56
    assert card["trade_gate"] == "BLOCK"
    assert card["bucket_conditioned_spine"]["recommended_action"] == "stand_down"
