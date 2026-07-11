from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_grammar import build_break_state  # noqa: E402
from failed_break_facts import failed_break_facts  # noqa: E402
from setups import detect_failed_breaks  # noqa: E402


def _failed_breakdown_bars() -> list[tuple]:
    return [
        (0, 100.02, 100.08, 99.96, 100.04, 1000),
        (1, 100.04, 100.06, 99.86, 99.94, 1100),
        (2, 99.94, 99.98, 99.74, 99.82, 1200),
        (3, 99.82, 100.08, 99.92, 100.04, 1300),
        (4, 100.04, 100.09, 99.98, 100.06, 1250),
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


def test_failed_break_facts_stay_deterministic_and_opinion_free():
    facts = failed_break_facts(
        _failed_breakdown_bars(),
        "ORL",
        100.0,
        recent_window=4,
    )

    assert facts == {
        "schema": "sharpedge.failed_break_facts.v1",
        "level_name": "ORL",
        "level_price": 100.0,
        "buffer": 0.1,
        "total_bars": 5,
        "latest_bar_index": 4,
        "recent_window": 4,
        "recent_window_used": 4,
        "current_close": 100.06,
        "current_close_above_level": True,
        "current_close_below_level": False,
        "recent_high": 100.09,
        "recent_low": 99.74,
        "recent_breach_above": False,
        "recent_breach_below": True,
        "breach_above_latest_index": None,
        "breach_above_highest_high": None,
        "breach_above_extension_pct": None,
        "reject_below_level_index": None,
        "bars_since_reject_below_level": None,
        "breach_below_latest_index": 2,
        "breach_below_deepest_low": 99.74,
        "breach_below_depth_pct": 0.2600000000000051,
        "reclaim_above_level_index": 3,
        "bars_since_reclaim_above_level": 1,
    }
    assert {"score", "bias", "tag", "state", "trade_bias"}.isdisjoint(facts)


def test_detect_failed_breaks_keeps_same_failed_breakdown_setup_card():
    cards = detect_failed_breaks(_failed_breakdown_bars(), {"ORL": 100.0})

    assert cards == [
        {
            "tag": "FAILED BREAKDOWN",
            "bias": "CALLS (bullish)",
            "kind": "ok",
            "detail": "reclaimed ORL $100.00 1m ago after stabbing $99.74 (-0.26% below) - bear trap",
            "score": 5.260000000000005,
            "level_name": "ORL",
            "level_price": 100.0,
            "trigger_price": 99.74,
            "bars_ago": 1,
        }
    ]


def test_build_break_state_keeps_same_failed_breakout_interpretation():
    state = build_break_state(
        _failed_breakout_bars(),
        {"PDH": 100.0, "ORH": 100.0, "ORL": 98.8, "PDL": 98.5},
    )

    assert state == {
        "state": "failed_breakout",
        "bias": "PUTS",
        "level_name": "PDH",
        "level_price": 100.0,
        "trigger_price": 100.42,
        "score": 88,
        "reason": "buyers trapped above PDH 100.00; rejected from 100.42",
    }
