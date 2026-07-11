from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from balance import (  # noqa: E402
    balance_label,
    build_balance_stack,
    position_in_balance,
    recent_balance_context_from_rows,
)
from make_cockpit import read_price_action  # noqa: E402


def test_position_in_balance_normalizes_and_clamps():
    assert position_in_balance(108.0, 100.0, 110.0) == 0.8
    assert position_in_balance(95.0, 100.0, 110.0) == 0.0
    assert position_in_balance(115.0, 100.0, 110.0) == 1.0
    assert position_in_balance(100.0, 100.0, 100.0) == 0.5


def test_balance_label_reads_cleanly_for_fast_decisions():
    assert balance_label(0.05) == "BOTTOM"
    assert balance_label(0.50) == "MIDDLE"
    assert balance_label(0.95) == "TOP"


def test_recent_balance_detects_above_breakout():
    rows = [(idx, 100.0, 100.2, 99.8, 100.0, 1_000) for idx in range(19)] + [
        (19, 100.0, 102.5, 99.9, 102.0, 2_000),
    ]

    breakout = recent_balance_context_from_rows(rows)

    assert breakout["balance_state"] == "above"
    assert breakout["position_in_balance"] == 1.0
    assert breakout["balance_window_bars"] == 19


def test_build_balance_stack_uses_opening_balance_early():
    rows = [(minute, 100.0, 100.2, 99.8, 100.0, 1_000) for minute in range(25)] + [
        (25, 100.0, 100.4, 99.9, 100.3, 1_200),
    ]

    balance = build_balance_stack(rows)

    assert balance["dominant_balance_name"] == "opening_balance"
    assert balance["balance_reference"].startswith("opening_")
    assert balance["dominant_balance_flip"]["flipped"] is False


def test_build_balance_stack_uses_recent_balance_midday():
    rows = [
        (120 + minute, 100.0, 100.15, 99.85, 100.0 + minute * 0.01, 900)
        for minute in range(25)
    ]

    balance = build_balance_stack(rows)

    assert balance["dominant_balance_name"] == "recent_balance"
    assert balance["balance_reference"].startswith("recent_")
    assert balance["dominant_balance_flip"]["flipped"] is False


def test_build_balance_stack_uses_value_balance_late_day():
    rows = [
        (330 + minute, 100.0, 100.2, 99.8, 100.0 + (minute % 5) * 0.05, 950)
        for minute in range(25)
    ]

    balance = build_balance_stack(rows)

    assert balance["dominant_balance_name"] == "value_balance"
    assert balance["balance_reference"].startswith("value_")
    assert balance["dominant_balance_flip"]["flipped"] is False


def test_balance_stack_reports_confluence_disagreement_and_flip():
    rows = [
        (58, 100.0, 100.2, 99.8, 100.15, 1_000),
        (59, 100.15, 100.3, 100.0, 100.25, 1_000),
        (60, 100.25, 100.45, 100.2, 100.42, 1_100),
    ]

    balance = build_balance_stack(rows)

    assert balance["dominant_balance_name"] == "recent_balance"
    assert balance["dominant_balance_flip"]["flipped"] is True
    assert balance["dominant_balance_flip"]["from"] == "opening_balance"
    assert balance["balance_confluence"]["bias"] == "CALLS"
    assert balance["balance_disagreement"]["has_disagreement"] is False


def test_read_price_action_exposes_dominant_balance_fields():
    rows = [
        (150, 100.0, 100.2, 99.8, 100.0, 1_000),
        (151, 100.0, 110.2, 99.9, 110.0, 1_200),
        (152, 110.0, 110.1, 104.9, 105.0, 1_100),
    ]

    pa = read_price_action(rows)

    assert pa["balance_low"] == 99.8
    assert pa["balance_high"] == 110.2
    assert pa["position_in_balance"] == 0.5
    assert pa["dominant_balance_name"] == "recent_balance"
    assert pa["balance_reference"] == "recent_2_bar"
    assert pa["balance_confluence"]["state"] in {"lean", "aligned", "neutral", "disagreement"}
    assert "has_disagreement" in pa["balance_disagreement"]
    assert "flipped" in pa["dominant_balance_flip"]
    assert pa["session_position_in_range"] == 0.5
    assert pa["rng_pos"] == 50.0
