from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from spy_scalp_chart import render_spy_scalp_chart_svg
from spy_scalp_dashboard import (
    build_spy_scalp_packet,
    render_spy_scalp_dashboard_html,
)


def _call_break_rows(
    count: int = 60,
) -> list[tuple[int, float, float, float, float, int]]:
    rows: list[tuple[int, float, float, float, float, int]] = []
    close = 100.0
    for minute in range(count):
        if minute < 15:
            close = 100.0 + (0.01 if minute % 2 else -0.01)
            high = 100.08
            low = 99.94
            volume = 1000
        elif minute < count - 5:
            drift = [0.012, -0.008, 0.014, -0.006][minute % 4]
            close += drift
            high = close + 0.04
            low = close - 0.05
            volume = 1100
        else:
            drift = [0.02, -0.015, 0.025, -0.01, 0.015][minute - (count - 5)]
            close += drift
            high = close + 0.04
            low = close - 0.03
            volume = 2400
        open_ = close - 0.01
        rows.append((minute, open_, high, low, close, volume))
    return rows


def _put_break_rows(
    count: int = 60,
) -> list[tuple[int, float, float, float, float, int]]:
    rows: list[tuple[int, float, float, float, float, int]] = []
    close = 100.0
    for minute in range(count):
        if minute < 15:
            close = 100.0 + (0.01 if minute % 2 else -0.01)
            high = 100.06
            low = 99.92
            volume = 1000
        elif minute < count - 5:
            drift = [-0.012, 0.008, -0.014, 0.006][minute % 4]
            close += drift
            high = close + 0.05
            low = close - 0.04
            volume = 1100
        else:
            drift = [-0.02, 0.015, -0.025, 0.01, -0.015][minute - (count - 5)]
            close += drift
            high = close + 0.03
            low = close - 0.04
            volume = 2400
        open_ = close + 0.01
        rows.append((minute, open_, high, low, close, volume))
    return rows


def test_spy_scalp_packet_arms_clean_call_or15_break() -> None:
    rows = _call_break_rows()
    packet = build_spy_scalp_packet(
        rows,
        {"spot": rows[-1][4], "vwap": rows[-1][4] - 0.05},
        {
            "exp": "2026-07-17",
            "atm_strike": 100,
            "atm_call_delta": 0.51,
            "atm_call_spread_pct": 0.03,
        },
        "10:28:00",
    )

    assert packet["bias"] == "CALLS"
    assert packet["trigger"]["state"] == "armed"
    assert packet["status"] == "SCALP SETUP ARMED"
    assert packet["contract"]["delta_ok"] is True
    assert packet["score"] >= 75


def test_spy_scalp_packet_supports_put_break_contract_side() -> None:
    rows = _put_break_rows()
    packet = build_spy_scalp_packet(
        rows,
        {"spot": rows[-1][4], "vwap": rows[-1][4] + 0.05},
        {
            "exp": "2026-07-17",
            "atm_strike": 100,
            "atm_put_delta": -0.49,
            "atm_put_spread_pct": 0.04,
        },
        "10:28:00",
    )

    assert packet["bias"] == "PUTS"
    assert packet["trigger"]["state"] == "armed"
    assert packet["contract"]["side"] == "put"
    assert packet["contract"]["delta"] == 0.49


def test_spy_scalp_packet_blocks_midday_even_when_triggered() -> None:
    rows = _call_break_rows(170)
    packet = build_spy_scalp_packet(
        rows,
        {"spot": rows[-1][4], "vwap": rows[-1][4] - 0.05},
        {"atm_call_spread_pct": 0.03},
        "12:10:00",
    )

    assert packet["time_window"]["state"] == "avoid"
    assert packet["status"] == "AVOID MIDDAY CHOP"


def test_spy_scalp_dashboard_html_renders_rules_and_packet() -> None:
    rows = _call_break_rows()
    packet = build_spy_scalp_packet(
        rows,
        {"spot": rows[-1][4], "vwap": rows[-1][4] - 0.05},
        {"atm_call_spread_pct": 0.03},
        "10:28:00",
    )

    markup = render_spy_scalp_dashboard_html(packet)

    assert "SPY Options Scalp" in markup
    assert "If-Then Checklist" in markup
    assert "Entry Map" in markup
    assert "spy_scalp_chart.svg" in markup
    assert "+15% to +30% option premium" in markup
    assert "limit order only" in markup
    assert "sharpedge.spy_scalp_dashboard.v1" in markup


def test_spy_scalp_chart_renders_levels_channels_and_trigger_context() -> None:
    rows = _call_break_rows()
    packet = build_spy_scalp_packet(
        rows,
        {"spot": rows[-1][4], "vwap": rows[-1][4] - 0.05},
        {"atm_call_spread_pct": 0.03},
        "10:28:00",
    )

    svg = render_spy_scalp_chart_svg(rows, packet)

    assert "SPY scalp entry map" in svg
    assert "ORH" in svg
    assert "ORL" in svg
    assert "VWAP" in svg
    assert "EMA9" in svg
    assert "EMA20" in svg
    assert "20ch high" in svg
    assert "8ch high" in svg
    assert "trigger armed" in svg
