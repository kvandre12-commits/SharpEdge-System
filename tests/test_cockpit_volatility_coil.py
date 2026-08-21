from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from make_cockpit import chart_svg, read_options, write_signal
from monthly_context_chart import (
    build_monthly_context_svg,
    derive_monthly_levels,
    summarize_monthly_context,
)
from setups import detect_volatility_coil, read_volatility_structure
from weekly_context_chart import (
    build_weekly_context_svg,
    derive_today_carry_levels,
    summarize_weekly_context,
)


def _rows_from_five_minute_closes(
    closes: list[float],
) -> list[tuple[int, float, float, float, float, int]]:
    rows = []
    minute = 0
    prev = closes[0]
    for close in closes:
        start = prev
        for step in range(5):
            open_ = start if step == 0 else rows[-1][4]
            frac = (step + 1) / 5
            current = start + (close - start) * frac
            high = max(open_, current) + 0.01
            low = min(open_, current) - 0.01
            rows.append((minute, open_, high, low, current, 1_000))
            minute += 1
        prev = close
    return rows


def _coil_rows() -> list[tuple[int, float, float, float, float, int]]:
    history = [100 + (0.55 if idx % 2 else -0.55) for idx in range(40)]
    impulse = [100.0, 99.82, 99.64, 99.46, 99.28, 99.18, 99.12, 99.10]
    coil = [
        99.12,
        99.13,
        99.14,
        99.13,
        99.14,
        99.15,
        99.14,
        99.15,
        99.14,
        99.15,
        99.14,
        99.15,
    ]
    return _rows_from_five_minute_closes(history + impulse + coil)


def test_detect_volatility_coil_flags_post_selloff_compression():
    rows = _coil_rows()
    pa = {"vs_vwap": -0.20}

    state = read_volatility_structure(rows, pa)
    card = detect_volatility_coil(rows, pa, state)

    assert state["volatility_state"] == "squeeze"
    assert state["structure_state"] == "channel_breakout_setup"
    assert state["bias"] == "neutral_to_bearish"
    assert state["compression"] is True
    assert state["narrow_channel"] is True
    assert state["impulse_down"] is True
    assert state["coil"] is True
    assert state["prior_impulse_down_pct"] > 0.7
    assert state["channel_pct"] < 0.3
    assert card is not None
    assert card["tag"] == "POST-SELLOFF COIL"
    assert "trigger above" in card["detail"]
    assert "trigger below" in card["detail"]


def test_write_signal_persists_setup_cards_and_volatility_structure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    coil_state = {
        "volatility_state": "squeeze",
        "structure_state": "channel_breakout_setup",
        "bias": "neutral_to_bearish",
        "coil": True,
        "trigger_high": 99.16,
        "trigger_low": 99.12,
    }
    gcard = {
        "tag": "RUNNER DAY (wheee)",
        "bias": "RIDE momentum - go directional, breakouts run",
        "kind": "info",
        "detail": "negative gamma day",
        "score": 55,
    }
    coil_card = {
        "tag": "POST-SELLOFF COIL",
        "bias": "NEUTRAL-to-BEARISH | break high = reclaim, lose low = continuation",
        "kind": "warn",
        "detail": "channel is tight",
        "score": 63,
    }

    write_signal(
        pa={
            "spot": 99.14,
            "day_chg": -0.8,
            "vwap": 99.30,
            "vs_vwap": -0.16,
            "balance_high": 99.30,
            "balance_low": 99.10,
            "position_in_balance": 0.20,
            "balance_state": "inside",
            "balance_label": "BOTTOM",
            "balance_width_pct": 0.202,
            "balance_window_bars": 20,
            "balance_reference": "recent_20_bar",
            "dominant_balance_name": "recent_balance",
            "dominant_balance_reason": "mid-session: the active recent box matters most",
            "dominant_balance_previous_name": "recent_balance",
            "dominant_balance_flip": {
                "flipped": False,
                "from": "recent_balance",
                "to": "recent_balance",
                "reason": "dominant balance lens unchanged at recent_balance",
            },
            "balance_models": {
                "opening_balance": {"balance_reference": "opening_30m"},
                "recent_balance": {"balance_reference": "recent_20_bar"},
                "value_balance": {"balance_reference": "value_20_bar"},
            },
            "balance_confluence": {
                "state": "lean",
                "score": 58,
                "bias": "CALLS",
                "agreement_count": 1,
                "aligned_models": ["recent_balance"],
                "reason": "1 balance lens(es) align calls: recent_balance",
            },
            "balance_disagreement": {
                "has_disagreement": False,
                "bullish_models": ["recent_balance"],
                "bearish_models": [],
                "neutral_models": ["opening_balance", "value_balance"],
                "reason": "balance lenses are not fighting each other",
            },
            "session_position_in_range": 0.18,
            "rng_pos": 18.0,
            "mom15": -0.04,
            "vol_mult": 0.72,
        },
        op={
            "call_wall": 100.0,
            "put_wall": 99.0,
            "pcr": 1.11,
            "atm_iv": 0.19,
            "exp": "2026-06-23",
        },
        gp={
            "exp": "2026-06-23",
            "dte": 1,
            "regime": "negative",
            "net_gamma": -123.4567,
            "gamma_data_quality": "ok",
            "pin": 99.0,
            "pin_dist": -0.1413,
            "max_pain": 99.0,
        },
        gcard=gcard,
        signal_ts="2026-06-25T11:30:00",
        setups=[gcard, coil_card],
        micro={"ch_width_pct": 0.04},
        magnitude={"premium_read": "rich"},
        permission={
            "trade_gate": "CAUTION",
            "trade_permission_score": 61,
            "bias": "NEUTRAL",
            "setup_conviction": {
                "setup_tag": "FAILED BREAKDOWN",
                "entry_gate": {
                    "tag": "FAILED BREAKDOWN",
                    "bias": "CALLS (bullish)",
                    "gate_id": "failed_breakdown_reclaim",
                },
                "context_gate": {
                    "tag": "RUNNER DAY (wheee)",
                    "bias": "RIDE momentum - go directional, breakouts run",
                    "gate_id": "runner_day_directional_continuation",
                },
            },
        },
        volatility_structure=coil_state,
        target_plan={
            "label": "Channel lo",
            "price": 99.12,
            "reason": "continuation setup targets directional expansion",
        },
        decision_receipt={"schema": "sharpedge.decision_receipt.v1", "permission": 61},
        permission_score_trend={
            "schema": "sharpedge.permission_score_trend.v1",
            "direction": "new",
        },
        source_freshness={
            "price": {"provider": "yahoo", "last_bar_utc": "2026-06-25T15:30:00+00:00"},
            "options": {
                "provider": "cboe",
                "last_trade_time_raw": "2026-06-25T11:29:00",
            },
        },
        reference_levels={
            "ORH": 100.0,
            "ORL": 99.0,
            "PDH": 101.0,
            "PDL": 98.5,
            "PDC": 99.4,
        },
    )

    signal_path = tmp_path / "SharpEdge-System/outputs/signal.json"
    payload = json.loads(signal_path.read_text(encoding="utf-8"))

    assert payload["gamma_exp"] == "2026-06-23"
    assert payload["gamma_dte"] == 1
    assert payload["gamma_regime"] == "negative"
    assert payload["gamma_net"] == -123.4567
    assert payload["gamma_data_quality"] == "ok"
    assert payload["pin"] == 99.0
    assert payload["pin_dist"] == -0.1413
    assert payload["max_pain"] == 99.0
    assert payload["setup_tag"] == gcard["tag"]
    assert payload["entry_setup_tag"] == "FAILED BREAKDOWN"
    assert payload["entry_setup_bias"] == "CALLS (bullish)"
    assert payload["context_setup_tag"] == "RUNNER DAY (wheee)"
    assert payload["setup_cards"][1]["tag"] == "POST-SELLOFF COIL"
    assert payload["volatility_state"] == "squeeze"
    assert payload["structure_state"] == "channel_breakout_setup"
    assert payload["volatility_structure"]["coil"] is True
    assert payload["target_plan"]["label"] == "Channel lo"
    assert payload["decision_receipt"]["permission"] == 61
    assert payload["permission_score_trend"]["direction"] == "new"
    assert payload["edge_token_position"] == {}
    assert payload["source_freshness"]["price"]["provider"] == "yahoo"
    assert payload["source_freshness"]["options"]["provider"] == "cboe"
    assert payload["source_freshness"]["signal_generated_at"] == "2026-06-25T11:30:00"
    assert payload["reference_levels"]["ORH"] == 100.0
    assert payload["reference_levels"]["PDC"] == 99.4
    assert payload["balance_high"] == 99.3
    assert payload["balance_low"] == 99.1
    assert payload["position_in_balance"] == 0.2
    assert payload["balance_state"] == "inside"
    assert payload["balance_reference"] == "recent_20_bar"
    assert payload["dominant_balance_name"] == "recent_balance"
    assert payload["dominant_balance_flip"]["flipped"] is False
    assert payload["balance_confluence"]["state"] == "lean"
    assert payload["balance_disagreement"]["has_disagreement"] is False
    assert (
        payload["balance_models"]["value_balance"]["balance_reference"]
        == "value_20_bar"
    )
    assert payload["session_position_in_range"] == 0.18


def test_chart_svg_draws_reference_levels_without_failed_break_signal_by_default():
    rows = [
        (0, 100.0, 100.2, 99.8, 100.0, 1000),
        (1, 100.0, 100.4, 99.7, 100.3, 1000),
        (2, 100.3, 100.5, 99.6, 99.9, 1000),
    ]
    pa = {"vwap": 100.1, "vs_vwap": -0.1}
    levels = {"ORL": 99.75, "PDL": 99.5, "PDC": 100.0}
    setups = [
        {
            "tag": "FAILED BREAKDOWN",
            "level_name": "ORL",
            "level_price": 99.75,
            "trigger_price": 99.4,
        }
    ]

    svg = chart_svg(rows, pa, levels, setups)

    assert 'viewBox="0 0 1000 576"' in svg
    assert "ORL 99.75" in svg
    assert "PDL 99.50" in svg
    assert "PDC 100.00" in svg
    assert "TRIGGER 99.40" not in svg

    debug_svg = chart_svg(rows, pa, levels, setups, show_signal_overlays=True)
    assert "TRIGGER 99.40" in debug_svg


def test_chart_svg_renders_channel_levels_without_logic_badge_by_default():
    rows = [
        (0, 100.0, 100.2, 99.8, 100.0, 1000),
        (1, 100.0, 100.5, 99.9, 100.3, 1000),
        (2, 100.3, 100.7, 100.1, 100.6, 1000),
    ]
    pa = {"vwap": 100.2, "vs_vwap": 0.4}

    svg = chart_svg(
        rows,
        pa,
        {},
        [],
        {
            "channel_low": 99.8,
            "channel_high": 100.8,
            "channel_pct": 0.996,
            "channel_slope_pct": 0.123,
            "structure_state": "narrow_channel",
            "volatility_state": "contraction",
        },
    )

    assert "CHANNEL LOGIC" not in svg
    assert "PRESSING CHANNEL HIGH" not in svg
    assert "CHANNEL MID 100.30" in svg

    debug_svg = chart_svg(
        rows,
        pa,
        {},
        [],
        {
            "channel_low": 99.8,
            "channel_high": 100.8,
            "channel_pct": 0.996,
            "channel_slope_pct": 0.123,
            "structure_state": "narrow_channel",
            "volatility_state": "contraction",
        },
        show_signal_overlays=True,
    )
    assert "CHANNEL LOGIC" in debug_svg
    assert "PRESSING CHANNEL HIGH" in debug_svg
    assert "pos 80%" in debug_svg
    assert "width 0.996%" in debug_svg
    assert "slope +0.123%" in debug_svg


def test_chart_svg_hides_level_state_strip_by_default():
    rows = [
        (0, 100.0, 100.2, 99.8, 100.0, 1000),
        (1, 100.0, 100.4, 99.7, 100.3, 1000),
        (2, 100.3, 100.5, 100.0, 100.4, 1000),
    ]
    pa = {"vwap": 100.1, "vs_vwap": 0.1}

    svg = chart_svg(
        rows,
        pa,
        {"ORH": 100.25, "ORL": 99.75, "PDC": 100.0},
        [],
        level_states={
            "ORH": {"event_state": "accepted_above_resistance"},
            "ORL": {"event_state": "holding_above_support"},
            "PDC": {"event_state": "accepted_above_reference"},
        },
    )

    assert "LEVEL STATES" not in svg
    assert "ACCEPT &gt; R" not in svg and "ACCEPT > R" not in svg
    assert "HOLD SUPPORT" not in svg

    debug_svg = chart_svg(
        rows,
        pa,
        {"ORH": 100.25, "ORL": 99.75, "PDC": 100.0},
        [],
        level_states={
            "ORH": {"event_state": "accepted_above_resistance"},
            "ORL": {"event_state": "holding_above_support"},
            "PDC": {"event_state": "accepted_above_reference"},
        },
        show_signal_overlays=True,
    )
    assert "LEVEL STATES" in debug_svg
    assert "ACCEPT &gt; R" in debug_svg or "ACCEPT > R" in debug_svg
    assert "HOLD SUPPORT" in debug_svg
    assert "ACCEPT &gt; REF" in debug_svg or "ACCEPT > REF" in debug_svg


def test_derive_today_carry_levels_finds_primary_and_secondary_swings():
    rows = [
        (0, 99.8, 100.0, 99.0, 99.6, 1000),
        (1, 99.6, 101.0, 99.5, 100.4, 1000),
        (2, 100.4, 102.0, 100.0, 101.5, 1000),
        (3, 101.5, 104.0, 101.0, 103.6, 1000),
        (4, 103.6, 101.0, 99.0, 100.0, 1000),
        (5, 100.0, 100.0, 97.0, 98.4, 1000),
        (6, 98.4, 101.0, 98.5, 100.1, 1000),
        (7, 100.1, 103.0, 100.0, 102.2, 1000),
        (8, 102.2, 101.0, 99.5, 100.0, 1000),
        (9, 100.0, 102.0, 98.0, 99.2, 1000),
        (10, 99.2, 101.0, 99.5, 100.3, 1000),
        (11, 100.3, 100.5, 99.8, 100.1, 1000),
    ]

    levels = derive_today_carry_levels(rows)
    by_name = {level["name"]: level["price"] for level in levels}

    assert by_name == {"H1": 104.0, "LH1": 103.0, "HL1": 98.0, "L1": 97.0}


def test_build_weekly_context_svg_draws_carry_levels():
    recent_rows = [
        {"date": "2026-06-23", "close": 729.8},
        {"date": "2026-06-23", "close": 730.4},
        {"date": "2026-06-23", "close": 731.1},
        {"date": "2026-06-24", "close": 731.3},
        {"date": "2026-06-24", "close": 732.0},
        {"date": "2026-06-24", "close": 731.7},
        {"date": "2026-06-25", "close": 732.4},
        {"date": "2026-06-25", "close": 733.2},
        {"date": "2026-06-25", "close": 734.3},
    ]
    carry_levels = [
        {"name": "H1", "price": 734.9, "session_index": 50},
        {"name": "LH1", "price": 734.2, "session_index": 60},
        {"name": "HL1", "price": 732.1, "session_index": 75},
        {"name": "L1", "price": 731.4, "session_index": 20},
    ]

    svg = build_weekly_context_svg(
        recent_rows, carry_levels, symbol="SPY", lookback_days=5
    )

    assert 'viewBox="0 0 1000 360"' in svg
    assert "SPY 5-day carry map" in svg
    assert "today = bright blue • older days fade" in svg
    assert "TODAY" in svg
    assert "H1 734.90" in svg
    assert "LH1 734.20" in svg
    assert "HL1 732.10" in svg
    assert "L1 731.40" in svg


def test_derive_monthly_levels_uses_current_month_open_and_prior_month_rails():
    daily_rows = [
        {
            "date": "2026-05-28",
            "open": 98.2,
            "high": 99.4,
            "low": 97.9,
            "close": 99.1,
            "volume": 1,
        },
        {
            "date": "2026-05-29",
            "open": 99.1,
            "high": 100.8,
            "low": 98.4,
            "close": 99.7,
            "volume": 1,
        },
        {
            "date": "2026-06-02",
            "open": 99.2,
            "high": 99.9,
            "low": 98.8,
            "close": 99.5,
            "volume": 1,
        },
        {
            "date": "2026-06-03",
            "open": 99.5,
            "high": 100.1,
            "low": 99.1,
            "close": 100.0,
            "volume": 1,
        },
    ]

    levels = derive_monthly_levels(daily_rows)
    by_name = {level["name"]: level["price"] for level in levels}

    assert by_name == {"MOPEN": 99.2, "PMH": 100.8, "PMC": 99.7, "PML": 97.9}


def test_build_monthly_context_svg_draws_prior_month_levels():
    daily_rows = [
        {
            "date": "2026-01-05",
            "open": 94.8,
            "high": 95.4,
            "low": 94.3,
            "close": 95.0,
            "volume": 1,
        },
        {
            "date": "2026-02-05",
            "open": 95.1,
            "high": 96.0,
            "low": 94.9,
            "close": 95.8,
            "volume": 1,
        },
        {
            "date": "2026-03-05",
            "open": 96.0,
            "high": 97.0,
            "low": 95.7,
            "close": 96.7,
            "volume": 1,
        },
        {
            "date": "2026-04-07",
            "open": 97.1,
            "high": 98.1,
            "low": 96.8,
            "close": 97.9,
            "volume": 1,
        },
        {
            "date": "2026-05-28",
            "open": 98.2,
            "high": 99.4,
            "low": 97.9,
            "close": 99.1,
            "volume": 1,
        },
        {
            "date": "2026-05-29",
            "open": 99.1,
            "high": 100.8,
            "low": 98.4,
            "close": 99.7,
            "volume": 1,
        },
        {
            "date": "2026-06-02",
            "open": 99.2,
            "high": 99.9,
            "low": 98.8,
            "close": 99.5,
            "volume": 1,
        },
        {
            "date": "2026-06-03",
            "open": 99.5,
            "high": 100.1,
            "low": 99.1,
            "close": 100.0,
            "volume": 1,
        },
    ]
    monthly_levels = [
        {"name": "PMH", "price": 100.8, "month": "2026-05"},
        {"name": "MOPEN", "price": 99.2, "month": "2026-06"},
        {"name": "PMC", "price": 99.7, "month": "2026-05"},
        {"name": "PML", "price": 97.9, "month": "2026-05"},
    ]

    svg = build_monthly_context_svg(
        daily_rows, monthly_levels, symbol="SPY", lookback_months=6
    )

    assert 'viewBox="0 0 1000 340"' in svg
    assert "SPY 6-month structure" in svg
    assert "THIS MONTH" in svg
    assert "PMH 100.80" in svg
    assert "MOPEN 99.20" in svg
    assert "PMC 99.70" in svg
    assert "PML 97.90" not in svg


def test_summarize_monthly_context_surfaces_plain_english_structure_read():
    daily_rows = [
        {
            "date": "2026-01-05",
            "open": 94.8,
            "high": 95.4,
            "low": 94.3,
            "close": 95.0,
            "volume": 1,
        },
        {
            "date": "2026-02-05",
            "open": 95.1,
            "high": 96.0,
            "low": 94.9,
            "close": 95.8,
            "volume": 1,
        },
        {
            "date": "2026-03-05",
            "open": 96.0,
            "high": 97.0,
            "low": 95.7,
            "close": 96.7,
            "volume": 1,
        },
        {
            "date": "2026-04-07",
            "open": 97.1,
            "high": 98.1,
            "low": 96.8,
            "close": 97.9,
            "volume": 1,
        },
        {
            "date": "2026-05-28",
            "open": 98.2,
            "high": 99.4,
            "low": 97.9,
            "close": 99.1,
            "volume": 1,
        },
        {
            "date": "2026-05-29",
            "open": 99.1,
            "high": 100.8,
            "low": 98.4,
            "close": 99.7,
            "volume": 1,
        },
        {
            "date": "2026-06-02",
            "open": 99.2,
            "high": 99.9,
            "low": 98.8,
            "close": 99.5,
            "volume": 1,
        },
        {
            "date": "2026-06-03",
            "open": 99.5,
            "high": 100.1,
            "low": 99.1,
            "close": 100.0,
            "volume": 1,
        },
    ]
    monthly_levels = [
        {"name": "PMH", "price": 100.8, "month": "2026-05"},
        {"name": "MOPEN", "price": 99.2, "month": "2026-06"},
        {"name": "PMC", "price": 99.7, "month": "2026-05"},
        {"name": "PML", "price": 97.9, "month": "2026-05"},
    ]

    summary = summarize_monthly_context(
        daily_rows,
        monthly_levels,
        spot=100.0,
        symbol="SPY",
        lookback_months=6,
    )

    assert summary["lookback_months"] == 6
    assert summary["kind"] == "ok"
    assert (
        summary["headline"] == "Holding above monthly value inside the upper month band"
    )
    assert "above MOPEN $99.20 and PMC $99.70" in summary["detail"]


def test_summarize_weekly_context_surfaces_plain_english_structure_read():
    recent_rows = [
        {"date": "2026-06-23", "close": 729.8},
        {"date": "2026-06-23", "close": 730.4},
        {"date": "2026-06-24", "close": 731.3},
        {"date": "2026-06-24", "close": 732.0},
        {"date": "2026-06-25", "close": 732.4},
        {"date": "2026-06-25", "close": 733.2},
        {"date": "2026-06-25", "close": 734.3},
    ]
    carry_levels = [
        {"name": "H1", "price": 734.9, "session_index": 50},
        {"name": "LH1", "price": 734.2, "session_index": 60},
        {"name": "HL1", "price": 732.1, "session_index": 75},
        {"name": "L1", "price": 731.4, "session_index": 20},
    ]

    summary = summarize_weekly_context(
        recent_rows,
        carry_levels,
        spot=734.3,
        symbol="SPY",
        lookback_days=5,
    )

    assert summary["lookback_days"] == 3
    assert summary["kind"] == "ok"
    assert summary["headline"] == "Holding the upper carry shelf beneath H1"
    assert "between LH1 $734.20 and H1 $734.90" in summary["detail"]
    assert summary["legend"][0]["label"] == "H1 peak"


def test_read_options_surfaces_volume_and_atm_contract_detail():
    expiry = __import__("datetime").date(2026, 6, 25)
    book = {
        expiry: {
            99.0: {
                "C": {
                    "open_interest": 120,
                    "volume": 15,
                    "iv": 0.19,
                    "delta": 0.52,
                    "theta": -0.08,
                    "vega": 0.11,
                    "rho": 0.03,
                    "theo": 1.31,
                    "last_trade_price": 1.35,
                    "bid": 1.2,
                    "ask": 1.4,
                },
                "P": {
                    "open_interest": 80,
                    "volume": 22,
                    "iv": 0.21,
                    "delta": -0.48,
                    "theta": -0.07,
                    "vega": 0.1,
                    "rho": -0.02,
                    "theo": 1.21,
                    "last_trade_price": 1.18,
                    "bid": 1.1,
                    "ask": 1.3,
                },
            },
            100.0: {
                "C": {"open_interest": 150, "volume": 35},
                "P": {"open_interest": 60, "volume": 12},
            },
        }
    }

    op = read_options(99.1, book)

    assert op["exp"] == "2026-06-25"
    assert op["call_wall"] == 100.0
    assert op["put_wall"] == 99.0
    assert op["call_volume_wall"] == 100.0
    assert op["put_volume_wall"] == 99.0
    assert op["call_volume_total"] == 50.0
    assert op["put_volume_total"] == 34.0
    assert op["pcvr"] == 34.0 / 50.0
    assert op["atm_strike"] == 99.0
    assert op["atm_call_iv"] == 0.19
    assert op["atm_put_iv"] == 0.21
    assert round(op["atm_iv_skew"], 2) == 0.02
    assert op["atm_call_delta"] == 0.52
    assert op["atm_put_delta"] == -0.48
    assert op["atm_call_theta"] == -0.08
    assert op["atm_put_theta"] == -0.07
    assert op["atm_call_vega"] == 0.11
    assert op["atm_put_vega"] == 0.1
    assert op["atm_call_rho"] == 0.03
    assert op["atm_put_rho"] == -0.02
    assert op["atm_call_theo"] == 1.31
    assert op["atm_put_theo"] == 1.21
    assert op["atm_call_last_trade_price"] == 1.35
    assert op["atm_put_last_trade_price"] == 1.18
    assert round(op["atm_call_spread"], 2) == 0.2
    assert round(op["atm_put_spread"], 2) == 0.2
    assert round(op["atm_call_spread_pct"], 4) == 0.1538
    assert round(op["atm_put_spread_pct"], 4) == 0.1667
    assert op["atm_straddle_mid"] == 2.5
