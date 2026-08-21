from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from decision_receipts import build_decision_receipt
from setup_event_lifecycle import annotate_setup_conviction
from trade_permission import score_trade_permission

from tests.test_trade_permission import _bull_bars, _pa


def test_fresh_failed_break_setup_can_coexist_with_neutral_live_trap_corroboration():
    bars = _bull_bars()
    pa = _pa(
        bars,
        position_in_balance=0.1,
        balance_state="inside",
        balance_label="BOTTOM",
        rng_pos=35.0,
    )
    levels = {"ORH": 104.0, "ORL": 99.4, "PDC": 99.8}
    setup = {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": "reclaimed ORL",
        "level_name": "ORL",
        "level_price": 99.4,
        "trigger_price": 99.1,
        "bars_ago": 1,
    }

    card = score_trade_permission(bars, pa, levels, [setup], {"atm_iv": 0.20}, {}, {})

    assert card["setup_conviction"]["setup_tag"] == "FAILED BREAKDOWN"
    assert card["fresh_setup_evidence"]["status"] == "fresh_actionable_setup"
    assert card["fresh_setup_evidence"]["setup_tag"] == "FAILED BREAKDOWN"
    assert card["scores"]["trap_score"]["score"] == 35
    assert card["scores"]["trap_score"]["bias"] == "NEUTRAL"
    assert card["live_trap_corroboration"]["trap_score"] == 35
    assert card["live_trap_corroboration"]["trap_bias"] == "NEUTRAL"


def test_trap_corroboration_can_decay_while_lifecycle_persists_without_changing_authority():
    bars = _bull_bars()
    pa = _pa(bars, position_in_balance=0.15, balance_state="inside", rng_pos=32.0)
    levels = {"ORH": 104.0, "ORL": 99.4, "PDC": 99.8}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}
    failed_break_setup = {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": "reclaimed ORL",
        "level_name": "ORL",
        "level_price": 99.4,
        "trigger_price": 99.1,
        "bars_ago": 1,
    }
    sticky_context = {
        "tag": "STICKY DAY (calm/chop)",
        "bias": "FADE the edges - bet on snap-back to the magnet",
        "kind": "info",
        "detail": "positive gamma context only",
    }

    first_card = score_trade_permission(
        bars, pa, levels, [failed_break_setup], op, {}, {}
    )
    first_receipt = build_decision_receipt(
        "2026-07-10T10:12:00",
        "SPY",
        pa["spot"],
        first_card,
        {"label": "VWAP", "price": pa["vwap"]},
        [failed_break_setup],
    )
    second_receipt = build_decision_receipt(
        "2026-07-10T10:15:00",
        "SPY",
        pa["spot"],
        first_card,
        {"label": "VWAP", "price": pa["vwap"]},
        [failed_break_setup],
        previous_receipt=first_receipt,
    )

    context_only_card = score_trade_permission(
        bars,
        pa,
        levels,
        [sticky_context],
        op,
        {},
        {},
    )
    authority_before = (
        context_only_card["trade_permission_score"],
        context_only_card["execution_permission_score"],
        context_only_card["trade_gate"],
        context_only_card["market_day"]["bucket"],
    )
    sticky_receipt = build_decision_receipt(
        "2026-07-10T10:18:00",
        "SPY",
        pa["spot"],
        context_only_card,
        {"label": "Magnet", "price": pa["spot"] + 0.4},
        [sticky_context],
        previous_receipt=second_receipt,
    )

    annotate_setup_conviction(context_only_card, sticky_receipt["setup_events"])

    assert context_only_card["scores"]["trap_score"]["score"] == 35
    assert context_only_card["live_trap_corroboration"]["trap_score"] == 35
    assert context_only_card["fresh_setup_evidence"]["status"] == "fresh_context_setup"
    assert context_only_card["persisted_setup_thesis"]["active"] is True
    assert (
        context_only_card["persisted_setup_thesis"]["persisted_without_fresh_trigger"]
        is True
    )
    assert (
        context_only_card["persisted_setup_thesis"]["setup_tag"] == "FAILED BREAKDOWN"
    )
    assert (
        context_only_card["setup_conviction"]["event_lifecycle"][
            "persisted_without_fresh_trigger"
        ]
        is True
    )
    assert authority_before == (
        context_only_card["trade_permission_score"],
        context_only_card["execution_permission_score"],
        context_only_card["trade_gate"],
        context_only_card["market_day"]["bucket"],
    )


def test_iv_shock_can_override_failed_break_day_bucket_classification():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": 104.0, "ORL": 99.4, "PDC": 99.8}
    setup = {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": "reclaimed ORL",
        "level_name": "ORL",
        "level_price": 99.4,
        "trigger_price": 99.1,
        "bars_ago": 1,
    }

    card = score_trade_permission(
        bars,
        pa,
        levels,
        [setup],
        {"atm_iv": 0.35, "call_wall": 105.0, "put_wall": 99.0},
        {},
        {},
    )

    assert card["setup_conviction"]["setup_tag"] == "FAILED BREAKDOWN"
    assert card["market_day"]["bucket"] == "news_vol_shock_day"
    assert card["market_day"]["allowed_playbooks"] == []


def _market_data_provenance(option_session="2026-08-08"):
    return {
        "price_session_date": "2026-08-08",
        "options": {
            "latest_option_trade_time_raw": f"{option_session}T15:59:00",
        },
    }


def test_market_data_guard_blocks_stale_analysis_even_with_high_score():
    bars = _bull_bars()
    pa = _pa(bars)
    pa["price_authority"] = {
        "analysis_bar_stale": True,
        "analysis_bar_lag_minutes": 31.0,
    }
    card = score_trade_permission(
        bars,
        pa,
        {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5},
        [],
        {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0},
        {"regime": "negative", "gamma_data_quality": "ok", "dte": 0},
        {"premium_read": "cheap"},
        data_provenance=_market_data_provenance(),
    )

    assert card["trade_gate"] == "BLOCK"
    assert card["market_data_guard"]["status"] == "blocked"
    assert "analysis_bars_stale" in card["market_data_guard"]["blockers"]
    assert card["execution_flow"]["execution_permission"]["gate"] == "BLOCK"


def test_market_data_guard_blocks_mismatched_options_session():
    bars = _bull_bars()
    pa = _pa(bars)
    pa["price_authority"] = {"analysis_bar_stale": False}
    card = score_trade_permission(
        bars,
        pa,
        {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5},
        [],
        {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0},
        {"regime": "negative", "gamma_data_quality": "ok", "dte": 0},
        {"premium_read": "cheap"},
        data_provenance=_market_data_provenance("2026-08-07"),
    )

    assert card["trade_gate"] == "BLOCK"
    assert "options_session_mismatch" in card["market_data_guard"]["blockers"]


def test_market_data_guard_blocks_missing_gamma_quality():
    bars = _bull_bars()
    pa = _pa(bars)
    pa["price_authority"] = {"analysis_bar_stale": False}
    card = score_trade_permission(
        bars,
        pa,
        {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5},
        [],
        {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0},
        {},
        {"premium_read": "cheap"},
        data_provenance=_market_data_provenance(),
    )

    assert card["trade_gate"] == "BLOCK"
    assert "gamma_quality_missing" in card["market_data_guard"]["blockers"]


def test_market_data_guard_preserves_fresh_valid_scored_gate():
    bars = _bull_bars()
    pa = _pa(bars)
    pa["price_authority"] = {"analysis_bar_stale": False}
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}
    gp = {"regime": "negative", "gamma_data_quality": "ok", "dte": 0}
    baseline = score_trade_permission(
        bars, pa, levels, [], op, gp, {"premium_read": "cheap"}
    )
    guarded = score_trade_permission(
        bars,
        pa,
        levels,
        [],
        op,
        gp,
        {"premium_read": "cheap"},
        data_provenance=_market_data_provenance(),
    )

    assert guarded["market_data_guard"]["status"] == "eligible"
    assert guarded["market_data_guard"]["blockers"] == []
    assert guarded["trade_gate"] == baseline["trade_gate"]
    assert guarded["trade_permission_score"] == baseline["trade_permission_score"]
