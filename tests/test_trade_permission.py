from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from decision_receipts import build_decision_receipt  # noqa: E402
from execution_vector_weights import (  # noqa: E402
    DEFAULT_BASE_BIAS_WEIGHTS,
    DEFAULT_BASE_WEIGHTS,
)
from setup_event_lifecycle import annotate_setup_conviction  # noqa: E402
from trade_permission import ExecutionVectorEngine, score_trade_permission  # noqa: E402


def _bull_bars():
    bars = []
    price = 100.0
    for minute in range(45):
        open_ = price
        close = price + 0.08
        high = close + 0.04
        low = open_ - 0.02
        volume = 1000 + minute * 12
        bars.append((minute, open_, high, low, close, volume))
        price = close
    return bars


def _pa(bars, **overrides):
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "balance_high": closes[-2],
        "balance_low": closes[-8],
        "position_in_balance": 1.0,
        "balance_state": "above",
        "balance_label": "TOP",
        "balance_width_pct": 0.42,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "recent_balance",
        "dominant_balance_reason": "mid-session: the active recent box matters most",
        "session_position_in_range": 0.92,
        "rng_pos": 92.0,
        "day_chg": 1.2,
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
    }
    pa.update(overrides)
    return pa


def test_get_minutes_since_open_uses_datetime():
    engine = ExecutionVectorEngine()
    assert engine._get_minutes_since_open(datetime(2026, 1, 1, 10, 45)) == 75.0


def test_execution_vector_engine_matches_public_wrapper_core_legacy_contract():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}
    engine = ExecutionVectorEngine()

    raw_legacy = engine.build_card(
        bars, pa, levels, setups=[], op=op, gp={}, magnitude={"premium_read": "cheap"}
    )
    wrapped = score_trade_permission(
        bars, pa, levels, [], op, {}, {"premium_read": "cheap"}
    )

    assert raw_legacy["authority_engine"] == "legacy"
    assert wrapped["authority_engine"] == "legacy"
    assert raw_legacy["trade_permission_score"] == wrapped["trade_permission_score"]
    assert raw_legacy["bucket_conditioned_spine"] == wrapped["bucket_conditioned_spine"]
    assert any(
        voice.get("voice_id") == "ace_advisory"
        for voice in wrapped["authority_adjudication"]["competing_voices"]
    )


def test_regime_weight_is_dynamically_reduced_from_base_when_trend_agrees():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    engine = ExecutionVectorEngine()
    parts = engine.build_parts(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    score_weights = engine._score_weight_map(parts)
    bias_weights = engine._bias_weight_map(parts)

    assert score_weights["regime_score"] < engine.base_weights["regime_score"]
    assert bias_weights["regime_score"] < engine.base_bias_weights["regime_score"]


def test_pressure_weight_is_dynamically_reduced_when_trend_already_agrees():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    engine = ExecutionVectorEngine()
    parts = engine.build_parts(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    score_weights = engine._score_weight_map(parts)
    bias_weights = engine._bias_weight_map(parts)

    assert parts["trend_score"].bias == parts["pressure_score"].bias
    assert score_weights["pressure_score"] < engine.base_weights["pressure_score"]
    assert bias_weights["pressure_score"] < engine.base_bias_weights["pressure_score"]


def test_trade_permission_exposes_execution_hierarchy_without_changing_gate():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    hierarchy = card["execution_hierarchy"]
    core_names = [row["name"] for row in hierarchy["core_spine"]["features"]]
    drift_names = [row["name"] for row in hierarchy["suspect_drift_voices"]]
    advisory_names = [row["name"] for row in hierarchy["advisory_surfaces"]]

    assert card["trade_gate"] in {"PERMIT", "CAUTION"}
    assert hierarchy["schema"] == "sharpedge.execution_hierarchy.v1"
    assert core_names == [
        "structure_score",
        "acceptance_score",
        "trend_score",
        "location_score",
        "volume_score",
        "time_of_day_score",
        "dealer_gamma_score",
    ]
    assert hierarchy["core_spine"]["features"][1]["label"] == "Auction Acceptance"
    assert hierarchy["core_spine"]["features"][4]["label"] == "Participation"
    assert drift_names == ["pressure_score", "regime_score"]
    assert advisory_names == ["expansion_fuel_score"]
    assert hierarchy["advisory_surfaces"][0]["label"] == "Expansion Fuel"
    assert hierarchy["core_spine"]["normalized_weighted_score"] > 0


def test_trade_permission_permits_clean_bullish_acceptance():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}

    card = score_trade_permission(
        bars, pa, levels, [], op, {}, {"premium_read": "cheap"}
    )

    assert card["trade_gate"] in {"PERMIT", "CAUTION"}
    assert card["trade_permission_score"] >= 58
    assert card["execution_permission_score"] == card["trade_permission_score"]
    assert card["setup_conviction"]["setup_gate"] == "NONE"
    assert card["bias"] in {"CALLS", "NEUTRAL"}
    assert card["scores"]["acceptance_score"]["score"] >= 60
    assert card["structure_state"]["schema"] == "sharpedge.structure_state.v1"
    assert card["structure_state"]["state"] == "insufficient_sequence"
    assert card["acceptance_state"]["schema"] == "sharpedge.acceptance_state.v1"
    assert card["acceptance_state"]["state"] == "accepted_above_level"
    assert card["acceptance_state"]["accepted_level_count"] >= 1
    assert card["location_state"]["schema"] == "sharpedge.location_state.v1"
    assert card["location_state"]["state"] in {
        "near_reference",
        "between_references",
        "at_reference",
        "above_all_references",
    }
    assert card["dealer_state"]["schema"] == "sharpedge.dealer_state.v1"
    assert card["dealer_state"]["state"] in {
        "positive_gamma_context",
        "positive_gamma_gravity",
        "negative_gamma_expansion",
        "dealer_unknown",
    }
    assert card["volume_state"]["schema"] == "sharpedge.volume_profile.v1"
    assert card["volume_state"]["confirmation"] in {
        "confirmed",
        "participating",
        "mixed",
        "missing",
    }
    assert card["trend_state"]["schema"] == "sharpedge.trend_state.v1"
    assert card["trend_state"]["state"] in {
        "aligned_up",
        "aligned_down",
        "conflict",
        "neutral",
        "insufficient",
        "unknown",
    }
    assert card["time_state"]["schema"] == "sharpedge.time_state.v1"
    assert card["time_state"]["state"] in {
        "opening",
        "morning",
        "midday",
        "afternoon",
        "power_hour",
        "closed_or_unknown",
    }
    for key in (
        "exhaustion_score",
        "trap_score",
        "dealer_gamma_score",
        "regime_score",
        "expansion_fuel_score",
    ):
        assert key in card["scores"]


def test_candle_score_is_removed_from_permission_contract():
    bars = _bull_bars()
    levels = {"ORH": bars[-5][4] - 0.2, "ORL": 99.8, "PDC": 99.5}
    pa = _pa(bars, vs_vwap=0.18, mom15=0.35, vol_mult=1.3, rng_pos=84.0)
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}

    card = score_trade_permission(
        bars, pa, levels, [], op, {}, {"premium_read": "cheap"}
    )

    assert "candle_score" not in card["scores"]
    assert "candle_score" not in DEFAULT_BASE_WEIGHTS
    assert "candle_score" not in DEFAULT_BASE_BIAS_WEIGHTS


def test_bullish_acceptance_with_weak_governors_stays_out_of_permit():
    base_bars = _bull_bars()
    last_minute, _open, _high, _low, prior_close, _volume = base_bars[-1]
    ugly_bars = [
        (*bar[:5], 450 if idx >= len(base_bars) - 5 else bar[5])
        for idx, bar in enumerate(base_bars[:-1])
    ] + [
        (
            last_minute,
            prior_close + 0.24,
            prior_close + 0.26,
            prior_close - 0.02,
            prior_close,
            450,
        )
    ]
    spot = ugly_bars[-1][4]
    pa = _pa(
        ugly_bars,
        spot=spot,
        vs_vwap=0.18,
        mom15=0.28,
        vol_mult=0.5,
        rng_pos=84.0,
        balance_confluence={
            "score": 68,
            "bias": "CALLS",
            "reason": "opening balance supports continuation",
        },
        balance_disagreement={
            "has_disagreement": True,
            "reason": "bulls=opening_balance bears=recent_balance",
        },
    )
    levels = {"ORH": base_bars[-5][4] - 0.2, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.26, "call_wall": 105.0, "put_wall": 99.0}
    gp = {"regime": "positive", "pin": spot}

    card = score_trade_permission(
        ugly_bars, pa, levels, [], op, gp, {"premium_read": "rich"}
    )

    assert card["bias"] in {"CALLS", "NEUTRAL"}
    assert card["scores"]["acceptance_score"]["score"] >= 78
    assert card["scores"]["volume_score"]["score"] == 25
    assert card["scores"]["dealer_gamma_score"]["score"] <= 38
    assert card["scores"]["balance_context_score"]["score"] <= 30
    assert card["trade_gate"] in {"BLOCK", "CAUTION"}
    assert card["trade_permission_score"] < 72


def test_trade_permission_penalizes_midday_thin_chop():
    flat_bars = [
        (150 + idx, 100.0, 100.08, 99.94, 100.01 if idx % 2 else 99.99, 500)
        for idx in range(40)
    ]
    pa = _pa(
        flat_bars,
        spot=100.0,
        position_in_balance=0.5,
        balance_state="inside",
        balance_label="MIDDLE",
        rng_pos=50.0,
        vs_vwap=0.01,
        mom15=0.0,
        vol_mult=0.55,
    )
    levels = {"ORH": 101.4, "ORL": 98.6, "PDC": 101.0}

    card = score_trade_permission(flat_bars, pa, levels, [], {"atm_iv": 0.10}, {}, {})

    assert card["trade_gate"] in {"BLOCK", "CAUTION"}
    assert card["trade_permission_score"] < 72
    assert card["scores"]["volume_score"]["score"] <= 42


def test_failed_breakdown_is_kept_separate_from_execution_permission():
    bars = _bull_bars()
    pa = _pa(
        bars,
        position_in_balance=0.1,
        balance_state="inside",
        balance_label="BOTTOM",
        rng_pos=35.0,
    )
    setup = {"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)", "kind": "ok"}
    levels = {"ORH": 104.0, "ORL": 100.4, "PDC": 99.8}

    card = score_trade_permission(bars, pa, levels, [setup], {"atm_iv": 0.20}, {}, {})

    assert card["setup_conviction"]["setup_gate"] == "ACTIONABLE"
    assert card["setup_conviction"]["setup_tag"] == "FAILED BREAKDOWN"
    assert card["setup_conviction"]["bias"] == "CALLS"
    assert card["scores"]["pressure_score"]["score"] < 80
    assert card["scores"]["rejection_score"]["score"] < 80
    assert card["scores"]["trap_score"]["score"] < 90


def test_exhaustion_runner_handoff_is_treated_as_actionable_setup():
    bars = _bull_bars()
    pa = _pa(
        bars,
        spot=103.6,
        position_in_balance=0.7,
        balance_state="above",
        balance_label="TOP",
        rng_pos=76.0,
        vs_vwap=0.22,
        mom15=0.34,
        vol_mult=1.8,
    )
    levels = {"ORH": 103.0, "ORL": 100.0, "PDC": 100.0}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 100.0}
    gp = {"regime": "negative", "pin": 103.0}
    setups = [
        {
            "tag": "RUNNER DAY (wheee)",
            "bias": "RIDE momentum - go directional, breakouts run",
        },
        {
            "tag": "EXHAUSTION -> RUNNER HANDOFF",
            "bias": "CALLS (reversal promoted to runner)",
            "kind": "ok",
        },
    ]

    card = score_trade_permission(bars, pa, levels, setups, op, gp, {})

    assert card["setup_conviction"]["setup_gate"] == "ACTIONABLE"
    assert card["setup_conviction"]["setup_tag"] == "EXHAUSTION -> RUNNER HANDOFF"
    assert card["setup_conviction"]["bias"] == "CALLS"


def test_positive_gamma_pin_dampens_trade_permission():
    bars = _bull_bars()
    pa = _pa(
        bars,
        spot=103.6,
        position_in_balance=0.55,
        balance_state="inside",
        balance_label="MIDDLE",
        rng_pos=55.0,
        vs_vwap=0.02,
        mom15=0.01,
    )
    levels = {"ORH": 104.0, "ORL": 100.0, "PDC": 100.0}
    op = {"atm_iv": 0.18, "call_wall": 104.0, "put_wall": 100.0}
    gp = {"regime": "positive", "pin": 103.6}

    card = score_trade_permission(bars, pa, levels, [], op, gp, {})

    assert card["scores"]["dealer_gamma_score"]["score"] <= 40
    assert "pinning" in card["scores"]["dealer_gamma_score"]["reason"]


def test_failed_breakout_trap_scores_bearish():
    bars = _bull_bars()
    level = bars[-2][4] - 0.03
    trap_bars = bars + [
        (45, level, level + 0.25, level - 0.02, level - 0.04, 2500),
    ]
    pa = _pa(
        trap_bars,
        spot=level - 0.04,
        position_in_balance=0.92,
        balance_state="inside",
        balance_label="TOP",
        rng_pos=82.0,
        mom15=-0.12,
    )
    levels = {"ORH": level, "ORL": 100.0, "PDC": 100.0}

    card = score_trade_permission(trap_bars, pa, levels, [], {"atm_iv": 0.22}, {}, {})

    assert card["scores"]["trap_score"]["bias"] == "PUTS"
    assert card["scores"]["trap_score"]["score"] >= 78


def test_location_uses_pure_reference_state_not_balance_interpretation():
    bars = _bull_bars()
    pa = _pa(
        bars,
        spot=100.0,
        position_in_balance=0.94,
        balance_state="inside",
        balance_label="TOP",
        vs_vwap=0.03,
        mom15=0.02,
        vol_mult=0.9,
    )
    levels = {"ORH": 110.0, "ORL": 99.0, "PDC": 100.0}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    assert card["location_state"]["state"] == "at_reference"
    assert card["scores"]["location_score"]["bias"] == "NEUTRAL"
    assert (
        "at decision reference PDC 100.00" in card["scores"]["location_score"]["reason"]
    )


def test_location_middle_of_day_chop_no_longer_lives_in_location_vertical():
    bars = [
        (330 + idx, 100.0, 100.08, 99.94, 100.01 if idx % 2 else 99.99, 500)
        for idx in range(40)
    ]
    pa = _pa(
        bars,
        spot=100.0,
        position_in_balance=0.5,
        balance_state="inside",
        balance_label="MIDDLE",
        balance_reference="value_60_bar",
        dominant_balance_name="value_balance",
        rng_pos=50.0,
        vs_vwap=0.01,
        mom15=0.0,
        vol_mult=0.55,
    )
    levels = {"ORH": 101.4, "ORL": 98.6, "PDC": 101.0}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.10}, {}, {})

    assert card["location_state"]["state"] == "between_references"
    assert card["scores"]["location_score"]["score"] == 42
    assert (
        "between VWAP 99.66 and PDC 101.00"
        in card["scores"]["location_score"]["reason"]
    )


def test_pressure_needs_one_sided_closes_not_just_bullish_tape_context():
    bars = [
        (0, 100.00, 100.18, 99.98, 100.10, 1000),
        (1, 100.10, 100.22, 100.02, 100.18, 1010),
        (2, 100.18, 100.24, 100.05, 100.12, 1020),
        (3, 100.12, 100.28, 100.08, 100.22, 1030),
        (4, 100.22, 100.26, 100.10, 100.16, 1040),
        (5, 100.16, 100.34, 100.14, 100.28, 1050),
        (6, 100.28, 100.32, 100.18, 100.20, 1060),
        (7, 100.20, 100.40, 100.18, 100.34, 1070),
    ]
    pa = _pa(bars, vs_vwap=0.25, mom15=0.6, vol_mult=1.7)
    levels = {"ORH": 100.0, "ORL": 99.5, "PDC": 99.8}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    assert card["trend_state"]["state"] == "aligned_up"
    assert card["scores"]["trend_score"]["score"] >= 80
    assert card["scores"]["pressure_score"]["bias"] == "NEUTRAL"
    assert card["scores"]["pressure_score"]["score"] <= 40
    assert "mixed" in card["scores"]["pressure_score"]["reason"]


def test_opening_auction_context_decays_later_in_day():
    morning = _bull_bars()
    late = [
        (330 + minute, open_, high, low, close, volume)
        for minute, open_, high, low, close, volume in morning
    ]
    levels = {"ORH": 100.4, "ORL": 99.8, "PDC": 99.6}
    pa_morning = _pa(morning)
    pa_late = _pa(late)

    morning_card = score_trade_permission(
        morning, pa_morning, levels, [], {"atm_iv": 0.18}, {}, {}
    )
    late_card = score_trade_permission(
        late, pa_late, levels, [], {"atm_iv": 0.18}, {}, {}
    )

    morning_delta = abs(morning_card["scores"]["opening_auction_score"]["score"] - 50)
    late_delta = abs(late_card["scores"]["opening_auction_score"]["score"] - 50)

    assert late_delta < morning_delta
    assert "late session" in late_card["scores"]["opening_auction_score"]["reason"]


def test_post_selloff_coil_changes_permission_score_and_bias():
    bars = [
        (150 + idx, 100.0, 100.07, 99.93, 99.98 if idx % 2 else 100.0, 650)
        for idx in range(40)
    ]
    pa = _pa(
        bars,
        spot=99.12,
        position_in_balance=0.14,
        balance_state="inside",
        balance_label="BOTTOM",
        rng_pos=22.0,
        vs_vwap=-0.18,
        mom15=-0.03,
        vol_mult=0.72,
    )
    levels = {"ORH": 100.2, "ORL": 99.0, "PDC": 100.0}
    base = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})
    coil = score_trade_permission(
        bars,
        pa,
        levels,
        [{"tag": "POST-SELLOFF COIL", "bias": "NEUTRAL-to-BEARISH", "kind": "warn"}],
        {"atm_iv": 0.18},
        {},
        {},
        {
            "volatility_state": "squeeze",
            "structure_state": "channel_breakout_setup",
            "bias": "neutral_to_bearish",
            "coil": True,
            "trigger_high": 99.18,
            "trigger_low": 99.10,
        },
    )

    assert base["scores"]["compression_score"]["bias"] == "NEUTRAL"
    assert coil["scores"]["compression_score"]["bias"] == "PUTS"
    assert coil["scores"]["compression_score"]["score"] >= 70
    assert coil["trade_permission_score"] == base["trade_permission_score"]
    assert coil["bias"] == "NEUTRAL"


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
