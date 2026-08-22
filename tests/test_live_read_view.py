from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_read_view import (
    _active_setup_level_badge,
    infer_target,
    reachability_context,
    render_confluence_zones_block,
    render_execution_state_packets_block,
    render_live_read_html,
    render_location_strip,
    render_permission_section,
    render_regime_read_block,
    render_structure_state_block,
    summarize_permission_scores,
)


def _permission(bias: str = "CALLS") -> dict:
    return {
        "trade_gate": "CAUTION",
        "trade_permission_score": 66,
        "execution_permission_score": 66,
        "bias": bias,
        "setup_conviction": {
            "setup_conviction_score": 84,
            "setup_gate": "ACTIONABLE",
            "bias": "CALLS",
            "setup_tag": "FAILED BREAKDOWN",
            "reason": "reclaimed ORL 1m ago after stabbing lower support",
            "entry_gate": {"gate_id": "failed_breakdown_reclaim"},
            "context_gate": {"gate_id": "sticky_day_magnet_fade"},
            "event_lifecycle": {
                "status": "confirmed",
                "confidence": 70,
                "first_seen_ts": "2026-06-25T10:12:00",
                "last_seen_ts": "2026-06-25T10:15:00",
                "last_confirmed_ts": "2026-06-25T10:15:00",
                "observation_count": 2,
                "level_name": "ORL",
                "level_price": 99.75,
            },
        },
        "structure_state": {
            "state": "bullish_sequence",
            "bias": "CALLS",
            "reason": "HH/HL structure intact",
            "sequence_quality": "confirmed",
            "spacing_ok": True,
            "amplitude_ok": True,
            "freshness_ok": True,
            "has_sequence": True,
            "swing_high_count": 2,
            "swing_low_count": 2,
        },
        "acceptance_state": {
            "state": "accepted_above_level",
            "bias": "CALLS",
            "accepted_level_count": 2,
            "representative_level": {"level_name": "ORH", "level_price": 100.2},
            "reason": "3 closes accepted above ORH 100.20",
        },
        "location_state": {
            "state": "near_reference",
            "bias": "NEUTRAL",
            "reference_count": 3,
            "nearest_reference": {"reference_name": "VWAP", "reference_price": 100.5},
            "reason": "near VWAP 100.50 (0.10% away)",
        },
        "dealer_state": {
            "state": "negative_gamma_expansion",
            "bias": "NEUTRAL",
            "gamma_state": {"state": "gamma_expansion"},
            "pin_state": {"state": "far_pin"},
            "wall_state": {"state": "no_near_wall"},
            "reason": "negative gamma/OI proxy may support expansion; accepted breaks can run",
        },
        "volume_state": {
            "schema": "sharpedge.volume_profile.v1",
            "confirmation": "confirmed",
            "move_direction": "up",
            "local_mult": 1.4,
            "session_mult": 1.2,
            "aligned_volume_share": 0.66,
            "reason": "confirmed: local 1.4x, session 1.2x, aligned 66%, efficiency 62%",
        },
        "trend_state": {
            "state": "aligned_up",
            "bias": "CALLS",
            "component_states": {"slope": "up", "vwap": "up", "momentum": "up"},
            "detail": "trend components aligned up",
        },
        "time_state": {
            "state": "morning",
            "clock": "10:15",
            "minutes_since_open": 45,
            "detail": "morning continuation window",
        },
        "scores": {
            "volume_score": {"score": 85, "reason": "participation confirms move"},
            "structure_score": {"score": 82, "reason": "HH/HL structure intact"},
            "acceptance_score": {"score": 78, "reason": "accepted above ORH"},
            "expansion_fuel_score": {
                "score": 92,
                "bias": "CALLS",
                "reason": "expansion fuel is active",
            },
            "balance_context_score": {"score": 28, "reason": "balance disagreement"},
            "trap_score": {"score": 35, "reason": "no failed-break trap detected"},
            "pressure_score": {"score": 35, "reason": "no clear trapped side"},
        },
        "execution_expansion_potential": {
            "surface": {
                "score": 80,
                "bias": "CALLS",
                "reason": "expansion fuel is active: gamma/OI proxy implies hedging feedback may keep price moving",
            },
            "summary": {
                "state": "low_confirmation_high_fuel",
                "participation_confirmation": "low",
                "expansion_fuel": "high",
                "dominant_mechanism": "dealer_gamma_feedback",
                "note": "Participation is not confirming much, but other mechanisms can still let price travel.",
            },
            "mechanisms": [
                {
                    "mechanism_id": "dealer_gamma_feedback",
                    "label": "Gamma proxy may amplify the move",
                    "family": "fuel",
                    "strength": "high",
                    "reason": "negative gamma/OI proxy may support hedging feedback and expansion",
                }
            ],
        },
        "execution_vector_interactions": {
            "summary": {
                "interaction_balance": "mixed",
                "favorable_count": 2,
                "warning_count": 1,
                "strong_favorable_count": 1,
                "strong_warning_count": 1,
            },
            "best": [
                {
                    "interaction_id": "trend_acceptance_alignment",
                    "classification": "strongly_good",
                    "label": "Trend + acceptance aligned",
                    "reason": "Directional drive and acceptance are working together.",
                }
            ],
            "warnings": [
                {
                    "interaction_id": "trend_volume_conflict",
                    "classification": "strongly_bad",
                    "label": "Trend without participation",
                    "reason": "Tape is moving but volume is not backing it.",
                }
            ],
        },
    }


def _trend() -> dict:
    return {
        "points": [
            {
                "time": "08:30",
                "score": 54,
                "event_markers": ["RUNNER DAY (wheee) CANDIDATE"],
            },
            {
                "time": "09:45",
                "score": 71,
                "event_markers": ["FAILED BREAKDOWN CANDIDATE @ ORL"],
            },
            {
                "time": "11:15",
                "score": 66,
                "event_markers": ["FAILED BREAKDOWN CONFIRMED @ ORL"],
            },
        ],
        "direction": "weakening",
        "delta": -5,
        "largest_changes_since_last_update": [
            {"feature": "Participation", "delta": 12},
            {"feature": "Acceptance", "delta": 8},
            {"feature": "Trend", "delta": -10},
            {"feature": "Balance Context", "delta": -7},
        ],
        "setup_transitions_since_last_update": [
            {
                "label": "FAILED BREAKDOWN CONFIRMED @ ORL",
                "ts": "2026-06-25T11:15:00",
            }
        ],
    }


def _edge_token_position() -> dict:
    return {
        "suggested_action": "hold",
        "position_state": "open",
        "contracts_held": 1,
        "action_reason": "edge token is still active; keep the single-contract position on.",
        "policy": {"contracts_per_token": 1},
        "current_token": {
            "event_type": "FAILED BREAKDOWN",
            "side": "CALLS",
            "status": "confirmed",
            "observation_count": 2,
            "level_name": "ORL",
            "level_price": 99.75,
        },
    }


def _weekly_context() -> dict:
    return {
        "lookback_days": 5,
        "panel_note": "Middle chart = 5-day carry map. Bottom chart = monthly structure map.",
        "headline": "Holding the upper carry shelf beneath H1",
        "detail": "Spot $100.00 is between LH1 $99.80 and H1 $100.40. Nearest carry pivot: LH1 lower high $99.80 (0.20% away). 5-day range $98.40 -> $100.60 (73% up the range).",
        "kind": "ok",
        "legend": [
            {"label": "H1 peak", "price": 100.4, "color": "#f85149"},
            {"label": "LH1 lower high", "price": 99.8, "color": "#d29922"},
            {"label": "HL1 higher low", "price": 99.1, "color": "#39c5cf"},
            {"label": "L1 washout low", "price": 98.7, "color": "#26a641"},
        ],
    }


def _monthly_context() -> dict:
    return {
        "lookback_months": 6,
        "panel_note": "Bottom chart = 6-month structure map built from prior month rails + current month open.",
        "headline": "Holding above monthly value inside the upper month band",
        "detail": "Spot $100.00 is above MOPEN $99.20 and PMC $99.70. Nearest monthly rail: prior month close $99.70 (0.30% away). 6-month range $94.40 -> $101.20 (82% up the range).",
        "kind": "ok",
        "legend": [
            {"label": "Prior month high", "price": 100.8, "color": "#f85149"},
            {"label": "Month open", "price": 99.2, "color": "#58a6ff"},
            {"label": "Prior month close", "price": 99.7, "color": "#bc8cff"},
            {"label": "Prior month low", "price": 97.9, "color": "#26a641"},
        ],
    }


def test_summarize_permission_scores_surfaces_top_and_bottom_three():
    summary = summarize_permission_scores(_permission())

    assert [item["label"] for item in summary["best"]] == [
        "Participation",
        "Structure",
        "Auction Acceptance",
    ]
    assert [item["score"] for item in summary["worst"]] == [28, 35, 35]


def test_infer_target_uses_actionable_gate_even_when_context_card_is_first():
    pa = {"spot": 100.0, "vwap": 100.6}
    op = {"call_wall": 101.5, "put_wall": 98.5}
    gp = {"pin": 100.7, "regime": "positive"}
    micro = {"ch_lo": 99.1, "ch_hi": 101.1}
    magnitude = {"exp_move_realized_usd": 0.9}

    target = infer_target(
        pa,
        op,
        _permission("CALLS"),
        gp,
        micro,
        magnitude,
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            },
            {"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)"},
        ],
    )

    assert target["setup_tag"] == "FAILED BREAKDOWN"
    assert target["label"] == "VWAP"
    assert target["price"] == 100.6


def test_infer_target_uses_handoff_continuation_targets():
    pa = {"spot": 100.0, "vwap": 99.6}
    op = {"call_wall": 101.8, "put_wall": 98.5}
    gp = {"pin": 99.7, "regime": "negative"}
    micro = {"ch_lo": 99.2, "ch_hi": 100.7}
    magnitude = {"exp_move_realized_usd": 1.1}

    target = infer_target(
        pa,
        op,
        _permission("CALLS"),
        gp,
        micro,
        magnitude,
        [
            {
                "tag": "RUNNER DAY (wheee)",
                "bias": "RIDE momentum - go directional, breakouts run",
            },
            {
                "tag": "EXHAUSTION -> RUNNER HANDOFF",
                "bias": "CALLS (reversal promoted to runner)",
            },
        ],
    )

    assert target["setup_tag"] == "EXHAUSTION -> RUNNER HANDOFF"
    assert target["objective"] == "handoff_continuation"
    assert target["label"] == "Channel hi"
    assert target["price"] == 100.7


def test_infer_target_uses_setup_aware_priority_before_directional_fallback():
    pa = {"spot": 100.0, "vwap": 100.6}
    op = {"call_wall": 101.5, "put_wall": 98.5}
    gp = {"pin": 100.7, "regime": "negative"}
    micro = {"ch_lo": 99.1, "ch_hi": 101.1}
    magnitude = {"exp_move_realized_usd": 0.9}

    failed_breakdown = infer_target(
        pa,
        op,
        _permission("CALLS"),
        gp,
        micro,
        magnitude,
        [{"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)"}],
    )
    runner = infer_target(
        pa,
        op,
        _permission("PUTS"),
        gp,
        micro,
        magnitude,
        [{"tag": "POST-SELLOFF COIL", "bias": "NEUTRAL-to-BEARISH"}],
    )
    sticky = infer_target(
        pa,
        op,
        _permission("NEUTRAL"),
        gp,
        micro,
        magnitude,
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            }
        ],
    )

    assert failed_breakdown["label"] == "VWAP"
    assert failed_breakdown["price"] == 100.6
    assert runner["label"] == "Channel lo"
    assert runner["price"] == 99.1
    assert sticky["label"] == "Magnet"
    assert sticky["price"] == 100.7
    assert sticky["reachable_today"]["label"] == "VWAP"


def test_reachability_context_flags_target_within_expected_move():
    ctx = reachability_context(
        pa={"spot": 100.0, "vwap": 100.6},
        op={"call_wall": 101.2, "put_wall": 99.0},
        permission=_permission("CALLS"),
        magnitude={"exp_move_realized_usd": 1.5},
        gp={"pin": 100.4, "regime": "negative"},
        micro={"ch_lo": 99.4, "ch_hi": 101.0},
        setups=[{"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)"}],
    )

    assert ctx["status"] == "within"
    assert ctx["distance"] == 0.6
    assert ctx["coverage_ratio"] == 0.4


def test_render_location_strip_shows_unique_visual_markers():
    html = render_location_strip(
        pa={"spot": 100.0},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3},
    )

    assert "LOCATION STRIP" in html
    assert "Magnet" in html
    assert "Exp low" in html
    assert "Exp high" in html
    assert "Channel hi" in html
    assert "Channel lo" in html
    assert "Put wall" in html
    assert "Call wall" in html


def test_render_location_strip_drops_far_away_wall_outliers():
    html = render_location_strip(
        pa={"spot": 728.99},
        op={"put_wall": 565.0, "call_wall": 750.0},
        micro={"ch_lo": 716.58, "ch_hi": 733.60},
        magnitude={"exp_move_realized_usd": 1.68},
        gp={"pin": 732.0},
    )

    assert "LOCATION STRIP" in html
    assert "Call wall" in html
    assert "Put wall $565.00" not in html
    assert "Channel lo $716.58" in html
    assert "Magnet $732.00" in html


def test_active_setup_level_badge_shows_failed_break_level_and_trigger():
    html = _active_setup_level_badge(
        [
            {
                "tag": "STICKY DAY (calm/chop)",
                "bias": "FADE the edges - bet on snap-back to the magnet",
            },
            {
                "tag": "FAILED BREAKDOWN",
                "level_name": "ORL",
                "level_price": 99.75,
                "trigger_price": 99.4,
                "bars_ago": 2,
            },
        ]
    )

    assert "ACTIVE SETUP LEVEL" in html
    assert "ORL" in html
    assert "$99.75" in html
    assert "trigger $99.40" in html
    assert "2m ago" in html


def test_render_permission_section_shows_expansion_potential_block():
    html = render_permission_section(
        permission=_permission(),
        pa={"spot": 100.0, "vwap": 100.5},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[{"tag": "POST-SELLOFF COIL", "bias": "NEUTRAL-to-BEARISH"}],
        permission_trend=_trend(),
    )

    assert "EXPANSION POTENTIAL" in html
    assert "surface:" in html
    assert "participation:" in html
    assert "fuel:" in html
    assert "Gamma proxy may amplify the move" in html
    assert "Participation is not confirming much" in html


def test_render_permission_section_shows_vector_interactions_block():
    html = render_permission_section(
        permission=_permission(),
        pa={"spot": 100.0, "vwap": 100.5},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[{"tag": "POST-SELLOFF COIL", "bias": "NEUTRAL-to-BEARISH"}],
        permission_trend=_trend(),
    )

    assert "EXPANSION POTENTIAL" in html
    assert "VECTOR INTERACTIONS" in html
    assert "Good combos" in html
    assert "Bad combos" in html
    assert "Trend + acceptance aligned" in html
    assert "Trend without participation" in html
    assert "STRONGLY GOOD" in html
    assert "STRONGLY BAD" in html


def test_render_permission_section_includes_reason_summary_and_reachability():
    html = render_permission_section(
        permission=_permission(),
        pa={"spot": 100.0, "vwap": 100.5},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[
            {
                "tag": "POST-SELLOFF COIL",
                "bias": "NEUTRAL-to-BEARISH",
                "kind": "warn",
                "detail": "tight channel",
            }
        ],
        permission_trend=_trend(),
    )

    assert "SETUP CONVICTION" in html
    assert "EXECUTION READ" in html
    assert "approval_decision is final authority" in html
    assert "status: CONFIRMED" in html
    assert "first seen 10:12" in html
    assert "TOP REASONS TO TRADE" not in html
    assert "TOP REASONS TO WAIT" not in html
    assert "REMAINING EXPECTED MOVE VS DISTANCE TO TARGET" in html
    assert "Strategic target:" in html
    assert "Reachable today:" in html
    assert "PERMISSION SCORE TREND" not in html
    assert "Largest changes since last update" not in html
    assert "LOCATION STRIP" in html


def test_render_structure_state_block_renders_minimal_live_audit_card():
    html = render_structure_state_block(_permission()["structure_state"])

    assert "STRUCTURE STATE" in html
    assert "BULLISH SEQUENCE / CALLS / quality CONFIRMED" in html
    assert "swing highs: 2" in html
    assert "spacing ok: True" in html
    assert "HH/HL structure intact" in html


def test_render_execution_state_packets_block_shows_live_logic_brain_states():
    html = render_execution_state_packets_block(_permission())

    assert "EXECUTION STATE PACKETS" in html
    assert "ACCEPTANCE STATE" in html
    assert "LOCATION STATE" in html
    assert "DEALER STATE" in html
    assert "VOLUME STATE" in html
    assert "TREND STATE" in html
    assert "TIME STATE" in html
    assert "trend components aligned up" in html
    assert "negative gamma/OI proxy may support expansion" in html


def test_render_html_embeds_new_live_read_sections():
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": 0.4, "vwap": 100.5},
        op={"put_wall": 98.5, "call_wall": 101.5},
        lines=[("Volume SURGE", "ok", "move is confirmed")],
        setups=[
            {
                "tag": "POST-SELLOFF COIL",
                "bias": "NEUTRAL-to-BEARISH",
                "kind": "warn",
                "detail": "tight channel",
            }
        ],
        permission=_permission(),
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        permission_trend=_trend(),
        edge_token_position=_edge_token_position(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="11:15:00",
    )

    assert "SharpEdge Live Read - SPY" in html
    assert "<main " in html
    assert "</main>" in html
    assert "overflow-x:auto" in html
    assert "overflow-x:hidden" not in html
    assert 'http-equiv="refresh" content="10"' in html
    assert "auto 10s" in html
    assert "SETUP CONVICTION" in html
    assert "EXECUTION READ" in html
    assert "TOP REASONS TO TRADE" not in html
    assert "TOP REASONS TO WAIT" not in html
    assert "REMAINING EXPECTED MOVE VS DISTANCE TO TARGET" in html
    assert "Strategic target:" in html
    assert "Reachable today:" in html
    assert "PERMISSION SCORE TREND" in html
    assert "EDGE TOKEN ENGINE" not in html
    assert "Execution state packet details (debug)" in html
    assert "EXECUTION STATE PACKETS" in html
    assert "ACCEPTANCE STATE" in html
    assert "DEALER STATE" in html
    assert "TODAY'S LIVE BATTLEFIELD + EXECUTION SPINE" in html
    assert "08:30" in html
    assert "FAILED BREAKDOWN CONFIRMED @ ORL" in html
    assert "Setup lifecycle since last update" in html
    assert "LOCATION STRIP" in html
    assert "EXPANSION POTENTIAL" in html
    assert "VECTOR INTERACTIONS" in html
    assert "WEEKLY CONTEXT" in html
    assert "MONTHLY CONTEXT" in html
    assert "Holding the upper carry shelf beneath H1" in html
    assert "Holding above monthly value inside the upper month band" in html
    assert "H1 peak $100.40" in html
    assert "Prior month high $100.80" in html
    assert "ACTIVE SETUP LEVEL" not in html
    assert html.index("TODAY'S LIVE BATTLEFIELD + EXECUTION SPINE") < html.index(
        "EXECUTION STATE PACKETS"
    )
    assert html.index("TODAY'S LIVE BATTLEFIELD + EXECUTION SPINE") < html.index(
        "WEEKLY CONTEXT"
    )


def test_render_html_shows_cboe_quote_under_authoritative_price():
    html = render_live_read_html(
        pa={
            "spot": 737.09,
            "display_spot": 737.09,
            "spot_source": "yahoo_regular_market_price",
            "day_chg": -1.38,
            "vwap": 738.42,
            "price_authority": {
                "state": "yahoo_regular_market_price",
                "cboe_bid": 737.47,
                "cboe_ask": 737.49,
                "cboe_last_trade_time_raw": "2026-07-23T15:19:15",
            },
        },
        op={"put_wall": 728.0, "call_wall": 752.0},
        lines=[("Bears in control", "bad", "below VWAP")],
        permission=_permission(),
        stamp="15:39:00",
    )

    assert "$737.09" in html
    assert "yahoo_regular_market_price" in html
    assert "CBOE delayed options quote" in html
    assert "bid/ask $737.47 / $737.49" in html
    assert "mid $737.48" in html
    assert "Δ vs display +0.39" in html
    assert "context only, not top-price authority" in html


def test_stale_regime_read_is_backdrop_only():
    html = render_regime_read_block(
        {
            "available": True,
            "date": "2026-06-10",
            "headline": "REGIME mrll0",
            "story": "Structural backdrop: mid vol, rising voltrend.",
            "stale_days": 37,
        }
    )

    assert "REGIME BACKDROP ONLY" in html
    assert "STALE BATCH CONTEXT" in html
    assert "do not use this as today's live day classifier" in html


def test_render_html_places_live_battlefield_before_stale_regime():
    permission = _permission()
    permission["market_day"] = {
        "bucket": "unclassified_day",
        "score": 45,
        "bias": "NEUTRAL",
        "allowed_playbooks": [],
        "risk_posture": "wait_for_trigger",
        "reason": "not enough evidence for a clean day type",
    }
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": 0.1, "vwap": 100.1},
        op={"put_wall": 98.5, "call_wall": 101.5},
        lines=[("Neutral", "info", "waiting")],
        permission=permission,
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        regime_read={
            "available": True,
            "date": "2026-06-10",
            "headline": "REGIME mrll0",
            "story": "Structural backdrop only.",
            "stale_days": 37,
        },
        stamp="11:15:00",
    )

    assert "TODAY'S LIVE BATTLEFIELD: AWAITING CLEAN DAY TYPE" in html
    assert html.index("TODAY'S LIVE BATTLEFIELD") < html.index("REGIME BACKDROP ONLY")


def test_render_html_makes_handoff_setup_loud():
    permission = _permission()
    permission["setup_conviction"] = {
        **permission["setup_conviction"],
        "setup_gate": "ACTIONABLE",
        "setup_conviction_score": 79,
        "setup_tag": "EXHAUSTION -> RUNNER HANDOFF",
        "reason": "recent downside exhaustion reclaimed and expanded",
        "entry_gate": {"gate_id": "exhaustion_runner_handoff"},
        "context_gate": {"gate_id": "runner_day_directional_continuation"},
    }
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": 0.8, "vwap": 99.6},
        op={"put_wall": 98.5, "call_wall": 101.5},
        lines=[("Bulls in control", "ok", "runner context")],
        setups=[
            {
                "tag": "EXHAUSTION -> RUNNER HANDOFF",
                "bias": "CALLS (reversal promoted to runner)",
                "kind": "ok",
                "detail": "recent downside exhaustion reclaimed and expanded",
            }
        ],
        permission=permission,
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        permission_trend=_trend(),
        edge_token_position=_edge_token_position(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="11:15:00",
    )

    assert "PHASE PROMOTION • NOT JUST A VWAP FADE" in html
    assert (
        "Manage this like continuation now — the fade has already handed off." in html
    )
    assert "PHASE PROMOTION • CONTINUATION MANAGEMENT" in html
    assert "This setup has graduated beyond a simple VWAP fade." in html


def test_render_html_wraps_execution_logic_text_for_phone_viewer():
    html = render_permission_section(
        _permission(),
        pa={"spot": 100.0, "vwap": 99.8},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "detail": "reclaimed ORL after a violent liquidity vacuum with a stupidly long explanation string for wrapping",
            }
        ],
        permission_trend=_trend(),
    )

    assert "overflow-wrap:anywhere;word-break:break-word" in html


def test_render_permission_section_limits_main_spine_to_core_vectors():
    html = render_permission_section(
        _permission(),
        pa={"spot": 100.0, "vwap": 99.8},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[{"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)", "kind": "ok"}],
        permission_trend=_trend(),
    )

    assert "approval_decision is final authority" in html
    assert "Authority inputs:" in html
    assert "Diagnostic/supporting surfaces, not authority scores:" in html
    assert "Participation" in html
    assert "Structure" in html
    assert "Auction Acceptance" in html
    assert "Balance Context" in html
    assert "Secondary confirmations: Trap" in html
    assert "Pressure" in html
    assert "Balance Context" in html
    assert "Context governors: Balance Context" not in html
    assert "Advisory surfaces: Expansion Fuel" in html
    assert "state: BULLISH SEQUENCE • quality CONFIRMED" in html
    assert "state: CONFIRMED • local 1.40x • session 1.20x" in html
    assert "Execution state packet details (debug)" in html


def test_render_html_shows_active_setup_level_badge_for_failed_break():
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": -0.4, "vwap": 99.8},
        op={"put_wall": 98.5, "call_wall": 101.5},
        lines=[("Bear trap", "ok", "reclaim underway")],
        setups=[
            {
                "tag": "FAILED BREAKDOWN",
                "bias": "CALLS (bullish)",
                "kind": "ok",
                "detail": "reclaimed ORL",
                "level_name": "ORL",
                "level_price": 99.75,
                "trigger_price": 99.4,
                "bars_ago": 2,
            }
        ],
        permission=_permission(),
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        permission_trend=_trend(),
        edge_token_position=_edge_token_position(),
        weekly_context=_weekly_context(),
        monthly_context=_monthly_context(),
        stamp="11:15:00",
    )

    assert "ACTIVE SETUP LEVEL" in html
    assert "ORL" in html
    assert "$99.75" in html


def test_render_confluence_zones_block_empty_when_no_zones():
    assert render_confluence_zones_block(None) == ""
    assert render_confluence_zones_block({"zones": []}) == ""


def test_render_confluence_zones_block_shows_bounce_and_rejection():
    cz = {"zones": [
        {"side": "resistance", "stance": "rejection", "zone_lo": 765.99, "zone_hi": 766.49,
         "conviction": 60, "conviction_band": "medium", "factor_count": 9,
         "regime_gate": {"applied": "penalty"},
         "contributing_factors": [{"name": "VWAP"}, {"name": "EMA9"}],
         "trigger": "1m close rejecting from and failing back below 765.99"},
        {"side": "support", "stance": "bounce", "zone_lo": 764.17, "zone_hi": 765.0,
         "conviction": 28, "conviction_band": "low", "factor_count": 3,
         "regime_gate": {"applied": "trap_veto"},
         "contributing_factors": [{"name": "PUT_WALL"}, {"name": "PIN"}],
         "trigger": "reclaim required"},
    ]}
    html = render_confluence_zones_block(cz)
    assert "CONFLUENCE ZONES" in html
    assert "REJECTION" in html and "BOUNCE" in html
    assert "VWAP + EMA9" in html and "trap_veto" in html
