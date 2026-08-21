from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from candle_coach import build_candle_coach
from candle_coach_view import render_candle_coach_block
from candle_expectancy_adapter import lookup_candle_expectancy


def test_candle_expectancy_adapter_matches_tiered_matrix(tmp_path):
    db_path = tmp_path / "spy_truth.db"
    cols = {
        "match_tier": "TEXT",
        "event_name": "TEXT",
        "event_direction": "TEXT",
        "nearest_reference_name": "TEXT",
        "nearest_reference_relation": "TEXT",
        "reference_distance_bucket": "TEXT",
        "acceptance_state": "TEXT",
        "volume_confirmation": "TEXT",
        "vol_state": "TEXT",
        "macro_state": "TEXT",
        "dp_state": "TEXT",
        "regime_label": "TEXT",
        "open_regime_label": "TEXT",
        "time_bucket": "TEXT",
        "n": "INTEGER",
        "target_before_stop_rate": "REAL",
        "stop_before_target_rate": "REAL",
        "same_bar_rate": "REAL",
        "no_resolution_rate": "REAL",
        "up_target_first_rate": "REAL",
        "down_target_first_rate": "REAL",
        "avg_realized_R": "REAL",
        "avg_favorable_excursion_pct": "REAL",
        "avg_adverse_excursion_pct": "REAL",
        "sample_quality": "TEXT",
        "sample_bucket": "TEXT",
        "confidence_score": "REAL",
        "confidence_label": "TEXT",
        "positive_edge": "INTEGER",
        "deployment_tier": "TEXT",
        "deployment_ready": "INTEGER",
        "confidence_notes": "TEXT",
        "confidence_ts": "TEXT",
        "confidence_version": "TEXT",
    }
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE candle_confidence_matrix ("
        + ", ".join(f"{name} {kind}" for name, kind in cols.items())
        + ")"
    )
    row = {
        "match_tier": "tier_2_execution",
        "event_name": "bullish_conviction",
        "event_direction": "CALLS",
        "nearest_reference_name": "PDH",
        "nearest_reference_relation": "ANY",
        "reference_distance_bucket": "at_reference",
        "acceptance_state": "accepted_below",
        "volume_confirmation": "mixed",
        "vol_state": "ANY",
        "macro_state": "ANY",
        "dp_state": "ANY",
        "regime_label": "ANY",
        "open_regime_label": "ANY",
        "time_bucket": "opening_60m",
        "n": 42,
        "target_before_stop_rate": 0.57,
        "stop_before_target_rate": 0.31,
        "same_bar_rate": 0.02,
        "no_resolution_rate": 0.10,
        "up_target_first_rate": 0.60,
        "down_target_first_rate": 0.30,
        "avg_realized_R": 0.18,
        "avg_favorable_excursion_pct": 0.004,
        "avg_adverse_excursion_pct": 0.002,
        "sample_quality": "usable",
        "sample_bucket": "mature",
        "confidence_score": 62.5,
        "confidence_label": "MEDIUM",
        "positive_edge": 1,
        "deployment_tier": "WATCHLIST_ONLY",
        "deployment_ready": 0,
        "confidence_notes": "fixture row",
        "confidence_ts": "2026-07-28T00:00:00+00:00",
        "confidence_version": "test",
    }
    con.execute(
        f"INSERT INTO candle_confidence_matrix ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[name] for name in cols],
    )
    con.commit()
    con.close()

    result = lookup_candle_expectancy(
        {
            "latest_single": {
                "name": "Strong bullish candle",
                "candles": [{"minute": 30}],
            }
        },
        {
            "permission": {
                "location_state": {
                    "nearest_reference": {
                        "reference_name": "PDH",
                        "relation": "below",
                        "distance_pct": 0.05,
                    }
                },
                "acceptance_state": {"state": "accepted_below_level"},
                "volume_state": {"confirmation": "mixed"},
            },
            "volatility_structure": {"volatility_state": "squeeze"},
        },
        db_path=db_path,
    )

    assert result["available"] is True
    assert result["match_tier"] == "tier_2_execution"
    assert result["match"]["n"] == 42
    assert result["match"]["confidence_label"] == "MEDIUM"
    assert result["authority"] == "education_only_not_trade_permission"


def test_candle_coach_detects_bullish_engulfing():
    rows = [
        (0, 100.0, 100.15, 99.95, 100.05, 1000),
        (1, 100.05, 100.08, 99.8, 99.9, 1100),
        (2, 99.86, 100.28, 99.82, 100.22, 1500),
    ]

    packet = build_candle_coach(rows)

    assert packet["available"] is True
    assert packet["authority"] == "education_only_not_trade_permission"
    assert packet["latest_pair"]["name"] == "Bullish engulfing"
    assert "follow-through above" in packet["latest_pair"]["watch_next"]


def test_candle_coach_detects_dragonfly_doji_anatomy():
    rows = [
        (0, 100.0, 100.1, 99.9, 99.95, 1000),
        (1, 99.94, 100.02, 99.25, 99.98, 1800),
    ]

    packet = build_candle_coach(rows)

    assert packet["latest_single"]["name"] == "Dragonfly doji"
    assert packet["latest_single"]["anatomy"]["lower_wick_pct"] > 0.55


def test_candle_coach_detects_morning_star():
    packet = build_candle_coach(
        [
            (0, 100.4, 100.45, 99.7, 99.8, 2000),
            (1, 99.78, 99.9, 99.65, 99.82, 1100),
            (2, 99.83, 100.25, 99.8, 100.18, 2300),
        ]
    )

    assert packet["latest_three"]["name"] == "Morning star"
    assert len(packet["latest_three"]["candles"]) == 3


def test_candle_coach_detects_dark_cloud_cover():
    packet = build_candle_coach(
        [
            (0, 99.7, 100.35, 99.65, 100.3, 2000),
            (1, 100.32, 100.38, 99.85, 99.95, 2300),
        ]
    )

    assert packet["latest_pair"]["name"] == "Dark cloud cover"


def test_candle_coach_detects_bull_flag_structure():
    packet = build_candle_coach(
        [
            (0, 100.00, 100.08, 99.96, 100.02, 1000),
            (1, 100.02, 100.18, 100.00, 100.16, 1400),
            (2, 100.16, 100.31, 100.13, 100.29, 1600),
            (3, 100.29, 100.42, 100.25, 100.37, 1800),
            (4, 100.37, 100.40, 100.22, 100.31, 1200),
            (5, 100.31, 100.36, 100.18, 100.27, 1100),
            (6, 100.27, 100.32, 100.15, 100.25, 1050),
            (7, 100.25, 100.29, 100.14, 100.24, 1000),
        ]
    )

    structure = packet["latest_structure"]
    assert structure["name"] == "Bull flag / controlled pullback"
    assert structure["window"] == "8-candle structure"
    assert "breakout above the flag high" in structure["watch_next"]
    assert "Bull flag / controlled pullback" in packet["pattern_library"]


def test_candle_coach_detects_ascending_triangle_structure():
    packet = build_candle_coach(
        [
            (0, 100.0, 100.45, 99.95, 100.25, 1000),
            (1, 100.2, 100.50, 100.05, 100.35, 1100),
            (2, 100.3, 100.52, 100.15, 100.40, 1200),
            (3, 100.36, 100.51, 100.22, 100.44, 1300),
            (4, 100.40, 100.53, 100.30, 100.47, 1400),
            (5, 100.45, 100.52, 100.36, 100.49, 1500),
        ]
    )

    assert packet["latest_structure"]["name"] == "Ascending triangle"
    assert "compressing into resistance" in packet["latest_structure"]["meaning"]


def test_candle_coach_view_renders_education_only_block():
    packet = build_candle_coach(
        [
            (0, 100.05, 100.18, 99.98, 100.10, 900),
            (1, 100.08, 100.16, 99.96, 100.02, 950),
            (2, 100.0, 100.2, 99.95, 100.15, 1000),
            (3, 100.16, 100.18, 99.7, 99.8, 1300),
            (4, 99.8, 100.05, 99.75, 99.92, 1250),
        ]
    )

    html = render_candle_coach_block(packet)

    assert "CANDLE COACH" in html
    assert "education only" in html
    assert "LATEST 1-CANDLE EVENT" in html
    assert "LATEST 2-CANDLE CONFIGURATION" in html
    assert "LATEST 3-CANDLE PATTERN" in html
    assert "LARGER CANDLE STRUCTURE" in html
    assert "Pattern encyclopedia covered" in html
    assert "Deep candlestick encyclopedia" in html
    assert "False positive:" in html
    assert "<svg" in html
    assert "Watch next:" in html
    assert "CONDITIONAL TRADEABILITY GATES" in html
    assert "AUCTION EXECUTION BOX" in html
    assert "CBOE DELAYED OPTIONS PROXY" in html
    assert "Missing tape/depth" in html
    assert "EV = P(W)" in html


def test_candle_coach_exports_deep_candle_encyclopedia():
    packet = build_candle_coach(
        [
            (0, 100.0, 100.2, 99.9, 100.1, 1000),
            (1, 100.1, 100.4, 100.0, 100.35, 1800),
        ]
    )

    encyclopedia = packet["pattern_encyclopedia"]
    entries = encyclopedia["entries"]
    names = {entry["name"] for entry in entries}

    assert encyclopedia["authority"] == "education_only_not_trade_permission"
    assert encyclopedia["entry_count"] >= 25
    assert "Kicker" in names
    assert "Rising/Falling three methods" in names
    assert "Bullish/Bearish engulfing" in packet["pattern_library"]
    assert "Auction" not in entries[0]["name"]
    assert entries[0]["false_positive"]


def test_candle_coach_teaches_vectors_and_graph_canon():
    packet = build_candle_coach(
        [
            (0, 100.2, 100.25, 99.85, 99.9, 1000),
            (1, 99.9, 100.35, 99.86, 100.28, 1800),
        ],
        {
            "permission": {
                "scores": {
                    "trap_score": {
                        "score": 78,
                        "bias": "CALLS",
                        "reason": "sellers trapped below ORL",
                    },
                    "rejection_score": {
                        "score": 70,
                        "bias": "CALLS",
                        "reason": "last candle rejected lower prices",
                    },
                    "pressure_score": {
                        "score": 64,
                        "bias": "CALLS",
                        "reason": "buying pressure persists across bar closes",
                    },
                    "acceptance_score": {
                        "score": 68,
                        "bias": "CALLS",
                        "reason": "accepted back above ORL",
                    },
                    "location_score": {
                        "score": 72,
                        "bias": "NEUTRAL",
                        "reason": "near ORL reference",
                    },
                    "volume_score": {
                        "score": 85,
                        "bias": "CALLS",
                        "reason": "participation confirms move",
                    },
                },
                "graph_state": {
                    "graph_bias": "CALLS",
                    "graph_reason": "fresh setup marker favors calls: FAILED BREAKDOWN",
                    "authority_role": "operator_visual_canon",
                    "final_authority_source": "approval_decision_plus_operator",
                },
            }
        },
    )

    lesson = packet["candle_vector_lesson"]
    rows = {row["part"]: row for row in lesson["vector_rows"]}
    html = render_candle_coach_block(packet)

    assert lesson["authority"] == "education_only_not_trade_permission"
    assert lesson["graph_bridge"]["graph_bias"] == "CALLS"
    assert "Candle → execution vectors → graph canon" in lesson["headline"]
    assert rows["trap_score"]["score"] == 78
    assert rows["trap_score"]["correlation_family"] == "auction"
    assert rows["trap_score"]["graph_relation"] == "aligned_with_graph"
    assert "CANDLE → VECTOR → GRAPH LESSON" in html
    assert "sellers trapped below ORL" in html
    assert "fresh setup marker favors calls" in html


def test_candle_coach_uses_sharpedge_execution_surfaces():
    packet = build_candle_coach(
        [
            (0, 100.0, 100.2, 99.9, 100.1, 1000),
            (1, 100.1, 100.4, 100.0, 100.35, 1800),
        ],
        {
            "pa": {
                "volume_profile": {
                    "confirmation": "participating",
                    "local_mult": 1.4,
                    "session_mult": 1.1,
                    "composite_mult": 1.3,
                    "move_direction": "up",
                    "aligned_volume_share": 0.62,
                    "path_efficiency": 0.48,
                    "reason": "participating: local 1.4x, aligned 62%, efficiency 48%",
                },
            },
            "op": {
                "pcvr": 1.5,
                "pcr": 1.2,
                "atm_strike": 100.0,
                "atm_call_bid": 1.2,
                "atm_call_ask": 1.24,
                "atm_call_spread": 0.04,
                "atm_call_spread_pct": 0.0328,
                "atm_put_bid": 1.1,
                "atm_put_ask": 1.12,
                "atm_put_spread": 0.02,
                "atm_put_spread_pct": 0.018,
                "atm_iv": 0.18,
                "atm_call_iv": 0.17,
                "atm_put_iv": 0.19,
                "atm_iv_skew": 0.02,
                "call_volume_total": 1000,
                "put_volume_total": 1500,
                "call_volume_wall": 101.0,
                "put_volume_wall": 99.0,
                "call_wall": 102.0,
                "put_wall": 98.0,
            },
            "options_source": {
                "provider": "cboe",
                "endpoint": "delayed_quotes/options",
                "latest_option_trade_time_raw": "2026-07-22T14:00:00",
            },
            "permission": {
                "execution_permission_score": 67,
                "trade_gate": "CAUTION",
                "location_state": {
                    "state": "at_reference",
                    "reason": "at decision reference PDH 100.40",
                    "nearest_reference": {
                        "reference_name": "PDH",
                        "distance_pct": 0.05,
                        "relation": "below",
                    },
                },
                "acceptance_state": {
                    "state": "accepted_below_level",
                    "reason": "3 closes accepted below PDH 100.40",
                    "representative_level": {
                        "reason": "3 closes accepted below PDH 100.40"
                    },
                },
                "volume_state": {
                    "confirmation": "mixed",
                    "reason": "mixed: local 2.0x but efficiency low",
                },
                "dealer_state": {
                    "state": "positive_gamma_gravity",
                    "reason": "positive gamma pinning",
                },
            },
            "volatility_structure": {
                "volatility_state": "squeeze",
                "structure_state": "narrow_channel",
            },
            "micro": {
                "lower_wick": 12.5,
                "upper_wick": 3.1,
                "body": 44.0,
                "ch_pos": 82.0,
            },
            "magnitude": {
                "premium_read": "cheap",
                "exp_move_realized_pct": 0.45,
                "exp_move_implied_pct": 0.31,
            },
            "transition_pressure": {
                "transition_state": "pressure_building",
                "transition_pressure_score": 72,
                "attention_state": "focus",
                "reason": "pressure building near reference",
            },
        },
    )

    gates = {gate["label"]: gate for gate in packet["execution_framework"]["gates"]}
    assert gates["Location"]["status"] == "at_reference"
    assert "PDH" in gates["Location"]["message"]
    assert gates["Acceptance"]["status"] == "accepted_below_level"
    assert gates["Participation and order flow"]["status"] == "mixed"
    assert gates["Net expectancy"]["status"] in {
        "historical_context_attached",
        "deployment_ready_research_row",
    }
    assert packet["candle_expectancy"]["available"] is True
    assert packet["candle_expectancy"]["authority"] == (
        "education_only_not_trade_permission"
    )
    box = packet["execution_framework"]["auction_execution_box"]
    assert box["schema"] == "sharpedge.auction_execution_box.v1"
    assert box["acceptance"]["state"] == "accepted_below_level"
    assert box["participation"]["state"] == "mixed"
    assert box["micro_proxy"]["lower_wick"] == 12.5
    assert box["magnitude_context"]["premium_read"] == "cheap"
    assert box["transition_pressure"]["state"] == "pressure_building"
    assert box["options_flow_proxy"]["authority"] == (
        "delayed_options_proxy_not_live_tape"
    )
    assert box["options_flow_proxy"]["flow_pressure"]["state"] == (
        "put_volume_dominant"
    )
    assert box["options_flow_proxy"]["spread_proxy"]["put_quality"] == "tight"
    facts = {fact["label"]: fact for fact in box["facts"]}
    assert facts["Aggressor side / imbalance"]["status"] == "proxy"
    assert facts["Transition pressure"]["value"]["score"] == 72
    assert facts["CBOE option flow proxy"]["status"] == "delayed_proxy"
    assert "aggressor_side" in box["missing_microstructure"]
    assert "Acceptance supplies confirmation" in box["doctrine"]
    assert packet["execution_framework"]["next_vector_surface"]["name"] == (
        "candle_conditioned_expectancy_surface"
    )


def test_candle_coach_marks_stale_cboe_options_proxy():
    packet = build_candle_coach(
        [
            (0, 100.0, 100.2, 99.9, 100.1, 1000),
            (1, 100.1, 100.4, 100.0, 100.35, 1800),
        ],
        {
            "price_source": {"session_date": "2026-07-22"},
            "options_source": {
                "latest_option_trade_time_raw": "2026-07-21T16:14:59",
                "last_trade_time_raw": "2026-07-21T15:59:59",
            },
            "op": {
                "pcvr": 1.4,
                "pcr": 1.2,
                "call_volume_total": 1000,
                "put_volume_total": 1400,
                "atm_call_bid": 1.0,
                "atm_call_ask": 1.03,
                "atm_call_spread_pct": 0.0296,
            },
        },
    )

    box = packet["execution_framework"]["auction_execution_box"]
    proxy = box["options_flow_proxy"]
    facts = {fact["label"]: fact for fact in box["facts"]}
    assert proxy["available"] is False
    assert proxy["stale"] is True
    assert proxy["freshness"]["state"] == "stale_session_mismatch"
    assert "STALE CBOE OPTIONS DATA" in proxy["summary"]
    assert facts["CBOE option flow proxy"]["status"] == "stale_proxy"


def test_candle_coach_zero_range_blocks_directional_inference():
    packet = build_candle_coach(
        [
            (0, 100.0, 100.2, 99.9, 100.1, 1000),
            (1, 100.0, 100.0, 100.0, 100.0, 0),
        ]
    )

    assert packet["output_state"] == "No information"
    assert packet["latest_single"]["name"] == "Insufficient bar information"
    assert "No directional inference" in packet["data_integrity"]["message"]
    assert packet["execution_framework"]["expected_value_formula"].startswith("EV =")
