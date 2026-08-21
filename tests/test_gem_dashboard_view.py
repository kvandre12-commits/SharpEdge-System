from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from gem_dashboard_view import render_gem_dashboard_html


def test_gem_dashboard_renders_master_state_execution_focus_and_screamer_filter():
    signal = {
        "spot": 754.95,
        "vs_vwap": 0.19,
        "rng_pos": 92.9,
        "vol_mult": 3.68,
        "gamma_regime": "positive",
        "entry_setup_tag": "DOWNSIDE EXHAUSTION",
        "target_plan": {
            "label": "Magnet",
            "price": 759.0,
            "objective": "exhaustion_fade",
            "reason": "fade back toward value",
            "likely_travel": "partial reversion only",
            "reachable_today": {"label": "VWAP", "price": 755.4},
        },
        "entry_gate": {
            "actionable": True,
            "trigger_price": 754.6,
            "level_name": "VWAP",
            "level_price": 754.1,
            "bars_ago": 1,
        },
        "trade_permission": {
            "trade_gate": "PERMIT",
            "trade_permission_score": 78,
            "bias": "CALLS",
            "setup_conviction": {
                "setup_gate": "ACTIONABLE",
                "bias": "CALLS",
                "setup_tag": "DOWNSIDE EXHAUSTION",
                "reason": "long lower wick into exhaustion pocket",
            },
            "scores": {
                "structure_score": {"score": 78},
                "acceptance_score": {"score": 75},
                "trend_score": {"score": 61},
                "volume_score": {"score": 58},
                "exhaustion_score": {
                    "score": 81,
                    "reason": "stretched low reclaimed into value",
                },
            },
        },
        "permission_score_trend": {
            "points": [
                {"time": "13:00", "score": 65, "event_markers": []},
                {
                    "time": "13:05",
                    "score": 78,
                    "event_markers": ["DOWNSIDE EXHAUSTION CANDIDATE"],
                },
            ],
            "direction": "strengthening",
            "delta": 13,
        },
        "fair_value_gaps": {
            "nearest_open_gap": {
                "direction": "bullish",
                "gap_low": 753.2,
                "gap_high": 753.7,
                "fill_state": "open",
                "fill_pct": 0.0,
                "age_bars": 4,
                "position_vs_spot": "below",
                "distance_from_spot": 1.5,
                "fill_direction": "down",
            },
            "nearest_open_gap_above": {},
            "nearest_open_gap_below": {
                "direction": "bullish",
                "gap_low": 753.2,
                "gap_high": 753.7,
                "fill_state": "open",
                "fill_pct": 0.0,
                "age_bars": 4,
                "position_vs_spot": "below",
                "distance_from_spot": 1.5,
                "fill_direction": "down",
            },
        },
        "decision_receipt": {
            "primary_setup_event": {
                "event_type": "DOWNSIDE EXHAUSTION",
                "status": "confirmed",
                "level": {"name": "VWAP"},
            },
            "setup_event_transitions": [],
        },
    }

    html = render_gem_dashboard_html(signal, "14:22:10")

    assert "SHARPEDGE • GEM FIRST" in html
    assert "SPY $754.95" in html
    assert "MASTER STATE" in html
    assert ">LIVE<" in html
    assert "Trigger armed at $754.60; fail level $754.10; permission 78/100." in html
    assert "GEM GRAPH" in html
    assert "gem_chart.svg" in html
    assert "PERMISSION TREND" in html
    assert "CALLS LANE" in html
    assert "PUTS LANE" in html
    assert "SCREAMER FILTER" in html
    assert "LIVE SCREAMER" in html
    assert "EXECUTION PLAN" in html
    assert "ENTRY $754.60 • EXIT $755.40" in html
    assert "kill-switch VWAP $754.10" in html
    assert "trigger candle 1 bars ago" in html
    assert "EXHAUSTION MARKER" in html
    assert "TRAVERSE TARGET" in html
    assert "FAIR VALUE GAP MAP" in html
    assert "BULLISH FVG 753.20-753.70" in html
    assert "DOWNSIDE EXHAUSTION CONFIRMED @ VWAP" in html
