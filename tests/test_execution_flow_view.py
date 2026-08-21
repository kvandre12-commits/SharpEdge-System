from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_flow_view import (
    bucket_display_label,
    render_bucket_conditioned_spine_block,
    render_market_day_block,
)


def test_market_day_view_renders_bucket_context_without_mutating_input():
    market_day = {
        "bucket": "range_balance_day",
        "score": 58,
        "bias": "NEUTRAL",
        "allowed_playbooks": ["magnet_fade"],
        "risk_posture": "fade edges, respect the magnet",
        "vwap_context": {
            "state": "hugging_vwap",
            "posture": "magnet_chop",
            "vs_vwap_pct": 0.03,
        },
        "reason": "VWAP magnet context says range/balance",
    }
    flow = {
        "bucket_conditioned_spine": {
            "gate": "CAUTION",
            "score": 65,
            "bias": "CALLS",
            "diagnostic_posture": "watch_edges_context_only",
        }
    }
    before = (deepcopy(market_day), deepcopy(flow))

    html = render_market_day_block(market_day, flow)

    assert "TODAY'S LIVE BATTLEFIELD: RANGE / BALANCE DAY / 58 / NEUTRAL" in html
    assert "Allowed playbooks: magnet_fade" in html
    assert "Risk posture: fade edges, respect the magnet" in html
    assert "VWAP: hugging_vwap / magnet_chop" in html
    assert (market_day, flow) == before


def test_market_day_view_humanizes_unclassified_day():
    html = render_market_day_block(
        {
            "bucket": "unclassified_day",
            "score": 45,
            "bias": "NEUTRAL",
            "allowed_playbooks": [],
            "risk_posture": "wait_for_trigger",
            "reason": "battlefield is not clean",
        },
        {},
    )

    assert bucket_display_label("unclassified_day") == "AWAITING CLEAN DAY TYPE"
    assert "TODAY'S LIVE BATTLEFIELD: AWAITING CLEAN DAY TYPE" in html
    assert "BUCKET BRAIN" not in html


def test_bucket_conditioned_spine_view_renders_diagnostic_posture_without_mutating_input():
    spine = {
        "gate": "PERMIT",
        "score": 82,
        "bias": "CALLS",
        "bias_strength": 0.44,
        "diagnostic_posture": "calls_context_only",
        "authority_role": "diagnostic_advisory",
        "advisory_only": True,
        "reason": "a_plus_trend_day conditions the core spine; base 76 with bucket offset +6.",
        "best": [
            {
                "name": "trend_score",
                "score": 88,
                "reason": "trend is aligned and persistent",
            },
            {
                "name": "structure_score",
                "score": 84,
                "reason": "higher highs and higher lows intact",
            },
        ],
    }
    before = deepcopy(spine)

    html = render_bucket_conditioned_spine_block(spine)

    assert "DIAGNOSTIC EXECUTION READ: PERMIT / 82 / CALLS" in html
    assert "Bucket-conditioned diagnostic posture: calls_context_only" in html
    assert "trend_score 88" in html
    assert "structure_score 84" in html
    assert spine == before
