from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_flow_view import (  # noqa: E402
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
            "recommended_action": "watch_edges",
        }
    }
    before = (deepcopy(market_day), deepcopy(flow))

    html = render_market_day_block(market_day, flow)

    assert "BUCKET BRAIN: range_balance_day / 58 / NEUTRAL" in html
    assert "Allowed playbooks: magnet_fade" in html
    assert "Risk posture: fade edges, respect the magnet" in html
    assert "VWAP: hugging_vwap / magnet_chop" in html
    assert (market_day, flow) == before


def test_bucket_conditioned_spine_view_renders_action_without_mutating_input():
    spine = {
        "gate": "PERMIT",
        "score": 82,
        "bias": "CALLS",
        "bias_strength": 0.44,
        "recommended_action": "candidate_calls",
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

    assert "BUCKET-CONDITIONED SPINE: PERMIT / 82 / CALLS" in html
    assert "Action: candidate_calls" in html
    assert "trend_score 88" in html
    assert "structure_score 84" in html
    assert spine == before
