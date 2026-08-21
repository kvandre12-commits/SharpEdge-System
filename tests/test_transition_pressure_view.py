from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_read_view import render_live_read_html
from transition_pressure_view import render_transition_pressure_block

PACKET = {
    "transition_pressure_score": 78,
    "transition_state": "release_candidate",
    "attention_state": "require_trigger",
    "directional_bias": "upside_release_possible",
    "persistence": {
        "state": "holding",
        "bars": 3,
        "label": "holding_3_bars",
    },
    "reason": "permission rising, acceptance building, gamma constraint active",
    "permission_leads_price": {
        "active": True,
        "reason": "permission +8 across 3 reads while spot changed +0.01%",
    },
    "deltas": {
        "permission_delta": {
            "velocity": 8,
            "acceleration": 3,
            "status": "accelerating",
        },
        "trend_delta": {"velocity": 6, "acceleration": 2, "status": "strengthening"},
        "acceptance_delta": {
            "velocity": 7,
            "acceleration": 1,
            "status": "strengthening",
        },
        "participation_delta": {
            "velocity": 5,
            "acceleration": 1,
            "status": "strengthening",
        },
    },
    "potential_energy": {
        "compression_score": {"score": 80},
        "failed_auction_score": {"score": 55},
        "location_pressure": {"score": 68},
        "gamma_constraint": {"score": 72},
    },
}


def test_transition_pressure_block_renders_summary_and_attention():
    html = render_transition_pressure_block(PACKET)

    assert "TRANSITION PRESSURE" in html
    assert "RELEASE_CANDIDATE (78)" in html
    assert "attention: require trigger" in html
    assert "persistence: holding 3 bars" in html
    assert "bias: upside release possible" in html
    assert "permission leading price" in html
    assert "Permission Δ" in html


def test_live_read_html_includes_transition_pressure_above_timeframe_agreement():
    html = render_live_read_html(
        pa={"spot": 100.0, "day_chg": 0.5},
        op={},
        lines=[("BULLS in control", "ok", "price above VWAP")],
        setups=[],
        permission={},
        micro={},
        magnitude={},
        gp={},
        permission_trend={},
        edge_token_position={},
        regime_refinement={},
        weekly_context={},
        monthly_context={},
        stamp="10:15:00",
        timeframe_agreement={
            "summary": "Higher-timeframe trend remains bullish.",
            "timeframes": {
                "weekly": {
                    "timeframe": "Weekly",
                    "label": "Bullish",
                    "score": 82,
                    "kind": "ok",
                    "detail": "weekly",
                },
                "daily": {
                    "timeframe": "Daily",
                    "label": "Bullish",
                    "score": 76,
                    "kind": "ok",
                    "detail": "daily",
                },
                "intraday": {
                    "timeframe": "Intraday",
                    "label": "Neutral/Caution",
                    "score": 68,
                    "kind": "warn",
                    "detail": "intra",
                },
            },
        },
        transition_pressure=PACKET,
    )

    assert "TRANSITION PRESSURE" in html
    assert "TIMEFRAME AGREEMENT" in html
    assert html.index("TRANSITION PRESSURE") < html.index("TIMEFRAME AGREEMENT")
