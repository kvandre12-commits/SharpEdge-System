from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_read_view import render_live_read_html  # noqa: E402
from timeframe_agreement_view import render_timeframe_agreement_block  # noqa: E402


TIMEFRAME_PACKET = {
    "summary": "Higher-timeframe trend remains bullish, but intraday conditions favor fading extensions into resistance until participation or momentum improves.",
    "timeframes": {
        "weekly": {
            "timeframe": "Weekly",
            "label": "Bullish",
            "score": 82,
            "kind": "ok",
            "detail": "Holding the upper carry shelf beneath H1. Range position 73%.",
        },
        "daily": {
            "timeframe": "Daily",
            "label": "Bullish",
            "score": 78,
            "kind": "ok",
            "detail": "spot $105.50 vs 5d avg $104.90 / 20d avg $102.10; 20d range position 81%.",
        },
        "intraday": {
            "timeframe": "Intraday",
            "label": "Neutral/Caution",
            "score": 68,
            "kind": "warn",
            "detail": "CAUTION | watch edges. range_balance_day conditions the core spine.",
        },
    },
}


def test_timeframe_agreement_block_renders_all_three_rows():
    html = render_timeframe_agreement_block(TIMEFRAME_PACKET)

    assert "TIMEFRAME AGREEMENT" in html
    assert "Weekly" in html
    assert "Daily" in html
    assert "Intraday" in html
    assert "Bullish (82)" in html
    assert "Neutral/Caution (68)" in html


def test_live_read_html_includes_timeframe_agreement_near_top():
    html = render_live_read_html(
        pa={"spot": 105.5, "day_chg": 0.8},
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
        timeframe_agreement=TIMEFRAME_PACKET,
    )

    assert "TIMEFRAME AGREEMENT" in html
    assert "Higher-timeframe trend remains bullish" in html
    assert html.index("TIMEFRAME AGREEMENT") < html.index(
        "BUCKET-CONDITIONED EXECUTION SPINE"
    )
