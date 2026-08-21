from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from gem_chart_svg import render_gem_chart_svg  # noqa: E402


def test_gem_chart_renders_channels_entry_zone_exit_and_fvg_zones():
    rows = [
        (0, 754.2, 754.5, 754.1, 754.3, 1000),
        (1, 754.3, 754.7, 754.25, 754.55, 1200),
        (2, 754.55, 754.9, 754.4, 754.8, 1300),
        (3, 754.8, 755.05, 754.6, 754.95, 1400),
    ]
    svg = render_gem_chart_svg(
        rows,
        {"vwap": 754.5, "vs_vwap": 0.06},
        {
            "label": "Magnet",
            "price": 759.0,
            "reachable_today": {"label": "Channel hi", "price": 755.2},
        },
        {"channel_low": 754.1, "channel_high": 755.2},
        {
            "nearest_open_gap_above": {
                "direction": "bearish",
                "gap_low": 755.3,
                "gap_high": 755.7,
            },
            "nearest_open_gap_below": {
                "direction": "bullish",
                "gap_low": 753.9,
                "gap_high": 754.2,
            },
        },
        {
            "actionable": True,
            "trigger_price": 754.62,
            "level_name": "ORL",
            "level_price": 754.18,
            "bars_ago": 1,
        },
    )

    assert "CHANNEL HI 755.20" in svg
    assert "CHANNEL LO 754.10" in svg
    assert "VWAP 754.50" in svg
    assert "ENTRY TRIGGER 754.62" in svg
    assert "FAIL ORL 754.18" in svg
    assert "ENTRY ZONE" in svg
    assert "TRIGGER CANDLE" in svg
    assert "EXIT CHANNEL HI 755.20" in svg
    assert "STRATEGIC MAGNET 759.00" in svg
    assert "BULLISH FVG 753.90-754.20" in svg
    assert "BEARISH FVG 755.30-755.70" in svg
    assert "SPOT 754.95" in svg
