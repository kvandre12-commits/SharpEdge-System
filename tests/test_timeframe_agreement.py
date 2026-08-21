from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from timeframe_agreement import (
    build_daily_timeframe,
    build_intraday_timeframe,
    build_timeframe_agreement,
    build_weekly_timeframe,
)


def _daily_rows() -> list[dict[str, float | str]]:
    rows = []
    base = 96.0
    for index in range(20):
        close = base + index * 0.45
        rows.append(
            {
                "date": f"2026-06-{index + 1:02d}",
                "open": close - 0.25,
                "high": close + 0.4,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000 + index * 10_000,
            }
        )
    return rows


def _weekly_context() -> dict:
    return {
        "lookback_days": 5,
        "headline": "Holding the upper carry shelf beneath H1",
        "detail": "Spot is between LH1 and H1.",
        "kind": "ok",
        "range_position_pct": 73,
    }


def _caution_permission() -> dict:
    return {
        "trade_gate": "CAUTION",
        "trade_permission_score": 68,
        "bias": "CALLS",
        "bucket_conditioned_spine": {
            "gate": "CAUTION",
            "score": 68,
            "bias": "CALLS",
            "diagnostic_posture": "watch_edges_context_only",
            "reason": "range_balance_day conditions the core spine; participation is not convincing enough yet.",
        },
    }


def test_weekly_and_daily_timeframes_express_bullish_bias_cleanly():
    weekly = build_weekly_timeframe(_weekly_context())
    daily = build_daily_timeframe(_daily_rows(), spot=105.5)

    assert weekly["label"] == "Bullish"
    assert weekly["score"] == 82
    assert daily["label"] == "Bullish"
    assert daily["score"] >= 75


def test_intraday_caution_uses_neutral_caution_label():
    intraday = build_intraday_timeframe(_caution_permission())

    assert intraday["label"] == "Neutral/Caution"
    assert intraday["stance"] == "caution"
    assert intraday["score"] == 68
    assert intraday["basis"]["posture"] == "watch_edges_context_only"


def test_weekly_lower_shelf_can_still_be_constructive_when_high_in_range():
    weekly = build_weekly_timeframe(
        {
            "lookback_days": 5,
            "headline": "Leaning on the lower carry shelf",
            "kind": "warn",
            "range_position_pct": 90,
        }
    )

    assert weekly["label"] == "Bullish"
    assert weekly["score"] == 72
    assert weekly["kind"] == "warn"


def test_timeframe_agreement_builds_richer_summary_for_bullish_higher_timeframes():
    packet = build_timeframe_agreement(
        {"spot": 105.5, "vs_vwap": 0.18, "rng_pos": 78},
        _weekly_context(),
        _daily_rows(),
        _caution_permission(),
    )

    assert packet["higher_timeframe_bias"] == "bullish"
    assert packet["timeframes"]["weekly"]["label"] == "Bullish"
    assert packet["timeframes"]["daily"]["label"] == "Bullish"
    assert packet["timeframes"]["intraday"]["label"] == "Neutral/Caution"
    assert (
        packet["summary"]
        == "Higher-timeframe trend remains bullish, but intraday conditions favor fading extensions into resistance until participation or momentum improves."
    )
