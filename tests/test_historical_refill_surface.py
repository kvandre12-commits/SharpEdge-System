from __future__ import annotations

import json
from datetime import date, timedelta

from cockpit.historical_refill_surface import build_historical_refill_context
from cockpit.historical_refill_view import render_historical_refill_context_block


def test_historical_refill_context_and_view_include_fill_mode(tmp_path) -> None:
    surface_path = tmp_path / "historical_refill_surface.json"
    surface_path.write_text(
        json.dumps(
            {
                "schema": "sharpedge.historical_refill_surface.v1",
                "generated_at": "2026-07-20T00:00:00Z",
                "rows": [
                    {
                        "mode": "intraday_dip",
                        "threshold_pct": 1.5,
                        "event_count": 164,
                        "fill_rate_pct": 99.39,
                        "fill_median_trading_days": 1,
                        "fill_mode_trading_days": 0,
                        "fill_within_0d_count": 73,
                        "fill_mean_trading_days": 13.35,
                        "fill_max_trading_days": 479,
                        "fill_within_1d_count": 112,
                        "fill_within_1d_rate_pct": 68.29,
                        "fill_within_20d_count": 148,
                        "fill_within_20d_rate_pct": 90.24,
                        "fill_within_60d_rate_pct": 94.51,
                        "ema200_context": {
                            "basis": "prior close vs prior-session EMA200 at event time; no look-ahead",
                            "sides": {
                                "near_ema200": {
                                    "event_count": 42,
                                    "fill_rate_pct": 97.6,
                                    "median_trading_days": 1,
                                }
                            },
                            "distance_buckets": {
                                "near_ema200": {
                                    "event_count": 42,
                                    "fill_rate_pct": 97.6,
                                    "median_trading_days": 1,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stack_surface_path = tmp_path / "historical_refill_stack_surface.json"
    stack_stats = {
        "reference_count": 4,
        "new_event_fill_rate_pct": 100.0,
        "new_event_median_trading_days": 0,
        "new_event_max_trading_days": 5,
        "full_stack_resolved_rate_pct": 50.0,
        "full_stack_resolution_calendar_median_days": 8.5,
    }
    stack_surface_path.write_text(
        json.dumps(
            {
                "schema": "sharpedge.historical_refill_stack_surface.v1",
                "generated_at": "2026-07-20T00:00:00Z",
                "latest_event_date_excluded_from_stats": True,
                "basis": "test stack basis",
                "overall": stack_stats,
                "exact_signatures": {
                    "intraday_dip:1.5:near_ema200+intraday_dip:1.5:near_ema200": stack_stats
                },
                "by_last_pair_distance_bucket": {
                    "near_ema200 -> near_ema200": stack_stats
                },
                "by_last_pair_side": {"near_ema200 -> near_ema200": stack_stats},
                "by_new_ema_side": {"near_ema200": stack_stats},
                "by_new_ema_distance_bucket": {"near_ema200": stack_stats},
            }
        ),
        encoding="utf-8",
    )

    start = date(2026, 1, 1)
    bars = [
        {
            "date": (start + timedelta(days=offset)).isoformat(),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
        }
        for offset in range(200)
    ]
    bars.extend(
        [
            {
                "date": (start + timedelta(days=200)).isoformat(),
                "open": 99,
                "high": 99.7,
                "low": 98.4,
                "close": 99.6,
            },
            {
                "date": (start + timedelta(days=201)).isoformat(),
                "open": 99.1,
                "high": 99.2,
                "low": 98.0,
                "close": 98.9,
            },
        ]
    )

    ctx = build_historical_refill_context(
        bars,
        spot=98.9,
        surface_path=surface_path,
        stack_surface_path=stack_surface_path,
    )

    assert ctx["available"] is True
    assert ctx["event_count"] == 164
    assert ctx["active_refill_stack"]["active_count"] == 2
    assert ctx["active_refill_stack"]["stack_label"] == "double_dip_stack"
    assert len(ctx["active_refill_stack"]["items"]) == 2
    assert (
        ctx["active_refill_stack"]["interaction_signature"]
        == "intraday_dip:1.5:near_ema200+intraday_dip:1.5:near_ema200"
    )
    stack_history = ctx["active_refill_stack"]["historical_stack_surface"]
    assert stack_history["exact_signature_stats"]["reference_count"] == 4
    assert ctx["estimated"]["mode_trading_days"] == 0
    assert ctx["estimated"]["mode_count"] == 73
    assert round(ctx["estimated"]["mode_rate_pct"], 1) == 44.5
    assert ctx["ema200_context"]["side"] == "near_ema200"
    assert ctx["ema200_context"]["distance_bucket"] == "near_ema200"
    assert ctx["ema200_context"]["distance_bucket_stats"]["event_count"] == 42
    assert ctx["ema200_context"]["all_side_stats"]["near_ema200"]["event_count"] == 42
    assert (
        ctx["ema200_context"]["all_distance_bucket_stats"]["near_ema200"]["event_count"]
        == 42
    )
    html = render_historical_refill_context_block(ctx)
    assert "median fill" in html
    assert "1.0 trading days (112× / ≤1d / 68.3%)" in html
    assert "mode fill" in html
    assert "0.0 trading days (73× / 44.5%)" in html
    assert "mean fill" in html
    assert "13.3 trading days (148× / ≤20d / 90.2%)" in html
    assert "DOUBLE DIP STACK" in html
    assert "ACTIVE REFILL STACK" in html
    assert "EMA200 side" in html
    assert "42 events · 97.6% fill · med 1.0d" in html
    assert "STACK READ" in html
    assert "FAST LATEST REFILL / OLDER TARGET MAGNET" in html
    assert "latest refill score 100/100" in html
    assert "exact stack · 4 refs · latest med 0.0d" in html
    assert "HISTORICAL STACK SURFACE" in html
    assert (
        "4 refs · latest fill 100.0% · med 0.0d · max 5.0d · "
        "full stack 50.0% · stack med 8.5 cal-d" in html
    )
    assert "EMA200 REFILL BUCKET SURFACE" in html
    assert "Side buckets" in html
    assert "Distance buckets" in html
    assert "near_ema200 ← current" in html
