from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_chart_svg import chart_svg  # noqa: E402
from live_read_view import render_price_feed_lag_line  # noqa: E402
from price_authority import apply_yahoo_display_price  # noqa: E402


def test_apply_yahoo_display_price_preserves_completed_bar_spot():
    pa = {
        "spot": 746.88,
        "day_open": 746.88,
        "day_chg": 0.0,
        "hi": 747.48,
        "lo": 746.37,
        "rng_pos": 46.0,
        "vwap": 746.95,
        "vs_vwap": -0.01,
    }
    source = {
        "regular_market_price": 747.29,
        "chart_previous_close": 748.32,
        "regular_market_time_utc": "2026-07-22T13:32:02+00:00",
        "last_bar_utc": "2026-07-22T13:32:02+00:00",
    }

    updated = apply_yahoo_display_price(pa, source)

    assert updated["spot"] == 747.29
    assert updated["display_spot"] == 747.29
    assert updated["analysis_spot"] == 746.88
    assert updated["spot_source"] == "yahoo_regular_market_price"
    assert updated["price_authority"]["state"] == "yahoo_regular_market_price"
    assert updated["day_chg"] == (747.29 / 748.32 - 1.0) * 100.0


def test_apply_yahoo_display_price_keeps_yahoo_quote_over_cboe_midpoint():
    pa = {
        "spot": 747.33,
        "day_open": 748.28,
        "day_chg": -0.13,
        "hi": 748.30,
        "lo": 746.37,
        "rng_pos": 50.0,
        "vwap": 748.28,
        "vs_vwap": -0.13,
    }
    price_source = {
        "regular_market_price": 747.41,
        "regular_market_time_utc": "2026-07-22T20:00:00+00:00",
        "last_bar_utc": "2026-07-22T20:00:00+00:00",
        "extended_session_price": 748.09,
        "extended_session_time_utc": "2026-07-22T21:01:49+00:00",
        "chart_previous_close": 748.28,
    }
    quote_source = {
        "current_price": 748.23,
        "bid": 748.17,
        "ask": 748.23,
        "last_trade_time_raw": "2026-07-22T15:59:59",
    }

    updated = apply_yahoo_display_price(pa, price_source, quote_source)

    assert updated["spot"] == 748.09
    assert updated["display_spot"] == 748.09
    assert updated["analysis_spot"] == 747.33
    assert updated["spot_source"] == "yahoo_extended_session_price"
    assert updated["price_authority"]["state"] == "yahoo_extended_session_price"
    assert updated["price_authority"]["cboe_bid"] == 748.17
    assert updated["price_authority"]["cboe_ask"] == 748.23


def test_apply_yahoo_display_price_prefers_live_quote_over_stale_yahoo():
    pa = {
        "spot": 738.25,
        "day_open": 738.18,
        "day_chg": 0.01,
        "hi": 739.95,
        "lo": 737.33,
        "rng_pos": 31.0,
        "vwap": 738.76,
        "vs_vwap": -0.07,
    }
    price_source = {
        "regular_market_price": 738.25,
        "regular_market_time_utc": "2000-01-01T14:18:39+00:00",
        "last_bar_utc": "2000-01-01T14:18:39+00:00",
        "chart_previous_close": 738.18,
    }
    live_quote_source = {
        "provider": "cnbc",
        "last_price": 740.23,
        "last_time_utc": "2999-01-01T15:09:37+00:00",
    }

    updated = apply_yahoo_display_price(
        pa,
        price_source,
        quote_source={},
        live_quote_source=live_quote_source,
    )

    assert updated["spot"] == 740.23
    assert updated["display_spot"] == 740.23
    assert updated["analysis_spot"] == 738.25
    assert updated["spot_source"] == "cnbc_last_price"
    assert updated["price_authority"]["state"] == "cnbc_last_price"
    assert updated["price_authority"]["live_quote_provider"] == "cnbc"
    assert updated["price_authority"]["live_quote_price"] == 740.23
    assert updated["price_authority"]["price_feed_stale"] is True
    assert updated["price_authority"]["price_feed_lag_state"] == "future_skew"
    assert updated["price_authority"]["price_feed_lag_minutes"] < 0


def test_apply_yahoo_display_price_uses_cboe_when_yahoo_quote_missing():
    pa = {
        "spot": 747.33,
        "day_open": 748.28,
        "day_chg": -0.13,
        "hi": 748.30,
        "lo": 746.37,
        "rng_pos": 50.0,
        "vwap": 748.28,
        "vs_vwap": -0.13,
    }
    quote_source = {
        "current_price": 748.23,
        "bid": 748.17,
        "ask": 748.23,
        "last_trade_time_raw": "2026-07-22T15:59:59",
    }

    updated = apply_yahoo_display_price(pa, {}, quote_source)

    assert updated["spot"] == 748.20
    assert updated["display_spot"] == 748.20
    assert updated["analysis_spot"] == 747.33
    assert updated["spot_source"] == "cboe_bid_ask_midpoint"
    assert updated["price_authority"]["state"] == "cboe_bid_ask_midpoint"


def test_apply_yahoo_display_price_prefers_fresh_extended_session_quote():
    pa = {
        "spot": 747.33,
        "day_open": 748.28,
        "day_chg": -0.13,
        "hi": 748.30,
        "lo": 746.37,
        "rng_pos": 50.0,
        "vwap": 748.28,
        "vs_vwap": -0.13,
    }
    source = {
        "regular_market_price": 747.41,
        "regular_market_time_utc": "2026-07-22T20:00:00+00:00",
        "last_bar_utc": "2026-07-22T20:00:00+00:00",
        "extended_session_price": 748.50,
        "extended_session_time_utc": "2026-07-22T20:30:51+00:00",
        "chart_previous_close": 748.28,
    }

    updated = apply_yahoo_display_price(pa, source)

    assert updated["spot"] == 748.50
    assert updated["display_spot"] == 748.50
    assert updated["analysis_spot"] == 747.33
    assert updated["spot_source"] == "yahoo_extended_session_price"
    assert updated["price_authority"]["state"] == "yahoo_extended_session_price"
    assert updated["price_authority"]["display_time_utc"] == "2026-07-22T20:30:51+00:00"


def test_apply_yahoo_display_price_flags_stale_source_timestamp():
    pa = {
        "spot": 738.25,
        "day_open": 738.18,
        "day_chg": 0.01,
        "hi": 739.95,
        "lo": 737.33,
        "rng_pos": 31.0,
        "vwap": 738.76,
        "vs_vwap": -0.07,
    }
    source = {
        "regular_market_price": 738.25,
        "regular_market_time_utc": "2000-01-01T14:18:39+00:00",
        "last_bar_utc": "2000-01-01T14:18:39+00:00",
        "chart_previous_close": 738.18,
    }

    updated = apply_yahoo_display_price(pa, source)

    assert updated["price_authority"]["price_feed_stale"] is True
    assert updated["price_authority"]["price_feed_lag_minutes"] > 15
    assert updated["price_authority"]["price_feed_lag_state"] == "stale"
    assert updated["price_authority"]["analysis_bar_stale"] is True
    assert updated["price_authority"]["analysis_bar_lag_minutes"] > 15


def test_render_price_feed_lag_line_warns_on_stale_price_authority():
    html = render_price_feed_lag_line(
        {
            "spot_source": "yahoo_regular_market_price",
            "price_authority": {
                "price_feed_stale": True,
                "price_feed_lag_minutes": 42.4,
                "price_feed_max_age_minutes": 15,
                "display_time_utc": "2026-07-24T14:18:39+00:00",
            },
        }
    )

    assert "PRICE FEED LAG" in html
    assert "yahoo_regular_market_price is 42.4 min old" in html
    assert "confirm against broker/live chart" in html


def test_render_price_feed_lag_line_warns_on_stale_analysis_bars_only():
    html = render_price_feed_lag_line(
        {
            "spot_source": "cnbc_last_price",
            "price_authority": {
                "price_feed_stale": False,
                "analysis_bar_stale": True,
                "analysis_bar_lag_minutes": 56.2,
                "analysis_bar_max_age_minutes": 15,
                "last_bar_utc": "2026-07-24T14:18:39+00:00",
            },
        }
    )

    assert "PRICE FEED LAG" not in html
    assert "ANALYTICS BAR LAG" in html
    assert "VWAP/momentum/volume may be stale" in html


def test_chart_svg_renders_cnbc_current_quote_marker():
    rows = [
        (0, 738.00, 738.50, 737.80, 738.25, 1313873),
        (1, 738.20, 738.60, 738.10, 738.30, 123460),
    ]
    pa = {
        "spot": 740.23,
        "display_spot": 740.23,
        "spot_source": "cnbc_last_price",
        "vwap": 738.95,
        "vs_vwap": 0.17,
    }

    svg = chart_svg(rows, pa)

    assert "CNBC 740.23" in svg
    assert "BAR 738.30" in svg


def test_chart_svg_renders_yahoo_current_quote_marker():
    rows = [
        (0, 746.62, 747.20, 746.37, 746.88, 1313873),
        (1, 746.92, 747.48, 746.86, 747.39, 123460),
    ]
    pa = {
        "spot": 747.29,
        "display_spot": 747.29,
        "spot_source": "yahoo_regular_market_price",
        "vwap": 746.95,
        "vs_vwap": 0.045,
    }

    svg = chart_svg(rows, pa)

    assert "YHOO 747.29" in svg
    assert "BAR 747.39" in svg
