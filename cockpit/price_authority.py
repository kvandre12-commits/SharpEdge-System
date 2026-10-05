"""Price authority helpers for SharpEdge cockpit.

Completed Yahoo regular-session bars drive analytics. Yahoo's freshest available
quote/extended-session bar drives the current SPY display/spot authority.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

SCHEMA = "sharpedge.price_authority.v1"
STALE_PRICE_MAX_AGE_MINUTES = 15


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_to_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _fresh_extended_quote(source: dict[str, Any]) -> tuple[float | None, str | None]:
    extended = _num(source.get("extended_session_price"))
    if extended is None or extended <= 0:
        return None, None
    extended_time = _iso_to_datetime(source.get("extended_session_time_utc"))
    regular_time = _iso_to_datetime(source.get("regular_market_time_utc"))
    last_bar_time = _iso_to_datetime(source.get("last_bar_utc"))
    stale_against = regular_time or last_bar_time
    if extended_time and stale_against and extended_time <= stale_against:
        return None, None
    return extended, source.get("extended_session_time_utc")


def _cboe_quote(source: dict[str, Any]) -> tuple[float | None, str, str, str | None]:
    bid = _num(source.get("bid"))
    ask = _num(source.get("ask"))
    if bid and ask and bid > 0 and ask > 0 and bid <= ask:
        return (
            (bid + ask) / 2.0,
            "cboe_bid_ask_midpoint",
            "cboe_bid_ask_midpoint",
            source.get("last_trade_time_raw"),
        )
    current = _num(source.get("current_price"))
    if current and current > 0:
        return (
            current,
            "cboe_current_price",
            "cboe_current_price",
            source.get("last_trade_time_raw"),
        )
    return None, "", "", None


def _live_quote(source: dict[str, Any]) -> tuple[float | None, str, str, str | None]:
    last = _num(source.get("last_price"))
    if last is None or last <= 0:
        return None, "", "", None
    provider = str(source.get("provider") or "live").lower()
    return (
        last,
        f"{provider}_last_price",
        f"{provider}_last_price",
        source.get("last_time_utc") or source.get("regular_market_time_utc"),
    )


def _display_quote(
    price_source: dict[str, Any],
    quote_source: dict[str, Any],
    live_quote_source: dict[str, Any],
) -> tuple[float | None, str, str, str | None]:
    """Choose the cockpit's top-line display price.

    Yahoo's quote/bar feed is the price authority for the live SPY header.
    CBOE options quotes are useful context, but the free delayed feed can lag
    intraday; it must not override a fresher Yahoo price at the top of the
    cockpit. Use CBOE only as a fallback when Yahoo has no usable quote.
    """
    live, live_source, live_state, live_time = _live_quote(live_quote_source)
    if live is not None:
        return live, live_source, live_state, live_time

    extended, extended_time = _fresh_extended_quote(price_source)
    if extended is not None:
        return (
            extended,
            "yahoo_extended_session_price",
            "yahoo_extended_session_price",
            extended_time,
        )
    regular = _num(price_source.get("regular_market_price"))
    if regular is not None and regular > 0:
        return (
            regular,
            "yahoo_regular_market_price",
            "yahoo_regular_market_price",
            price_source.get("regular_market_time_utc")
            or price_source.get("last_bar_utc"),
        )
    return _cboe_quote(quote_source)


def _lag_packet(timestamp_utc: str | None) -> dict[str, Any]:
    timestamp = _iso_to_datetime(timestamp_utc)
    if timestamp is None:
        return {"stale": None, "lag_minutes": None, "lag_state": "unknown"}
    now = dt.datetime.now(dt.UTC)
    age_seconds = (now - timestamp).total_seconds()
    if age_seconds < -60.0:
        return {
            "stale": True,
            "lag_minutes": round(age_seconds / 60.0, 1),
            "lag_state": "future_skew",
        }
    age_minutes = max(0.0, age_seconds) / 60.0
    stale = age_minutes > STALE_PRICE_MAX_AGE_MINUTES
    return {
        "stale": stale,
        "lag_minutes": round(age_minutes, 1),
        "lag_state": "stale" if stale else "fresh",
    }


def _price_lag_packet(display_time_utc: str | None) -> dict[str, Any]:
    lag = _lag_packet(display_time_utc)
    return {
        "price_feed_stale": lag["stale"],
        "price_feed_lag_minutes": lag["lag_minutes"],
        "price_feed_lag_state": lag["lag_state"],
        "price_feed_max_age_minutes": STALE_PRICE_MAX_AGE_MINUTES,
    }


def _analysis_bar_lag_packet(last_bar_utc: str | None) -> dict[str, Any]:
    lag = _lag_packet(last_bar_utc)
    return {
        "analysis_bar_stale": lag["stale"],
        "analysis_bar_lag_minutes": lag["lag_minutes"],
        "analysis_bar_lag_state": lag["lag_state"],
        "analysis_bar_max_age_minutes": STALE_PRICE_MAX_AGE_MINUTES,
    }


def _day_change_pct(price: float, source: dict[str, Any], fallback: float) -> float:
    previous_close = _num(source.get("chart_previous_close"))
    if previous_close and previous_close > 0:
        return (price / previous_close - 1.0) * 100.0
    return fallback


def _range_position(price: float, low: float, high: float) -> float:
    span = (high - low) or 1e-9
    return (price - low) / span * 100.0


def apply_yahoo_display_price(
    pa: dict[str, Any],
    price_source: dict[str, Any] | None,
    quote_source: dict[str, Any] | None = None,
    live_quote_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return price-action packet with Yahoo's freshest quote as display spot.

    The original bar-close spot is preserved as ``analysis_spot``. If Yahoo's
    current quote is unavailable, the packet is returned with explicit fallback
    metadata and the completed-bar spot remains authoritative.
    """
    source = price_source or {}
    quote = quote_source or {}
    live_quote = live_quote_source or {}
    updated = dict(pa)
    bar_spot = _num(pa.get("spot"))
    display, spot_source, authority_state, display_time_utc = _display_quote(
        source, quote, live_quote
    )
    lag_packet = {
        **_price_lag_packet(display_time_utc),
        **_analysis_bar_lag_packet(source.get("last_bar_utc")),
    }
    if display is None or display <= 0:
        updated["analysis_spot"] = bar_spot
        updated["display_spot"] = bar_spot
        updated["spot_source"] = "yahoo_completed_bar_close"
        updated["price_authority"] = {
            "schema": SCHEMA,
            "state": "fallback_completed_bar_close",
            "reason": "External quote unavailable; using latest completed analytics bar close.",
            "analysis_spot": bar_spot,
            "display_spot": bar_spot,
            **lag_packet,
        }
        return updated

    old_hi = (
        _num(
            updated.get("hi"),
        )
        or display
    )
    old_lo = (
        _num(
            updated.get("lo"),
        )
        or display
    )
    high = max(old_hi, display)
    low = min(old_lo, display)
    vwap = _num(updated.get("vwap"))
    updated["analysis_spot"] = bar_spot
    updated["display_spot"] = display
    updated["spot"] = display
    updated["spot_source"] = spot_source
    updated["hi"] = high
    updated["lo"] = low
    updated["rng_pos"] = _range_position(display, low, high)
    if vwap:
        updated["vs_vwap"] = (display - vwap) / vwap * 100.0
    updated["day_chg"] = _day_change_pct(
        display, source, _num(pa.get("day_chg"), 0.0) or 0.0
    )
    updated["price_authority"] = {
        "schema": SCHEMA,
        "state": authority_state,
        "reason": "Current SPY display/spot uses the freshest available quote; analytics bars remain completed Yahoo regular-session 1m bars.",
        "analysis_spot": bar_spot,
        "display_spot": display,
        "regular_market_time_utc": source.get("regular_market_time_utc"),
        "last_bar_utc": source.get("last_bar_utc"),
        "extended_session_time_utc": source.get("extended_session_time_utc"),
        "display_time_utc": display_time_utc,
        **lag_packet,
        "chart_previous_close": source.get("chart_previous_close"),
        "cboe_current_price": quote.get("current_price"),
        "cboe_bid": quote.get("bid"),
        "cboe_ask": quote.get("ask"),
        "cboe_last_trade_time_raw": quote.get("last_trade_time_raw"),
        "live_quote_provider": live_quote.get("provider"),
        "live_quote_price": live_quote.get("last_price"),
        "live_quote_time_utc": live_quote.get("last_time_utc"),
    }
    return updated


__all__ = ["apply_yahoo_display_price"]
