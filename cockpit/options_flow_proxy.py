"""Delayed CBOE options-flow proxy for the Auction Execution Box.

This is explicitly not live tape, depth, or aggressor-side proof. It converts
available delayed chain fields into honest execution context.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "sharpedge.cboe_options_flow_proxy.v1"
AUTHORITY = "delayed_options_proxy_not_live_tape"


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _spread_quality(spread_pct: float | None) -> str:
    if spread_pct is None:
        return "unknown"
    if spread_pct <= 0.03:
        return "tight"
    if spread_pct <= 0.08:
        return "usable"
    return "wide"


def _flow_pressure(pcvr: float | None) -> str:
    if pcvr is None:
        return "unknown"
    if pcvr >= 1.25:
        return "put_volume_dominant"
    if pcvr <= 0.80:
        return "call_volume_dominant"
    return "balanced_volume"


def _iv_skew_read(skew: float | None) -> str:
    if skew is None:
        return "unknown"
    if skew >= 0.02:
        return "puts_richer_than_calls"
    if skew <= -0.02:
        return "calls_richer_than_puts"
    return "near_flat_atm_skew"


def _date_prefix(value: Any) -> str | None:
    text = str(value or "")
    return text[:10] if len(text) >= 10 and text[4:5] == "-" else None


def _freshness(
    source: dict[str, Any], expected_session_date: str | None
) -> dict[str, Any]:
    latest_trade_date = _date_prefix(source.get("latest_option_trade_time_raw"))
    last_trade_date = _date_prefix(source.get("last_trade_time_raw"))
    observed_date = latest_trade_date or last_trade_date
    if (
        expected_session_date
        and observed_date
        and observed_date != expected_session_date
    ):
        state = "stale_session_mismatch"
        usable = False
        reason = (
            f"CBOE latest option trade date {observed_date} does not match price session "
            f"{expected_session_date}; suppressing delayed options-flow read."
        )
    elif expected_session_date and not observed_date:
        state = "unknown_option_trade_date"
        usable = False
        reason = (
            "CBOE option trade timestamp is missing; suppressing options-flow read."
        )
    else:
        state = "session_aligned_or_unchecked"
        usable = True
        reason = "CBOE option trade timestamp is aligned with the expected session or no expected session was supplied."
    return {
        "state": state,
        "usable": usable,
        "expected_price_session_date": expected_session_date,
        "latest_option_trade_date": latest_trade_date,
        "last_trade_date": last_trade_date,
        "reason": reason,
    }


def _source_packet(
    source: dict[str, Any], expected_session_date: str | None
) -> dict[str, Any]:
    return {
        "provider": source.get("provider"),
        "endpoint": source.get("endpoint"),
        "latest_option_trade_time_raw": source.get("latest_option_trade_time_raw"),
        "last_trade_time_raw": source.get("last_trade_time_raw"),
        "expected_price_session_date": expected_session_date,
        "delay_note": "CBOE delayed quotes are not live execution tape; expect about 15 minutes of delay and market-hours availability limits.",
    }


def build_options_flow_proxy(
    op: dict[str, Any] | None,
    source: dict[str, Any] | None = None,
    expected_session_date: str | None = None,
) -> dict[str, Any]:
    """Build an honest options-flow proxy from delayed CBOE chain fields."""
    op = op or {}
    source = source or {}
    pcvr = _num(op.get("pcvr"))
    pcr = _num(op.get("pcr"))
    call_spread_pct = _num(op.get("atm_call_spread_pct"))
    put_spread_pct = _num(op.get("atm_put_spread_pct"))
    iv_skew = _num(op.get("atm_iv_skew"))
    call_volume = _num(op.get("call_volume_total"))
    put_volume = _num(op.get("put_volume_total"))
    flow_state = _flow_pressure(pcvr)
    call_quality = _spread_quality(call_spread_pct)
    put_quality = _spread_quality(put_spread_pct)
    freshness = _freshness(source, expected_session_date)

    fields_available = any(
        value is not None
        for value in (
            pcvr,
            pcr,
            call_spread_pct,
            put_spread_pct,
            iv_skew,
            call_volume,
            put_volume,
        )
    )
    available = fields_available and bool(freshness["usable"])
    if not fields_available:
        summary = "No delayed CBOE options proxy fields available."
    elif not freshness["usable"]:
        summary = f"STALE CBOE OPTIONS DATA: {freshness['reason']}"
    else:
        summary = (
            f"{flow_state}; ATM spreads call={call_quality}, put={put_quality}; "
            f"ATM skew {_iv_skew_read(iv_skew)}. Proxy only — no prints/depth/aggressor side."
        )

    return {
        "schema": SCHEMA,
        "available": available,
        "stale": fields_available and not freshness["usable"],
        "authority": AUTHORITY,
        "freshness": freshness,
        "source": _source_packet(source, expected_session_date),
        "flow_pressure": {
            "state": flow_state,
            "pcvr": pcvr,
            "pcr": pcr,
            "call_volume_total": call_volume,
            "put_volume_total": put_volume,
            "read": "Call/put volume pressure is delayed and cannot identify opening vs closing or buyer vs seller initiation.",
        },
        "spread_proxy": {
            "atm_strike": op.get("atm_strike"),
            "call_bid": _num(op.get("atm_call_bid")),
            "call_ask": _num(op.get("atm_call_ask")),
            "call_spread": _num(op.get("atm_call_spread")),
            "call_spread_pct": call_spread_pct,
            "call_quality": call_quality,
            "put_bid": _num(op.get("atm_put_bid")),
            "put_ask": _num(op.get("atm_put_ask")),
            "put_spread": _num(op.get("atm_put_spread")),
            "put_spread_pct": put_spread_pct,
            "put_quality": put_quality,
        },
        "strike_volume_concentration": {
            "call_volume_wall": op.get("call_volume_wall"),
            "put_volume_wall": op.get("put_volume_wall"),
            "call_wall": op.get("call_wall"),
            "put_wall": op.get("put_wall"),
            "read": "Volume/OI walls locate interest, not live queue depth or replenishment.",
        },
        "iv_context": {
            "atm_iv": _num(op.get("atm_iv")),
            "atm_call_iv": _num(op.get("atm_call_iv")),
            "atm_put_iv": _num(op.get("atm_put_iv")),
            "atm_iv_skew": iv_skew,
            "read": _iv_skew_read(iv_skew),
        },
        "microstructure_map": {
            "bid_ask_spread_at_level": "proxy_available_from_atm_option_bid_ask",
            "order_flow_imbalance": "delayed_proxy_from_call_put_volume_ratio",
            "sweep_or_absorption_proof": "proxy_context_only_not_proof",
            "aggressor_side": "unavailable_without_bid_ask_tagged_prints",
            "depth_ladder": "unavailable_without_order_book_depth",
            "replenishment_and_cancellation": "unavailable_without_quote_update_stream",
            "print_sequence": "unavailable_without_tick_trade_sequence",
        },
        "summary": summary,
    }


__all__ = ["build_options_flow_proxy"]
