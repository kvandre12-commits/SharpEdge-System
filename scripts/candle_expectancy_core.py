"""Core helpers for candle-conditioned expectancy events.

This module is deliberately causal: event/context fields may only use bars and
state known at the event bar. Forward bars are used only for outcome labels.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any

TARGET_FIRST = "target_before_stop"
STOP_FIRST = "stop_before_target"
SAME_BAR = "same_bar_target_stop"
NO_RESOLUTION = "no_resolution"
NO_DIRECTION = "no_direction"


def num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(part: float, whole: float) -> float | None:
    if not whole:
        return None
    return part / whole


def parse_ts(ts_text: str) -> datetime:
    return datetime.fromisoformat(str(ts_text).replace("Z", "+00:00"))


def minutes_since_open(ts_text: str) -> int | None:
    try:
        dt = parse_ts(ts_text)
    except ValueError:
        return None
    return (dt.hour * 60 + dt.minute) - (9 * 60 + 30)


def candle_anatomy(bar: dict[str, Any]) -> dict[str, Any]:
    open_ = num(bar.get("open"))
    high = num(bar.get("high"))
    low = num(bar.get("low"))
    close = num(bar.get("close"))
    raw_range = max(high - low, 0.0)
    safe_range = max(raw_range, 1e-9)
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    direction = "bull" if close > open_ else "bear" if close < open_ else "flat"
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": int(num(bar.get("volume"), 0.0)),
        "trade_count": bar.get("trade_count"),
        "vwap": bar.get("vwap"),
        "range": raw_range,
        "range_pct": pct(raw_range, close) or 0.0,
        "body_pct": body / safe_range,
        "upper_wick_pct": upper / safe_range,
        "lower_wick_pct": lower / safe_range,
        "direction": direction,
    }


def classify_event(
    current: dict[str, Any], previous: dict[str, Any] | None = None
) -> dict[str, str]:
    """Classify a candle event without implying trade permission."""
    cur = candle_anatomy(current)
    prev = candle_anatomy(previous) if previous else None
    body = cur["body_pct"]
    upper = cur["upper_wick_pct"]
    lower = cur["lower_wick_pct"]
    direction = cur["direction"]

    if cur["range"] <= 0 or cur["volume"] <= 0:
        return {"event_name": "insufficient_bar", "event_direction": "NEUTRAL"}

    if prev:
        prev_body_low = min(prev["open"], prev["close"])
        prev_body_high = max(prev["open"], prev["close"])
        cur_body_low = min(cur["open"], cur["close"])
        cur_body_high = max(cur["open"], cur["close"])
        if cur["high"] < prev["high"] and cur["low"] > prev["low"]:
            return {"event_name": "inside_bar", "event_direction": "NEUTRAL"}
        if cur["high"] > prev["high"] and cur["low"] < prev["low"]:
            return {"event_name": "outside_bar", "event_direction": "NEUTRAL"}
        if (
            prev["direction"] == "bear"
            and direction == "bull"
            and cur_body_low <= prev_body_low
            and cur_body_high >= prev_body_high
        ):
            return {"event_name": "bullish_engulfing", "event_direction": "CALLS"}
        if (
            prev["direction"] == "bull"
            and direction == "bear"
            and cur_body_low <= prev_body_low
            and cur_body_high >= prev_body_high
        ):
            return {"event_name": "bearish_engulfing", "event_direction": "PUTS"}

    if body <= 0.10 and lower >= 0.55:
        return {"event_name": "dragonfly_doji", "event_direction": "CALLS"}
    if body <= 0.10 and upper >= 0.55:
        return {"event_name": "gravestone_doji", "event_direction": "PUTS"}
    if lower >= 0.55 and upper <= 0.20 and body <= 0.35:
        return {"event_name": "demand_tail", "event_direction": "CALLS"}
    if upper >= 0.55 and lower <= 0.20 and body <= 0.35:
        return {"event_name": "supply_tail", "event_direction": "PUTS"}
    if body <= 0.10:
        return {"event_name": "doji", "event_direction": "NEUTRAL"}
    if body <= 0.35 and upper >= 0.25 and lower >= 0.25:
        return {"event_name": "spinning_top", "event_direction": "NEUTRAL"}
    if body >= 0.65 and direction == "bull":
        return {"event_name": "bullish_conviction", "event_direction": "CALLS"}
    if body >= 0.65 and direction == "bear":
        return {"event_name": "bearish_conviction", "event_direction": "PUTS"}
    return {"event_name": "ordinary_range", "event_direction": "NEUTRAL"}


def first_touch_outcome(
    entry: float,
    forward_bars: list[dict[str, Any]],
    direction: str,
    target_pct: float,
    stop_pct: float,
) -> dict[str, Any]:
    """Return ordered target/stop outcome for a directional hypothesis."""
    if direction not in {"CALLS", "PUTS"} or entry <= 0:
        return {
            "target_before_stop_label": NO_DIRECTION,
            "bars_to_resolution": None,
            "realized_R": None,
        }

    if direction == "CALLS":
        target = entry * (1.0 + target_pct)
        stop = entry * (1.0 - stop_pct)
    else:
        target = entry * (1.0 - target_pct)
        stop = entry * (1.0 + stop_pct)

    for offset, bar in enumerate(forward_bars, start=1):
        high = num(bar.get("high"))
        low = num(bar.get("low"))
        hit_target = high >= target if direction == "CALLS" else low <= target
        hit_stop = low <= stop if direction == "CALLS" else high >= stop
        if hit_target and hit_stop:
            return {
                "target_before_stop_label": SAME_BAR,
                "bars_to_resolution": offset,
                "realized_R": 0.0,
            }
        if hit_target:
            return {
                "target_before_stop_label": TARGET_FIRST,
                "bars_to_resolution": offset,
                "realized_R": 1.0,
            }
        if hit_stop:
            return {
                "target_before_stop_label": STOP_FIRST,
                "bars_to_resolution": offset,
                "realized_R": -1.0,
            }
    return {
        "target_before_stop_label": NO_RESOLUTION,
        "bars_to_resolution": None,
        "realized_R": 0.0,
    }


def two_sided_first_touch(
    entry: float, forward_bars: list[dict[str, Any]], target_pct: float
) -> dict[str, Any]:
    if entry <= 0:
        return {
            "two_sided_first_touch": "invalid_entry",
            "bars_to_two_sided_touch": None,
        }
    up_target = entry * (1.0 + target_pct)
    down_target = entry * (1.0 - target_pct)
    for offset, bar in enumerate(forward_bars, start=1):
        hit_up = num(bar.get("high")) >= up_target
        hit_down = num(bar.get("low")) <= down_target
        if hit_up and hit_down:
            return {
                "two_sided_first_touch": "both_same_bar",
                "bars_to_two_sided_touch": offset,
            }
        if hit_up:
            return {
                "two_sided_first_touch": "up_target_first",
                "bars_to_two_sided_touch": offset,
            }
        if hit_down:
            return {
                "two_sided_first_touch": "down_target_first",
                "bars_to_two_sided_touch": offset,
            }
    return {"two_sided_first_touch": NO_RESOLUTION, "bars_to_two_sided_touch": None}


def excursion_stats(
    entry: float, forward_bars: list[dict[str, Any]], direction: str
) -> dict[str, float | None]:
    if entry <= 0 or not forward_bars:
        return {"favorable_excursion_pct": None, "adverse_excursion_pct": None}
    max_high = max(num(bar.get("high")) for bar in forward_bars)
    min_low = min(num(bar.get("low")) for bar in forward_bars)
    if direction == "PUTS":
        favorable = (entry - min_low) / entry
        adverse = (max_high - entry) / entry
    else:
        favorable = (max_high - entry) / entry
        adverse = (entry - min_low) / entry
    return {
        "favorable_excursion_pct": favorable,
        "adverse_excursion_pct": adverse,
    }


def volume_confirmation(
    bars: list[dict[str, Any]], index: int, lookback: int = 20
) -> dict[str, Any]:
    prior = [num(bar.get("volume")) for bar in bars[max(0, index - lookback) : index]]
    prior = [value for value in prior if value > 0]
    current_volume = num(bars[index].get("volume"))
    if not prior or current_volume <= 0:
        return {"volume_confirmation": "unknown", "relative_volume": None}
    rel = current_volume / median(prior)
    if rel >= 1.5:
        state = "confirmed"
    elif rel >= 1.2:
        state = "participating"
    elif rel <= 0.7:
        state = "thin"
    else:
        state = "mixed"
    return {"volume_confirmation": state, "relative_volume": rel}


def nearest_reference(
    bar: dict[str, Any], references: dict[str, float | None]
) -> dict[str, Any]:
    close = num(bar.get("close"))
    candidates = [
        (name, float(price))
        for name, price in references.items()
        if price is not None and float(price) > 0
    ]
    if close <= 0 or not candidates:
        return {
            "nearest_reference_name": "UNKNOWN",
            "nearest_reference_distance_pct": None,
            "nearest_reference_relation": "UNKNOWN",
        }
    name, price = min(candidates, key=lambda item: abs(close - item[1]))
    relation = "above" if close > price else "below" if close < price else "at"
    return {
        "nearest_reference_name": name,
        "nearest_reference_price": price,
        "nearest_reference_distance_pct": abs(close - price) / close,
        "nearest_reference_relation": relation,
    }


def acceptance_state(
    bars: list[dict[str, Any]], index: int, reference: float | None, window: int = 3
) -> str:
    if reference is None or reference <= 0:
        return "unknown"
    closes = [
        num(bar.get("close")) for bar in bars[max(0, index - window + 1) : index + 1]
    ]
    closes = [close for close in closes if close > 0]
    if len(closes) < window:
        return "insufficient"
    buffer = reference * 0.0003
    if all(close > reference + buffer for close in closes):
        return "accepted_above"
    if all(close < reference - buffer for close in closes):
        return "accepted_below"
    return "testing_reference"


def build_references(
    bars: list[dict[str, Any]], index: int, daily: dict[str, Any] | None
) -> dict[str, float | None]:
    refs: dict[str, float | None] = {
        "PDH": None if not daily else daily.get("prior_high"),
        "PDL": None if not daily else daily.get("prior_low"),
        "PDC": None if not daily else daily.get("prior_close"),
        "VWAP": bars[index].get("vwap"),
    }
    if index >= 1:
        opening = bars[:2]
        refs["ORH_30m"] = max(num(bar.get("high")) for bar in opening)
        refs["ORL_30m"] = min(num(bar.get("low")) for bar in opening)
    return refs


def build_event_rows_for_session(
    *,
    symbol: str,
    session_date: str,
    bars: list[dict[str, Any]],
    daily: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
    open_regime: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    horizon_bars: int = 4,
    target_pct: float = 0.0010,
    stop_pct: float = 0.0008,
    include_ordinary: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regime = regime or {}
    open_regime = open_regime or {}
    options = options or {}
    for index, bar in enumerate(bars):
        previous = bars[index - 1] if index else None
        event = classify_event(bar, previous)
        if (
            event["event_name"] in {"ordinary_range", "insufficient_bar"}
            and not include_ordinary
        ):
            continue
        forward = bars[index + 1 : index + 1 + horizon_bars]
        anatomy = candle_anatomy(bar)
        references = build_references(bars, index, daily)
        nearest = nearest_reference(bar, references)
        reference_price = nearest.get("nearest_reference_price")
        acceptance = acceptance_state(bars, index, reference_price)
        volume = volume_confirmation(bars, index)
        directional = first_touch_outcome(
            anatomy["close"], forward, event["event_direction"], target_pct, stop_pct
        )
        two_sided = two_sided_first_touch(anatomy["close"], forward, target_pct)
        excursions = excursion_stats(
            anatomy["close"], forward, event["event_direction"]
        )
        rows.append(
            {
                "symbol": symbol,
                "session_date": session_date,
                "ts": bar.get("ts"),
                "bar_index": index,
                "minutes_since_open": minutes_since_open(str(bar.get("ts"))),
                "event_name": event["event_name"],
                "event_direction": event["event_direction"],
                "open": anatomy["open"],
                "high": anatomy["high"],
                "low": anatomy["low"],
                "close": anatomy["close"],
                "volume": anatomy["volume"],
                "trade_count": anatomy["trade_count"],
                "vwap": anatomy["vwap"],
                "range_pct": anatomy["range_pct"],
                "body_pct": anatomy["body_pct"],
                "upper_wick_pct": anatomy["upper_wick_pct"],
                "lower_wick_pct": anatomy["lower_wick_pct"],
                **nearest,
                "acceptance_state": acceptance,
                **volume,
                "vol_state": regime.get("vol_state"),
                "macro_state": regime.get("macro_state"),
                "dp_state": regime.get("dp_state"),
                "regime_label": regime.get("regime_label"),
                "open_regime_label": open_regime.get("open_regime_label"),
                "setup_dir": open_regime.get("setup_dir"),
                "gamma_wall_strike": options.get("gamma_wall_strike"),
                "pcr_oi": options.get("pcr_oi"),
                "horizon_bars": horizon_bars,
                "target_pct": target_pct,
                "stop_pct": stop_pct,
                **directional,
                **two_sided,
                **excursions,
                "forward_bar_count": len(forward),
            }
        )
    return rows
