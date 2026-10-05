"""Historical refill-surface diagnostics for the live cockpit.

This module turns the offline ``sharpedge.historical_refill_surface.v1`` artifact
into a live, diagnostic-only card. It does not alter execution permission,
setup selection, or approval authority.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_SURFACE_PATH = "~/SharpEdge-System/outputs/historical_refill_surface.json"
DEFAULT_STACK_SURFACE_PATH = (
    "~/SharpEdge-System/outputs/historical_refill_stack_surface.json"
)
MODES = ("gap_down_open", "intraday_dip")
EMA_PERIOD = 200
NEAR_EMA_PCT = 0.5
EMA_DISTANCE_BUCKETS = (
    (-999.0, -5.0, "below_5pct_plus"),
    (-5.0, -2.0, "below_2_to_5pct"),
    (-2.0, -0.5, "below_0_5_to_2pct"),
    (-0.5, 0.5, "near_ema200"),
    (0.5, 2.0, "above_0_5_to_2pct"),
    (2.0, 5.0, "above_2_to_5pct"),
    (5.0, 999.0, "above_5pct_plus"),
)


def _load_json_artifact(
    path: str | os.PathLike[str] | None,
    default_path: str,
    label: str,
) -> dict[str, Any]:
    surface_path = Path(os.path.expanduser(str(path or default_path)))
    if not surface_path.exists():
        return {
            "available": False,
            "reason": f"{label} missing: {surface_path}",
            "source": str(surface_path),
        }
    try:
        return json.loads(surface_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "reason": f"{label} unreadable: {exc}",
            "source": str(surface_path),
        }


def _load_surface(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    return _load_json_artifact(path, DEFAULT_SURFACE_PATH, "historical refill surface")


def _load_stack_surface(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    return _load_json_artifact(
        path,
        DEFAULT_STACK_SURFACE_PATH,
        "historical refill stack surface",
    )


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _ema_series(values: list[float], period: int = EMA_PERIOD) -> list[float | None]:
    ema_values: list[float | None] = [None] * len(values)
    if len(values) < period:
        return ema_values
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period
    ema_values[period - 1] = ema
    for index in range(period, len(values)):
        ema = values[index] * multiplier + ema * (1 - multiplier)
        ema_values[index] = ema
    return ema_values


def _ema200_context(price: float, ema200: float | None) -> dict[str, Any]:
    if ema200 is None or ema200 <= 0:
        return {
            "ema200": None,
            "distance_pct": None,
            "side": "unknown",
            "distance_bucket": "unknown",
        }
    distance_pct = (price / ema200 - 1.0) * 100.0
    if abs(distance_pct) <= NEAR_EMA_PCT:
        side = "near_ema200"
    elif distance_pct > 0:
        side = "above_ema200"
    else:
        side = "below_ema200"
    bucket = "unknown"
    for low, high, label in EMA_DISTANCE_BUCKETS:
        if low <= distance_pct < high:
            bucket = label
            break
    return {
        "ema200": ema200,
        "distance_pct": distance_pct,
        "side": side,
        "distance_bucket": bucket,
    }


def _event_move(mode: str, bar: dict[str, Any], prior_close: float) -> float | None:
    open_ = _as_float(bar.get("open"))
    low = _as_float(bar.get("low"))
    if prior_close <= 0:
        return None
    if mode == "gap_down_open" and open_ is not None:
        return (open_ / prior_close) - 1.0
    if mode == "intraday_dip" and low is not None:
        return (low / prior_close) - 1.0
    return None


def _known_thresholds(surface: dict[str, Any]) -> list[float]:
    thresholds = {
        round(float(row["threshold_pct"]) / 100.0, 6)
        for row in surface.get("rows", [])
        if row.get("threshold_pct") is not None
    }
    return sorted(thresholds)


def _bucket_threshold(move_pct: float, thresholds: list[float]) -> float | None:
    magnitude = abs(move_pct)
    index = bisect_right(thresholds, magnitude) - 1
    if index < 0:
        return None
    return thresholds[index]


def _surface_row(
    surface: dict[str, Any], mode: str, threshold: float
) -> dict[str, Any] | None:
    threshold_pct = round(threshold * 100.0, 6)
    for row in surface.get("rows", []):
        row_threshold = _as_float(row.get("threshold_pct"))
        if row.get("mode") == mode and row_threshold is not None:
            if abs(row_threshold - threshold_pct) < 0.001:
                return row
    return None


def _find_fill_index(
    bars: list[dict[str, Any]], start_index: int, target: float
) -> int | None:
    for index in range(start_index, len(bars)):
        high = _as_float(bars[index].get("high"))
        if high is not None and high >= target:
            return index
    return None


def _active_events(
    daily_bars: list[dict[str, Any]],
    surface: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = _known_thresholds(surface)
    if len(daily_bars) < 2 or not thresholds:
        return []

    closes = [_as_float(bar.get("close")) for bar in daily_bars]
    ema200_values = _ema_series([value or 0.0 for value in closes])
    active_events: list[dict[str, Any]] = []
    for index in range(1, len(daily_bars)):
        bar = daily_bars[index]
        prior = daily_bars[index - 1]
        prior_close = _as_float(prior.get("close"))
        if prior_close is None or prior_close <= 0:
            continue
        for mode in MODES:
            move_pct = _event_move(mode, bar, prior_close)
            if move_pct is None or move_pct >= 0:
                continue
            threshold = _bucket_threshold(move_pct, thresholds)
            if threshold is None:
                continue
            fill_index = _find_fill_index(daily_bars, index, prior_close)
            if fill_index is None:
                ema_context = _ema200_context(prior_close, ema200_values[index - 1])
                active_events.append(
                    {
                        "mode": mode,
                        "event_index": index,
                        "event_date": bar.get("date"),
                        "prior_close": prior_close,
                        "target": prior_close,
                        "threshold": threshold,
                        "threshold_pct": threshold * 100.0,
                        "move_pct": move_pct,
                        "open": _as_float(bar.get("open")),
                        "low": _as_float(bar.get("low")),
                        "ema200": ema_context["ema200"],
                        "ema200_distance_pct": ema_context["distance_pct"],
                        "ema200_side": ema_context["side"],
                        "ema200_distance_bucket": ema_context["distance_bucket"],
                    }
                )

    return sorted(active_events, key=lambda event: event["event_index"])


def _fmt_pct(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.1f}%"


def _nearest_fill_horizon(row: dict[str, Any], days: float | None) -> dict[str, Any]:
    if days is None:
        return {"window_days": None, "count": None, "rate_pct": None}
    for horizon in (0, 1, 3, 5, 10, 20, 60, 120):
        if days <= horizon:
            return {
                "window_days": horizon,
                "count": _as_float(row.get(f"fill_within_{horizon}d_count")),
                "rate_pct": _as_float(row.get(f"fill_within_{horizon}d_rate_pct")),
            }
    return {"window_days": None, "count": None, "rate_pct": None}


def _mode_frequency(row: dict[str, Any], mode_days: float | None) -> dict[str, Any]:
    count = _as_float(row.get("fill_mode_count"))
    rate = _as_float(row.get("fill_mode_rate_pct"))
    event_count = _as_float(row.get("event_count"))
    if count is None and mode_days == 0:
        count = _as_float(row.get("fill_within_0d_count"))
    if rate is None and count is not None and event_count:
        rate = count / event_count * 100.0
    return {
        "mode_count": int(count) if count is not None else None,
        "mode_rate_pct": rate,
    }


def _matching_ema200_stats(
    row: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    context = row.get("ema200_context") or {}
    sides = context.get("sides") or {}
    buckets = context.get("distance_buckets") or {}
    side = event.get("ema200_side") or "unknown"
    bucket = event.get("ema200_distance_bucket") or "unknown"
    return {
        "side_stats": sides.get(side),
        "distance_bucket_stats": buckets.get(bucket),
        "all_side_stats": sides,
        "all_distance_bucket_stats": buckets,
        "basis": context.get("basis"),
    }


def _estimate_status(row: dict[str, Any], elapsed: int) -> dict[str, Any]:
    median_days = _as_float(row.get("fill_median_trading_days"))
    mean_days = _as_float(row.get("fill_mean_trading_days"))
    max_days = _as_float(row.get("fill_max_trading_days"))
    mode_days = _as_float(row.get("fill_mode_trading_days"))
    horizon_20 = _as_float(row.get("fill_within_20d_rate_pct"))
    horizon_60 = _as_float(row.get("fill_within_60d_rate_pct"))
    mode_frequency = _mode_frequency(row, mode_days)
    median_horizon = _nearest_fill_horizon(row, median_days)
    mean_horizon = _nearest_fill_horizon(row, mean_days)

    if median_days is None:
        phase = "unknown"
        story = "No historical median available for this bucket."
    elif elapsed <= median_days:
        phase = "inside_median_window"
        story = (
            f"Inside historical median refill window ({median_days:g} trading days)."
        )
    elif max_days is not None and elapsed <= max_days:
        phase = "late_but_inside_observed_window"
        story = (
            f"Past median ({median_days:g}d) but inside observed max "
            f"({max_days:g}d). This is the slow/refusal zone."
        )
    else:
        phase = "beyond_observed_window"
        story = (
            f"Beyond observed max ({max_days:g}d); treat as regime change, not normal refill."
            if max_days is not None
            else "Past typical refill window; no observed max available."
        )

    return {
        "phase": phase,
        "story": story,
        "median_trading_days": median_days,
        "mean_trading_days": mean_days,
        "median_horizon_days": median_horizon["window_days"],
        "median_horizon_count": median_horizon["count"],
        "median_horizon_rate_pct": median_horizon["rate_pct"],
        "mean_horizon_days": mean_horizon["window_days"],
        "mean_horizon_count": mean_horizon["count"],
        "mean_horizon_rate_pct": mean_horizon["rate_pct"],
        "mode_trading_days": mode_days,
        "mode_count": mode_frequency["mode_count"],
        "mode_rate_pct": mode_frequency["mode_rate_pct"],
        "max_trading_days": max_days,
        "fill_within_20d_rate_pct": horizon_20,
        "fill_within_60d_rate_pct": horizon_60,
    }


def _stack_label(depth: int) -> str:
    if depth <= 1:
        return "single_active_dip"
    if depth == 2:
        return "double_dip_stack"
    if depth == 3:
        return "triple_dip_stack"
    return "multi_dip_stack"


def _event_stack_item(
    event: dict[str, Any],
    row: dict[str, Any] | None,
    latest_index: int,
) -> dict[str, Any]:
    elapsed = latest_index - int(event["event_index"])
    estimated = _estimate_status(row, elapsed) if row else {}
    return {
        "mode": event["mode"],
        "event_date": event["event_date"],
        "threshold_pct": round(event["threshold_pct"], 2),
        "move_pct": round(event["move_pct"] * 100.0, 2),
        "target": round(event["target"], 2),
        "elapsed_trading_days": elapsed,
        "phase": estimated.get("phase"),
        "median_trading_days": estimated.get("median_trading_days"),
        "fill_rate_pct": row.get("fill_rate_pct") if row else None,
        "event_count": row.get("event_count") if row else None,
        "ema200_side": event.get("ema200_side"),
        "ema200_distance_bucket": event.get("ema200_distance_bucket"),
    }


def _threshold_label(value: Any) -> str:
    number = _as_float(value)
    return str(value) if number is None else f"{number:g}"


def _active_stack_context(
    events: list[dict[str, Any]],
    surface: dict[str, Any],
    latest_index: int,
) -> dict[str, Any]:
    items = [
        _event_stack_item(
            event,
            _surface_row(surface, event["mode"], event["threshold"]),
            latest_index,
        )
        for event in events
    ]
    targets = [item["target"] for item in items]
    thresholds = [item["threshold_pct"] for item in items]
    return {
        "active_count": len(items),
        "stack_label": _stack_label(len(items)),
        "oldest_event_date": items[0]["event_date"] if items else None,
        "latest_event_date": items[-1]["event_date"] if items else None,
        "highest_target": max(targets) if targets else None,
        "nearest_target": min(targets) if targets else None,
        "max_threshold_pct": max(thresholds) if thresholds else None,
        "items": items,
        "interaction_signature": "+".join(
            f"{item['mode']}:{_threshold_label(item['threshold_pct'])}:{item.get('ema200_distance_bucket')}"
            for item in items
        ),
        "next_research": (
            "Backtest stack signatures to learn double/triple dip refill behavior "
            "and interactions across EMA200, gap, auction, gamma, and VWAP surfaces."
        ),
    }


def _stack_history_stats(
    stack_surface: dict[str, Any],
    active_stack: dict[str, Any],
) -> dict[str, Any] | None:
    if not stack_surface.get("schema"):
        return None
    items = active_stack.get("items") or []
    latest = items[-1] if items else {}
    previous = items[-2] if len(items) >= 2 else {}
    signature = active_stack.get("interaction_signature") or ""
    pair_bucket = (
        f"{previous.get('ema200_distance_bucket')} -> "
        f"{latest.get('ema200_distance_bucket')}"
        if previous and latest
        else ""
    )
    pair_side = (
        f"{previous.get('ema200_side')} -> {latest.get('ema200_side')}"
        if previous and latest
        else ""
    )
    return {
        "schema": stack_surface.get("schema"),
        "generated_at": stack_surface.get("generated_at"),
        "basis": stack_surface.get("basis"),
        "latest_event_date_excluded_from_stats": stack_surface.get(
            "latest_event_date_excluded_from_stats"
        ),
        "overall": stack_surface.get("overall"),
        "exact_signature_stats": (stack_surface.get("exact_signatures") or {}).get(
            signature
        ),
        "last_pair_distance_bucket_stats": (
            stack_surface.get("by_last_pair_distance_bucket") or {}
        ).get(pair_bucket),
        "last_pair_side_stats": (stack_surface.get("by_last_pair_side") or {}).get(
            pair_side
        ),
        "new_ema_side_stats": (stack_surface.get("by_new_ema_side") or {}).get(
            latest.get("ema200_side")
        ),
        "new_ema_distance_bucket_stats": (
            stack_surface.get("by_new_ema_distance_bucket") or {}
        ).get(latest.get("ema200_distance_bucket")),
    }


def build_historical_refill_context(
    daily_bars: list[dict[str, Any]],
    *,
    spot: float | None = None,
    surface_path: str | os.PathLike[str] | None = None,
    stack_surface_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a cockpit-ready active refill-window diagnostic packet."""
    surface = _load_surface(surface_path)
    stack_surface = _load_stack_surface(stack_surface_path)
    schema = "sharpedge.historical_refill_context.v1"
    if not surface.get("schema"):
        return {
            "schema": schema,
            "available": False,
            "reason": surface.get("reason") or "historical refill surface unavailable",
            "authority": "diagnostic_only",
            "source": surface.get("source") or DEFAULT_SURFACE_PATH,
        }

    events = _active_events(daily_bars, surface)
    if not events:
        return {
            "schema": schema,
            "available": False,
            "reason": "no active unfilled down-gap/dip window in loaded daily bars",
            "authority": "diagnostic_only",
            "surface_schema": surface.get("schema"),
            "surface_generated_at": surface.get("generated_at"),
        }

    event = events[-1]
    row = _surface_row(surface, event["mode"], event["threshold"])
    if not row:
        return {
            "schema": schema,
            "available": False,
            "reason": "active window found but matching surface bucket is missing",
            "authority": "diagnostic_only",
            "active_event": event,
        }

    latest_index = len(daily_bars) - 1
    elapsed = latest_index - int(event["event_index"])
    active_stack = _active_stack_context(events, surface, latest_index)
    stack_history = _stack_history_stats(stack_surface, active_stack)
    if stack_history:
        active_stack["historical_stack_surface"] = stack_history
    estimated = _estimate_status(row, elapsed)
    ema200_stats = _matching_ema200_stats(row, event)
    current_spot = spot if spot is not None else _as_float(daily_bars[-1].get("close"))
    remaining_points = (
        event["target"] - current_spot
        if isinstance(current_spot, (int, float))
        else None
    )
    event_dt = _as_date(event["event_date"])
    last_dt = _as_date(daily_bars[-1].get("date"))
    calendar_days_elapsed = (last_dt - event_dt).days if event_dt and last_dt else None

    headline = (
        f"ACTIVE {_fmt_pct(event['threshold_pct'])} {event['mode'].replace('_', ' ').upper()} "
        f"REFILL WINDOW → target ${event['target']:.2f}"
    )
    story = (
        f"Event {event['event_date']}: move {event['move_pct'] * 100:.2f}% from prior close. "
        f"Elapsed {elapsed} trading days; {estimated['story']}"
    )
    return {
        "schema": schema,
        "available": True,
        "authority": "diagnostic_only",
        "surface_schema": surface.get("schema"),
        "surface_generated_at": surface.get("generated_at"),
        "mode": event["mode"],
        "threshold_pct": round(event["threshold_pct"], 2),
        "event_date": event["event_date"],
        "active_refill_stack": active_stack,
        "gap_fill_target": round(event["target"], 2),
        "current_spot": round(current_spot, 2)
        if isinstance(current_spot, (int, float))
        else None,
        "remaining_points_to_fill": round(remaining_points, 2)
        if isinstance(remaining_points, (int, float))
        else None,
        "move_pct": round(event["move_pct"] * 100.0, 2),
        "ema200_context": {
            "ema200": round(event["ema200"], 2)
            if isinstance(event.get("ema200"), (int, float))
            else None,
            "distance_pct": round(event["ema200_distance_pct"], 2)
            if isinstance(event.get("ema200_distance_pct"), (int, float))
            else None,
            "side": event.get("ema200_side"),
            "distance_bucket": event.get("ema200_distance_bucket"),
            **ema200_stats,
        },
        "elapsed_trading_days": elapsed,
        "elapsed_calendar_days": calendar_days_elapsed,
        "event_count": row.get("event_count"),
        "event_frequency_pct": row.get("event_frequency_pct"),
        "fill_rate_pct": row.get("fill_rate_pct"),
        "estimated": estimated,
        "headline": headline,
        "story": story,
        "caveat": "Diagnostic historical context only; does not override final execution permission.",
    }


__all__ = ["build_historical_refill_context"]
