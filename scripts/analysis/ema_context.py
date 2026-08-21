from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

EMA_PERIOD = 200
NEAR_EMA_PCT = 0.5
DISTANCE_BUCKETS = (
    (-999.0, -5.0, "below_5pct_plus"),
    (-5.0, -2.0, "below_2_to_5pct"),
    (-2.0, -0.5, "below_0_5_to_2pct"),
    (-0.5, 0.5, "near_ema200"),
    (0.5, 2.0, "above_0_5_to_2pct"),
    (2.0, 5.0, "above_2_to_5pct"),
    (5.0, 999.0, "above_5pct_plus"),
)


def ema_series(values: list[float], period: int = EMA_PERIOD) -> list[float | None]:
    """Return an EMA series seeded by the first full-window SMA."""
    if period <= 0:
        raise ValueError("period must be positive")
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


def classify_ema200_context(price: float, ema200: float | None) -> dict[str, Any]:
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
    for low, high, label in DISTANCE_BUCKETS:
        if low <= distance_pct < high:
            bucket = label
            break

    return {
        "ema200": ema200,
        "distance_pct": distance_pct,
        "side": side,
        "distance_bucket": bucket,
    }


def _event_value(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _fill_days(events: list[Any]) -> list[int]:
    days: list[int] = []
    for event in events:
        if not _event_value(event, "filled"):
            continue
        value = _event_value(event, "trading_days_to_fill")
        if value is not None:
            days.append(int(value))
    return days


def _mode_days(days: list[int]) -> tuple[int | None, int, float | None]:
    if not days:
        return None, 0, None
    counts = Counter(days)
    mode, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return mode, count, count / len(days) * 100.0


def summarize_ema200_context(events: list[Any]) -> dict[str, Any]:
    total = len(events)
    groups: dict[str, dict[str, Any]] = {}
    for key in ("ema200_side", "ema200_distance_bucket"):
        labels = sorted(
            {str(_event_value(event, key) or "unknown") for event in events}
        )
        groups[key] = {
            label: _summarize_group(
                [
                    event
                    for event in events
                    if str(_event_value(event, key) or "unknown") == label
                ],
                total,
            )
            for label in labels
        }
    return {
        "schema": "sharpedge.ema200_context.v1",
        "basis": "prior close vs prior-session EMA200 at event time; no look-ahead",
        "total_events": total,
        "sides": groups["ema200_side"],
        "distance_buckets": groups["ema200_distance_bucket"],
    }


def _summarize_group(events: list[Any], total: int) -> dict[str, Any]:
    count = len(events)
    filled = [event for event in events if _event_value(event, "filled")]
    days = _fill_days(events)
    mode, mode_count, mode_rate = _mode_days(days)
    return {
        "event_count": count,
        "event_share_pct": count / total * 100.0 if total else None,
        "fill_rate_pct": len(filled) / count * 100.0 if count else None,
        "median_trading_days": median(days) if days else None,
        "mode_trading_days": mode,
        "mode_count": mode_count,
        "mode_rate_pct": mode_rate,
    }
