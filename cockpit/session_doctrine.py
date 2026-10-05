"""Shared session timing doctrine for SharpEdge."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

SESSION_OPEN_HOUR = 9
SESSION_OPEN_MINUTE = 30
REGULAR_SESSION_MINUTES = 390
OPENING_WINDOW_MINUTES = 30
MORNING_WINDOW_END = 120
MIDDAY_WINDOW_END = 240
POWER_HOUR_START = 330
OPENING_AUCTION_FULL_WEIGHT_UNTIL = 60
OPENING_AUCTION_MIDDAY_WEIGHT_UNTIL = 150
OPENING_AUCTION_MIDDAY_WEIGHT = 0.5
OPENING_AUCTION_LATE_WEIGHT = 0.2


def market_open_datetime(base: date | datetime | None = None) -> datetime:
    if isinstance(base, datetime):
        session_date = base.date()
        timezone = base.tzinfo
    else:
        session_date = base or datetime.now().astimezone().date()
        timezone = None
    return datetime.combine(
        session_date,
        time(SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE, tzinfo=timezone),
    )


def session_datetime_from_minute(
    minute: float,
    base: date | datetime | None = None,
) -> datetime:
    return market_open_datetime(base) + timedelta(minutes=float(minute))


def minutes_since_open(current_time: datetime) -> float:
    return (current_time - market_open_datetime(current_time)).total_seconds() / 60


def clock_label(current_time: datetime) -> str:
    return f"{current_time.hour}:{current_time.minute:02d}"


def classify_session_window(
    minutes_since_open_value: float,
    *,
    clock: str | None = None,
) -> dict[str, Any]:
    minutes = float(minutes_since_open_value)
    clock_text = str(clock or "n/a")
    if minutes < 0 or minutes > REGULAR_SESSION_MINUTES:
        return {
            "state": "closed_or_unknown",
            "reason": "outside_regular_hours",
            "detail": f"outside regular session at {clock_text}",
            "within_regular_session": False,
        }
    if minutes < OPENING_WINDOW_MINUTES:
        return {
            "state": "opening",
            "reason": "opening_auction",
            "detail": "opening auction: price discovery",
            "within_regular_session": True,
        }
    if minutes < MORNING_WINDOW_END:
        return {
            "state": "morning",
            "reason": "morning_continuation",
            "detail": "morning continuation window",
            "within_regular_session": True,
        }
    if minutes < MIDDAY_WINDOW_END:
        return {
            "state": "midday",
            "reason": "midday_chop",
            "detail": "midday chop window",
            "within_regular_session": True,
        }
    if minutes >= POWER_HOUR_START:
        return {
            "state": "power_hour",
            "reason": "power_hour_positioning",
            "detail": "power hour positioning window",
            "within_regular_session": True,
        }
    return {
        "state": "afternoon",
        "reason": "afternoon_rotation",
        "detail": f"neutral time window around {clock_text}",
        "within_regular_session": True,
    }


def opening_auction_decay_profile(
    minutes_since_open_value: float,
) -> dict[str, Any]:
    minutes = float(minutes_since_open_value)
    if minutes < OPENING_AUCTION_FULL_WEIGHT_UNTIL:
        return {"weight": 1.0, "label": "opening"}
    if minutes < OPENING_AUCTION_MIDDAY_WEIGHT_UNTIL:
        return {"weight": OPENING_AUCTION_MIDDAY_WEIGHT, "label": "midday"}
    return {"weight": OPENING_AUCTION_LATE_WEIGHT, "label": "late session"}


__all__ = [
    "MIDDAY_WINDOW_END",
    "MORNING_WINDOW_END",
    "OPENING_AUCTION_FULL_WEIGHT_UNTIL",
    "OPENING_AUCTION_LATE_WEIGHT",
    "OPENING_AUCTION_MIDDAY_WEIGHT",
    "OPENING_AUCTION_MIDDAY_WEIGHT_UNTIL",
    "OPENING_WINDOW_MINUTES",
    "POWER_HOUR_START",
    "REGULAR_SESSION_MINUTES",
    "SESSION_OPEN_HOUR",
    "SESSION_OPEN_MINUTE",
    "classify_session_window",
    "clock_label",
    "market_open_datetime",
    "minutes_since_open",
    "opening_auction_decay_profile",
    "session_datetime_from_minute",
]
