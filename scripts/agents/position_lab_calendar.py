from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    dt_value = datetime.fromisoformat(candidate)
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC)


def _signal_date(signal: dict[str, Any]) -> date | None:
    dt_value = _parse_timestamp(str(signal.get("ts") or "")) or _parse_timestamp(
        str((signal.get("price_authority") or {}).get("display_time_utc") or "")
    )
    return dt_value.date() if dt_value else None


def _parse_expiration_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def list_symbol_expirations(
    snapshot_quotes: list[dict[str, Any]], symbol: str
) -> list[str]:
    return sorted(
        {
            str(quote.get("expiration"))
            for quote in snapshot_quotes
            if str(quote.get("underlying") or "").upper() == symbol
        }
    )


def pick_expiration(
    signal: dict[str, Any],
    snapshot_quotes: list[dict[str, Any]],
    symbol: str,
) -> tuple[str | None, str]:
    available = list_symbol_expirations(snapshot_quotes, symbol)
    if not available:
        return None, "no expirations available"

    signal_date = _signal_date(signal)
    if signal_date and len(available) > 1 and available[0] == signal_date.isoformat():
        return available[1], "preferred next expiry over same-day 0DTE"

    return available[0], "nearest available expiry"


def highlighted_expirations(
    signal: dict[str, Any], available: list[str]
) -> dict[str, str | None]:
    signal_date = _signal_date(signal)
    parsed = [exp_date for item in available if (exp_date := _parse_expiration_date(item))]
    same_day = signal_date.isoformat() if signal_date and signal_date.isoformat() in available else None
    forward = [exp_date for exp_date in parsed if signal_date is None or exp_date > signal_date]
    fridays = [exp_date for exp_date in forward if exp_date.weekday() == 4]
    weekly = fridays[0] if fridays else None
    next_weekly = next((exp_date for exp_date in fridays if weekly and exp_date > weekly), None)
    monthlyish_floor = signal_date + timedelta(days=28) if signal_date else None
    monthlyish = next(
        (exp_date for exp_date in forward if monthlyish_floor and exp_date >= monthlyish_floor),
        forward[-1] if forward else None,
    )
    return {
        "same_day": same_day,
        "next_expiration": forward[0].isoformat() if forward else None,
        "weekly_anchor": weekly.isoformat() if weekly else None,
        "next_weekly_anchor": next_weekly.isoformat() if next_weekly else None,
        "monthlyish_anchor": monthlyish.isoformat() if monthlyish else None,
    }


def build_calendar_context(
    signal: dict[str, Any], snapshot_quotes: list[dict[str, Any]], symbol: str
) -> dict[str, Any]:
    available = list_symbol_expirations(snapshot_quotes, symbol)
    selected, reason = pick_expiration(signal, snapshot_quotes, symbol)
    return {
        "selected_expiration": selected,
        "selection_reason": reason,
        "available_expirations": available,
        "highlighted_expirations": highlighted_expirations(signal, available),
    }


__all__ = [
    "build_calendar_context",
    "highlighted_expirations",
    "list_symbol_expirations",
    "pick_expiration",
]
