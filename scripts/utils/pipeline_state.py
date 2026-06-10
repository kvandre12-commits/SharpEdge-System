#!/usr/bin/env python3
"""Small helpers for pipeline health/state artifacts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

HEALTH_DIR = Path("outputs/health")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def iso_utc_now() -> str:
    return utc_now().isoformat()


def parse_date(value: Any) -> dt.date | None:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_datetime(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.UTC).replace(tzinfo=None)
    return parsed


def age_hours(value: Any, now: dt.datetime | None = None) -> float | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return ((now or utc_now()) - parsed).total_seconds() / 3600.0


def is_fresh(value: Any, ttl_hours: float, now: dt.datetime | None = None) -> bool:
    age = age_hours(value, now=now)
    return age is not None and 0 <= age < ttl_hours


def write_state(name: str, payload: dict[str, Any]) -> Path:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    path = HEALTH_DIR / f"{name}_state.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
