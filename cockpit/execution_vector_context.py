from __future__ import annotations

from datetime import datetime

from level_state_engine import LEVEL_STATE_LEVEL_NAMES
from session_doctrine import session_datetime_from_minute


def bind_engine_context(
    engine,
    bars,
    pa,
    levels,
    setups=None,
    op=None,
    gp=None,
    magnitude=None,
    volatility_structure=None,
):
    engine.bars = bars
    engine.pa = pa or {}
    engine.levels = levels or {}
    engine.setups = setups or []
    engine.op = op or {}
    engine.gp = gp or {}
    engine.magnitude = magnitude or {}
    engine.volatility_structure = volatility_structure
    engine.full_levels = level_map(engine.pa, engine.levels, engine.op)
    engine.acceptance_levels = acceptance_level_map(engine.pa, engine.levels, engine.op)
    engine.location_references = location_reference_map(
        engine.pa,
        engine.levels,
        engine.op,
    )


def last_minute(bars):
    return int(bars[-1][0]) if bars else 0


def session_datetime(bars):
    return session_datetime_from_minute(last_minute(bars), datetime.today())


def nearest_level(spot, levels):
    clean = {name: value for name, value in levels.items() if value is not None}
    if not clean or not spot:
        return None, None, None
    name, value = min(clean.items(), key=lambda item: abs(item[1] - spot))
    return name, value, abs(value - spot) / spot * 100


def level_map(pa, levels, op):
    mapped = dict(levels or {})
    if pa.get("vwap"):
        mapped["VWAP"] = pa["vwap"]
    if op.get("call_wall"):
        mapped["CALL_WALL"] = op["call_wall"]
    if op.get("put_wall"):
        mapped["PUT_WALL"] = op["put_wall"]
    return mapped


def acceptance_level_map(pa, levels, op):
    full = level_map(pa, levels, op)
    return {
        name: full[name]
        for name in LEVEL_STATE_LEVEL_NAMES
        if name in full and full[name] is not None
    }


def location_reference_map(pa, levels, op):
    full = level_map(pa, levels, op)
    allowed = {*LEVEL_STATE_LEVEL_NAMES, "VWAP"}
    return {
        name: full[name] for name in allowed if name in full and full[name] is not None
    }


def recent_closes(bars, n=3):
    return [bar[4] for bar in bars[-n:]]


def ema(values, length=20):
    if not values:
        return 0.0
    alpha = 2 / (length + 1)
    value = values[0]
    for item in values[1:]:
        value = item * alpha + value * (1 - alpha)
    return value


def swing_points(bars, window=2):
    if len(bars) < window * 2 + 3:
        return [], []
    highs = []
    lows = []
    for idx in range(window, len(bars) - window):
        high = bars[idx][2]
        low = bars[idx][3]
        left = bars[idx - window : idx]
        right = bars[idx + 1 : idx + window + 1]
        if high >= max(bar[2] for bar in left + right):
            highs.append((idx, high))
        if low <= min(bar[3] for bar in left + right):
            lows.append((idx, low))
    return highs, lows


__all__ = [
    "bind_engine_context",
    "acceptance_level_map",
    "ema",
    "location_reference_map",
    "last_minute",
    "level_map",
    "nearest_level",
    "recent_closes",
    "session_datetime",
    "swing_points",
]
