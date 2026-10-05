"""Larger multi-candle structure classifier for Candle Coach.

These are educational auction-shape labels, not execution signals. The goal is to
explain 5-20 bar structure without bloating the single/two/three candle logic.
"""

from __future__ import annotations

from typing import Any

STRUCTURE_PATTERN_LIBRARY = [
    "Bull flag / controlled pullback",
    "Bear flag / controlled bounce",
    "Ascending triangle",
    "Descending triangle",
    "Compression coil",
    "Stair-step advance",
    "Stair-step decline",
]


def _price(bar: dict[str, Any], key: str) -> float:
    try:
        return float(bar.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clock_label(minute: Any) -> str:
    if not isinstance(minute, (int, float)):
        return str(minute or "n/a")
    total = 570 + int(minute)
    return f"{total // 60:02d}:{total % 60:02d}"


def _attach(
    name: str,
    bias: str,
    meaning: str,
    watch_next: str,
    bars: list[dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "bias_hint": bias,
        "meaning": meaning,
        "watch_next": watch_next,
        "window": f"{len(bars)}-candle structure",
        "clock": f"{_clock_label(bars[0].get('minute'))}→{_clock_label(bars[-1].get('minute'))}",
        "candles": [
            {
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "direction": bar["direction"],
                "minute": bar["minute"],
            }
            for bar in bars
        ],
        "diagnostics": diagnostics or {},
    }


def _higher_lows(bars: list[dict[str, Any]], *, tolerance: float) -> bool:
    lows = [_price(bar, "low") for bar in bars]
    return all(curr >= prev - tolerance for prev, curr in zip(lows, lows[1:]))


def _lower_highs(bars: list[dict[str, Any]], *, tolerance: float) -> bool:
    highs = [_price(bar, "high") for bar in bars]
    return all(curr <= prev + tolerance for prev, curr in zip(highs, highs[1:]))


def _mostly_higher_closes(bars: list[dict[str, Any]]) -> bool:
    closes = [_price(bar, "close") for bar in bars]
    up_steps = sum(curr > prev for prev, curr in zip(closes, closes[1:]))
    return up_steps >= max(2, len(closes) - 2)


def _mostly_lower_closes(bars: list[dict[str, Any]]) -> bool:
    closes = [_price(bar, "close") for bar in bars]
    down_steps = sum(curr < prev for prev, curr in zip(closes, closes[1:]))
    return down_steps >= max(2, len(closes) - 2)


def classify_structure_pattern(
    bars: list[dict[str, Any]], context: str = ""
) -> dict[str, Any]:
    """Classify the latest larger candle structure from normalized bar packets."""
    window = bars[-12:] if len(bars) >= 12 else list(bars)
    if len(window) < 5:
        return {}

    closes = [_price(bar, "close") for bar in window]
    highs = [_price(bar, "high") for bar in window]
    lows = [_price(bar, "low") for bar in window]
    ranges = [max(_price(bar, "range"), 0.0) for bar in window]
    start = closes[0]
    end = closes[-1]
    span = max(max(highs) - min(lows), 1e-9)
    tolerance = max(end * 0.00035, 0.04)
    move_pct = (end / start - 1.0) * 100 if start else 0.0
    first_half = window[: max(3, len(window) // 2)]
    second_half = window[-max(3, len(window) // 2) :]
    first_move_pct = (
        (_price(first_half[-1], "close") / _price(first_half[0], "close") - 1.0) * 100
        if _price(first_half[0], "close")
        else 0.0
    )
    pullback_pct = (
        (_price(second_half[-1], "close") / _price(second_half[0], "close") - 1.0) * 100
        if _price(second_half[0], "close")
        else 0.0
    )
    compression = bool(
        len(ranges) >= 5
        and max(ranges[-3:]) < max(ranges[:3]) * 0.72
        and span / max(end, 1e-9) < 0.006
    )
    flat_top = max(highs[-5:]) - min(highs[-5:]) <= tolerance * 2.0
    flat_bottom = max(lows[-5:]) - min(lows[-5:]) <= tolerance * 2.0
    diagnostics = {
        "move_pct": round(move_pct, 3),
        "first_move_pct": round(first_move_pct, 3),
        "pullback_pct": round(pullback_pct, 3),
        "span_pct": round(span / max(end, 1e-9) * 100, 3),
        "context": context,
    }

    if (
        first_move_pct >= 0.18
        and -0.18 <= pullback_pct <= 0.03
        and _lower_highs(second_half, tolerance=tolerance)
    ):
        return _attach(
            "Bull flag / controlled pullback",
            "bullish continuation watch",
            "Price pushed up, then drifted sideways/down in a controlled pullback instead of fully rejecting the impulse.",
            "Needs breakout above the flag high with participation; losing the impulse midpoint turns it into failed continuation.",
            window,
            diagnostics,
        )
    if (
        first_move_pct <= -0.18
        and -0.03 <= pullback_pct <= 0.18
        and _higher_lows(second_half, tolerance=tolerance)
    ):
        return _attach(
            "Bear flag / controlled bounce",
            "bearish continuation watch",
            "Price sold off, then bounced in a controlled grind instead of reclaiming the prior impulse.",
            "Needs breakdown below the flag low with participation; reclaiming the impulse midpoint weakens the bear read.",
            window,
            diagnostics,
        )
    if flat_top and _higher_lows(window[-5:], tolerance=tolerance):
        return _attach(
            "Ascending triangle",
            "bullish pressure / resistance test",
            "Repeated highs near one ceiling while lows rise. Buyers are compressing into resistance.",
            "A close through the ceiling must hold/retest with volume; failure becomes trap risk.",
            window,
            diagnostics,
        )
    if flat_bottom and _lower_highs(window[-5:], tolerance=tolerance):
        return _attach(
            "Descending triangle",
            "bearish pressure / support test",
            "Repeated lows near one floor while highs fall. Sellers are compressing into support.",
            "A close through the floor must hold/retest with volume; reclaiming the floor becomes bear-trap risk.",
            window,
            diagnostics,
        )
    if compression:
        return _attach(
            "Compression coil",
            "expansion watch",
            "Recent candle ranges contracted inside a tight total span. Energy is coiling, but direction is unresolved.",
            "Do not guess inside the coil; wait for expansion through a boundary with participation and acceptance.",
            window,
            diagnostics,
        )
    if _higher_lows(window[-6:], tolerance=tolerance) and _mostly_higher_closes(
        window[-6:]
    ):
        return _attach(
            "Stair-step advance",
            "bullish continuation",
            "Higher lows and mostly rising closes show buyers walking the auction upward in steps.",
            "Continuation needs the next pullback to hold a higher low; losing two steps warns momentum is tired.",
            window[-6:],
            diagnostics,
        )
    if _lower_highs(window[-6:], tolerance=tolerance) and _mostly_lower_closes(
        window[-6:]
    ):
        return _attach(
            "Stair-step decline",
            "bearish continuation",
            "Lower highs and mostly falling closes show sellers walking the auction downward in steps.",
            "Continuation needs the next bounce to hold a lower high; reclaiming two steps warns sellers are losing control.",
            window[-6:],
            diagnostics,
        )

    return _attach(
        "No clean larger structure",
        "neutral",
        "The recent candle sequence does not form a clean larger pattern. That is useful: no forced narrative.",
        "Lean on levels, VWAP, acceptance, participation, and the permission spine instead of inventing structure.",
        window,
        diagnostics,
    )


__all__ = ["STRUCTURE_PATTERN_LIBRARY", "classify_structure_pattern"]
