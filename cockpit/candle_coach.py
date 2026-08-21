"""Educational candlestick-pattern coach for the live cockpit.

This module is deliberately advisory-only. It teaches bar anatomy and common
single/two-candle configurations from the latest OHLC rows; it must never act as
execution authority or override the SharpEdge permission spine.
"""

from __future__ import annotations

from typing import Any

from candle_encyclopedia import encyclopedia_packet, pattern_names
from candle_expectancy_adapter import (
    expectancy_gate_from_lookup,
    lookup_candle_expectancy,
)
from candle_framework import build_candle_framework
from candle_structures import STRUCTURE_PATTERN_LIBRARY, classify_structure_pattern
from candle_vector_teacher import build_candle_vector_lesson

CANDLE_COACH_SCHEMA = "sharpedge.candle_coach.v1"

PATTERN_LIBRARY = [*pattern_names(), *STRUCTURE_PATTERN_LIBRARY]


def _bar_packet(row: tuple[Any, ...] | list[Any], index: int) -> dict[str, Any]:
    minute, open_, high, low, close, volume = row[:6]
    open_f = float(open_)
    high_f = float(high)
    low_f = float(low)
    close_f = float(close)
    raw_range = max(high_f - low_f, 0.0)
    candle_range = max(raw_range, 1e-9)
    body = abs(close_f - open_f)
    upper_wick = high_f - max(open_f, close_f)
    lower_wick = min(open_f, close_f) - low_f
    direction = "bull" if close_f > open_f else "bear" if close_f < open_f else "flat"
    return {
        "index": index,
        "minute": minute,
        "open": round(open_f, 4),
        "high": round(high_f, 4),
        "low": round(low_f, 4),
        "close": round(close_f, 4),
        "volume": int(volume or 0),
        "direction": direction,
        "range": round(raw_range, 4),
        "body": round(body, 4),
        "body_pct": round(body / candle_range, 3),
        "upper_wick_pct": round(upper_wick / candle_range, 3),
        "lower_wick_pct": round(lower_wick / candle_range, 3),
    }


def _clock_label(minute: Any) -> str:
    if not isinstance(minute, (int, float)):
        return str(minute or "n/a")
    total = 570 + int(minute)
    hour = total // 60
    mins = total % 60
    return f"{hour:02d}:{mins:02d}"


def _trend_context(bars: list[dict[str, Any]]) -> str:
    if len(bars) < 4:
        return "too little context"
    first = float(bars[0]["close"])
    last = float(bars[-1]["close"])
    move_pct = (last / first - 1) * 100 if first else 0.0
    if move_pct > 0.08:
        return "short-term push up"
    if move_pct < -0.08:
        return "short-term push down"
    return "short-term chop / balance"


def _base_pattern(
    name: str, bias: str, meaning: str, watch_next: str
) -> dict[str, str]:
    return {
        "name": name,
        "bias_hint": bias,
        "meaning": meaning,
        "watch_next": watch_next,
    }


def _attach_candles(
    pattern: dict[str, Any], candles: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        **pattern,
        "candles": [
            {
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "direction": candle["direction"],
                "minute": candle["minute"],
            }
            for candle in candles
        ],
    }


def _is_push_down(context: str) -> bool:
    return "down" in str(context).lower()


def _is_push_up(context: str) -> bool:
    return "up" in str(context).lower()


def classify_single_candle(bar: dict[str, Any], context: str = "") -> dict[str, Any]:
    body_pct = float(bar["body_pct"])
    upper = float(bar["upper_wick_pct"])
    lower = float(bar["lower_wick_pct"])
    direction = str(bar["direction"])

    if float(bar.get("range") or 0) <= 0 or int(bar.get("volume") or 0) <= 0:
        pattern = _base_pattern(
            "Insufficient bar information",
            "no directional inference",
            "Zero-range or non-participating bar. The candle shape does not prove buyers and sellers fought; it may contain no useful auction information.",
            "Verify actual volume, trade count, quote updates, spread, and feed continuity before interpreting the bar.",
        )
    elif body_pct <= 0.10 and lower >= 0.55:
        pattern = _base_pattern(
            "Dragonfly doji",
            "bullish reversal attempt",
            "A doji with a long lower tail: sellers drove price down, but buyers reclaimed the close.",
            "Confirmation is a close above the dragonfly high; below the low means sellers took it back.",
        )
    elif body_pct <= 0.10 and upper >= 0.55:
        pattern = _base_pattern(
            "Gravestone doji",
            "bearish reversal attempt",
            "A doji with a long upper tail: buyers drove price up, but sellers rejected the close.",
            "Confirmation is a close below the gravestone low; above the high means buyers reclaimed it.",
        )
    elif lower >= 0.55 and upper <= 0.20 and body_pct <= 0.35:
        pattern = _base_pattern(
            "Hanging man" if _is_push_up(context) else "Hammer / demand tail",
            "bearish warning" if _is_push_up(context) else "bullish reversal attempt",
            "A long lower wick means sellers tested down and buyers reclaimed; after an up-push it can warn of distribution.",
            "Need the next candle: close above the high favors demand, close below the low confirms supply pressure.",
        )
    elif upper >= 0.55 and lower <= 0.20 and body_pct <= 0.35:
        pattern = _base_pattern(
            "Inverted hammer"
            if _is_push_down(context)
            else "Shooting star / supply tail",
            "bullish reversal attempt"
            if _is_push_down(context)
            else "bearish reversal attempt",
            "A long upper wick means price probed higher and got rejected; after a selloff it can be early demand testing supply.",
            "Need follow-through: reclaim the high for bullish confirmation, lose the low for bearish confirmation.",
        )
    elif body_pct <= 0.10:
        pattern = _base_pattern(
            "Doji",
            "indecision",
            "Open and close are nearly equal during the interval. That marks contraction/indecision only; it is not a directional thesis.",
            "Wait for the next candle to break the doji high/low before trusting direction.",
        )
    elif body_pct >= 0.75 and direction == "bull" and lower <= 0.08:
        pattern = _base_pattern(
            "Bullish belt hold",
            "bullish continuation",
            "A strong green candle with almost no lower wick. Buyers took control from the open.",
            "Watch whether the next candle accepts above its close or immediately fades it.",
        )
    elif body_pct >= 0.75 and direction == "bear" and upper <= 0.08:
        pattern = _base_pattern(
            "Bearish belt hold",
            "bearish continuation",
            "A strong red candle with almost no upper wick. Sellers took control from the open.",
            "Watch whether the next candle accepts below its close or immediately reclaims it.",
        )
    elif body_pct >= 0.75 and direction == "bull":
        pattern = _base_pattern(
            "Bullish marubozu / conviction candle",
            "bullish continuation",
            "Most of the candle is body. Buyers controlled nearly the whole bar.",
            "Watch whether the next candle accepts above its close or immediately fades it.",
        )
    elif body_pct >= 0.75 and direction == "bear":
        pattern = _base_pattern(
            "Bearish marubozu / conviction candle",
            "bearish continuation",
            "Most of the candle is body. Sellers controlled nearly the whole bar.",
            "Watch whether the next candle accepts below its close or immediately reclaims it.",
        )
    elif body_pct <= 0.35 and upper >= 0.25 and lower >= 0.25:
        pattern = _base_pattern(
            "Spinning top",
            "indecision",
            "Small body with wicks on both sides. Both sides probed, neither owned the close.",
            "Treat it as a pause candle; direction needs a break of either wick extreme.",
        )
    elif body_pct >= 0.55 and direction == "bull":
        pattern = _base_pattern(
            "Strong bullish candle",
            "bullish pressure",
            "A large green body shows buyers controlled the close for this bar.",
            "Continuation needs the next bar to hold above the midpoint or high-volume follow-through.",
        )
    elif body_pct >= 0.55 and direction == "bear":
        pattern = _base_pattern(
            "Strong bearish candle",
            "bearish pressure",
            "A large red body shows sellers controlled the close for this bar.",
            "Continuation needs the next bar to hold below the midpoint or high-volume follow-through.",
        )
    else:
        pattern = _base_pattern(
            "Ordinary range candle",
            "neutral",
            "No standout wick/body pattern. This bar is mostly context, not a headline.",
            "Look to VWAP, levels, volume, and the next candle for the real clue.",
        )

    return _attach_candles(
        {
            **pattern,
            "window": "1-candle",
            "clock": _clock_label(bar.get("minute")),
            "context": context,
            "anatomy": {
                "body_pct": bar["body_pct"],
                "upper_wick_pct": bar["upper_wick_pct"],
                "lower_wick_pct": bar["lower_wick_pct"],
                "direction": direction,
            },
        },
        [bar],
    )


def _body_bounds(bar: dict[str, Any]) -> tuple[float, float]:
    return min(float(bar["open"]), float(bar["close"])), max(
        float(bar["open"]), float(bar["close"])
    )


def classify_two_candle(
    previous: dict[str, Any], current: dict[str, Any], context: str = ""
) -> dict[str, Any]:
    prev_low, prev_high = _body_bounds(previous)
    cur_low, cur_high = _body_bounds(current)
    prev_dir = str(previous["direction"])
    cur_dir = str(current["direction"])
    tolerance = max(float(current["close"]) * 0.00015, 0.03)
    prev_mid = (float(previous["open"]) + float(previous["close"])) / 2

    if (
        prev_dir == "bear"
        and cur_dir == "bull"
        and cur_low <= prev_low
        and cur_high >= prev_high
    ):
        pattern = _base_pattern(
            "Bullish engulfing",
            "bullish reversal / reclaim attempt",
            "The current green body fully swallowed the prior red body. Buyers reversed the prior auction.",
            "Best confirmation is follow-through above the engulfing high, especially near support/VWAP.",
        )
    elif (
        prev_dir == "bull"
        and cur_dir == "bear"
        and cur_low <= prev_low
        and cur_high >= prev_high
    ):
        pattern = _base_pattern(
            "Bearish engulfing",
            "bearish reversal / rejection attempt",
            "The current red body fully swallowed the prior green body. Sellers reversed the prior auction.",
            "Best confirmation is follow-through below the engulfing low, especially near resistance/VWAP.",
        )
    elif (
        prev_dir == "bear"
        and cur_dir == "bull"
        and float(current["close"]) > prev_mid
        and float(current["close"]) < float(previous["open"])
    ):
        pattern = _base_pattern(
            "Piercing line",
            "bullish reversal attempt",
            "After a red candle, the green candle reclaimed more than half of the prior body without fully engulfing it.",
            "Confirmation is acceptance above the prior candle open; failure back under midpoint weakens it.",
        )
    elif (
        prev_dir == "bull"
        and cur_dir == "bear"
        and float(current["close"]) < prev_mid
        and float(current["close"]) > float(previous["open"])
    ):
        pattern = _base_pattern(
            "Dark cloud cover",
            "bearish reversal attempt",
            "After a green candle, the red candle cut back through more than half of the prior body.",
            "Confirmation is acceptance below the prior candle open; reclaiming midpoint weakens it.",
        )
    elif cur_low > prev_low and cur_high < prev_high:
        pattern = _base_pattern(
            "Harami / inside body",
            "pause / compression",
            "The current candle body sits inside the prior body. Momentum paused and compressed.",
            "Wait for expansion out of the mother candle range; guessing inside it is chop bait.",
        )
    elif float(current["high"]) < float(previous["high"]) and float(
        current["low"]
    ) > float(previous["low"]):
        pattern = _base_pattern(
            "Inside bar",
            "compression",
            "The full candle range is inside the prior candle. Volatility compressed.",
            "Monitor both boundaries. No directional edge exists unless the boundary test has participation, order-flow support, acceptance, and positive net expectancy.",
        )
    elif float(current["high"]) > float(previous["high"]) and float(
        current["low"]
    ) < float(previous["low"]):
        pattern = _base_pattern(
            "Outside bar",
            "volatility expansion",
            "The current candle broke both sides of the prior range. The auction expanded violently.",
            "Close location is only the event marker; require location, participation, order flow, acceptance, and expectancy before upgrading it.",
        )
    elif abs(float(current["low"]) - float(previous["low"])) <= tolerance:
        pattern = _base_pattern(
            "Tweezer bottom",
            "support test",
            "Two candles defended nearly the same low. Sellers failed to get fresh downside progress.",
            "A close above both candle highs confirms demand; losing the shared low invalidates it.",
        )
    elif abs(float(current["high"]) - float(previous["high"])) <= tolerance:
        pattern = _base_pattern(
            "Tweezer top",
            "resistance test",
            "Two candles rejected nearly the same high. Buyers failed to get fresh upside progress.",
            "A close below both candle lows confirms supply; reclaiming the shared high invalidates it.",
        )
    else:
        pattern = _base_pattern(
            "No clean two-candle pattern",
            "neutral",
            "The last two bars do not form a classic configuration. That is still useful: no forced read.",
            "Use structure, VWAP, levels, and volume instead of inventing a pattern. Very rude, but honest.",
        )

    return _attach_candles(
        {
            **pattern,
            "window": "2-candle",
            "clock": f"{_clock_label(previous.get('minute'))}→{_clock_label(current.get('minute'))}",
            "context": context,
        },
        [previous, current],
    )


def _long_body(bar: dict[str, Any]) -> bool:
    return float(bar["body_pct"]) >= 0.50


def _small_body(bar: dict[str, Any]) -> bool:
    return float(bar["body_pct"]) <= 0.35


def classify_three_candle(
    first: dict[str, Any],
    second: dict[str, Any],
    third: dict[str, Any],
    context: str = "",
) -> dict[str, Any]:
    first_mid = (float(first["open"]) + float(first["close"])) / 2
    first_dir = str(first["direction"])
    second_dir = str(second["direction"])
    third_dir = str(third["direction"])
    rising_closes = (
        float(first["close"]) < float(second["close"]) < float(third["close"])
    )
    falling_closes = (
        float(first["close"]) > float(second["close"]) > float(third["close"])
    )

    if (
        first_dir == "bear"
        and _small_body(second)
        and third_dir == "bull"
        and float(third["close"]) > first_mid
    ):
        pattern = _base_pattern(
            "Morning star",
            "bullish reversal attempt",
            "A strong red candle, a pause candle, then a green reclaim through the first candle midpoint.",
            "Best confirmation is follow-through above the third candle high and support holding under the star.",
        )
    elif (
        first_dir == "bull"
        and _small_body(second)
        and third_dir == "bear"
        and float(third["close"]) < first_mid
    ):
        pattern = _base_pattern(
            "Evening star",
            "bearish reversal attempt",
            "A strong green candle, a pause candle, then a red rejection through the first candle midpoint.",
            "Best confirmation is follow-through below the third candle low and resistance holding over the star.",
        )
    elif (
        all(
            str(bar["direction"]) == "bull" and _long_body(bar)
            for bar in (first, second, third)
        )
        and rising_closes
    ):
        pattern = _base_pattern(
            "Three white soldiers",
            "bullish continuation",
            "Three strong green candles closing higher in sequence. Buyers are walking price up bar by bar.",
            "Watch for acceptance above the sequence; a fast fade of the third candle warns exhaustion.",
        )
    elif (
        all(
            str(bar["direction"]) == "bear" and _long_body(bar)
            for bar in (first, second, third)
        )
        and falling_closes
    ):
        pattern = _base_pattern(
            "Three black crows",
            "bearish continuation",
            "Three strong red candles closing lower in sequence. Sellers are walking price down bar by bar.",
            "Watch for acceptance below the sequence; a fast reclaim of the third candle warns exhaustion.",
        )
    elif (
        first_dir == "bear"
        and second_dir == "bull"
        and third_dir == "bull"
        and float(third["close"]) > float(first["high"])
    ):
        pattern = _base_pattern(
            "Three inside up",
            "bullish reversal attempt",
            "A bearish candle, an inside/reclaim candle, then a breakout above the first candle high.",
            "Confirmation is holding above the breakout high; falling back inside means chop reclaimed the move.",
        )
    elif (
        first_dir == "bull"
        and second_dir == "bear"
        and third_dir == "bear"
        and float(third["close"]) < float(first["low"])
    ):
        pattern = _base_pattern(
            "Three inside down",
            "bearish reversal attempt",
            "A bullish candle, an inside/rejection candle, then a breakdown below the first candle low.",
            "Confirmation is holding below the breakdown low; reclaiming inside means sellers lost control.",
        )
    else:
        pattern = _base_pattern(
            "No clean three-candle pattern",
            "neutral",
            "The last three bars do not form a classic named pattern. No need to force one.",
            "Study the bodies and closes, then lean on VWAP, levels, structure, and volume for context.",
        )

    return _attach_candles(
        {
            **pattern,
            "window": "3-candle",
            "clock": f"{_clock_label(first.get('minute'))}→{_clock_label(third.get('minute'))}",
            "context": context,
        },
        [first, second, third],
    )


def build_candle_coach(
    rows: list[tuple[Any, ...]] | list[list[Any]],
    sharpedge_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"schema": CANDLE_COACH_SCHEMA, "available": False, "reason": "no rows"}
    packets = [_bar_packet(row, idx) for idx, row in enumerate(rows)]
    context = _trend_context(packets[-6:])
    current = packets[-1]
    previous = packets[-2] if len(packets) >= 2 else None
    third_back = packets[-3] if len(packets) >= 3 else None
    single = classify_single_candle(current, context)
    two = classify_two_candle(previous, current, context) if previous else {}
    three = (
        classify_three_candle(third_back, previous, current, context)
        if third_back and previous
        else {}
    )
    structure = classify_structure_pattern(packets, context)

    notable: list[dict[str, Any]] = []
    start = max(0, len(packets) - 6)
    for idx in range(start, len(packets)):
        one = classify_single_candle(
            packets[idx], _trend_context(packets[max(0, idx - 5) : idx + 1])
        )
        if one["name"] != "Ordinary range candle":
            notable.append(one)
        if idx > 0:
            pair = classify_two_candle(
                packets[idx - 1],
                packets[idx],
                _trend_context(packets[max(0, idx - 5) : idx + 1]),
            )
            if pair["name"] != "No clean two-candle pattern":
                notable.append(pair)
        if idx > 1:
            trio = classify_three_candle(
                packets[idx - 2],
                packets[idx - 1],
                packets[idx],
                _trend_context(packets[max(0, idx - 5) : idx + 1]),
            )
            if trio["name"] != "No clean three-candle pattern":
                notable.append(trio)

    framework = build_candle_framework(packets, current, context, sharpedge_context)
    pattern_context = {
        "latest_single": single,
        "latest_pair": two,
        "latest_three": three,
        "latest_structure": structure,
    }
    expectancy = lookup_candle_expectancy(pattern_context, sharpedge_context)
    framework["candle_expectancy"] = expectancy
    for idx, gate in enumerate(framework.get("gates") or []):
        if gate.get("label") == "Net expectancy":
            permission = (sharpedge_context or {}).get("permission") or (
                sharpedge_context or {}
            ).get("trade_permission")
            framework["gates"][idx] = expectancy_gate_from_lookup(
                permission, expectancy
            )
            break
    if framework.get("auction_execution_box") or {}:
        framework["auction_execution_box"]["candle_expectancy"] = expectancy
    vector_lesson = build_candle_vector_lesson(
        patterns=[single, two, three, structure],
        framework=framework,
        sharpedge_context=sharpedge_context,
    )
    auction_box = framework.get("auction_execution_box") or {}
    if auction_box:
        auction_box.setdefault("premise", {})["event_stack"] = [
            item.get("name")
            for item in (single, two, three, structure)
            if item.get("name")
        ]
    headline = f"{framework['output']}: {single['name']} / {two.get('name', 'waiting for pair')}"
    if three and three.get("name") != "No clean three-candle pattern":
        headline = f"{headline} / {three['name']}"
    if structure and structure.get("name") != "No clean larger structure":
        headline = f"{headline} / {structure['name']}"

    return {
        "schema": CANDLE_COACH_SCHEMA,
        "available": True,
        "authority": "education_only_not_trade_permission",
        "headline": headline,
        "output_state": framework["output"],
        "execution_framework": framework,
        "candle_expectancy": expectancy,
        "candle_vector_lesson": vector_lesson,
        "data_integrity": framework["gates"][0],
        "context": context,
        "latest_single": single,
        "latest_pair": two,
        "latest_three": three,
        "latest_structure": structure,
        "recent_notable": notable[-2:],
        "pattern_library": PATTERN_LIBRARY,
        "pattern_encyclopedia": encyclopedia_packet(),
        "lesson": framework["lesson"],
    }


__all__ = [
    "build_candle_coach",
    "classify_single_candle",
    "classify_structure_pattern",
    "classify_three_candle",
    "classify_two_candle",
]
