"""Shared primitives for execution-vector scoring."""

from __future__ import annotations


def clamp(value, low=0, high=100):
    return int(max(low, min(high, round(value))))


def buffer_for_price(price):
    return max(0.10, price * 0.0003) if price else 0.10


def bar_personality(bar):
    _minute, open_, high, low, close, _volume = bar
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    close_pos = (close - low) / rng
    return body, upper_wick, lower_wick, close_pos


def bias_label(score):
    if score >= 0.20:
        return "CALLS"
    if score <= -0.20:
        return "PUTS"
    return "NEUTRAL"


def gate_label(score):
    if score >= 72:
        return "PERMIT"
    if score >= 58:
        return "CAUTION"
    return "BLOCK"


def reasons_from_parts(parts):
    ordered = sorted(parts.items(), key=lambda item: item[1].score, reverse=True)
    best = [
        f"{name.replace('_score', '')}: {part.reason}" for name, part in ordered[:3]
    ]
    worst = [
        f"{name.replace('_score', '')}: {part.reason}" for name, part in ordered[-2:]
    ]
    return {"supporting": best, "warnings": worst}


def serialize_parts(parts):
    return {
        name: {
            "score": part.score,
            "bias": bias_label(part.bias),
            "reason": part.reason,
        }
        for name, part in parts.items()
    }


__all__ = [
    "bar_personality",
    "bias_label",
    "buffer_for_price",
    "clamp",
    "gate_label",
    "reasons_from_parts",
    "serialize_parts",
]
