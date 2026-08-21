"""Volume-weighted RSI advisory read for the live cockpit.

This is not an entry trigger. It is a pressure-quality surface: when volume is
usable, it asks whether price momentum is being confirmed by volume-weighted
up/down pressure or whether exhaustion/divergence is forming.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "sharpedge.volume_weighted_rsi.v1"
DEFAULT_PERIOD = 14
DEFAULT_DIVERGENCE_LOOKBACK = 20
MIN_USABLE_VOLUME_SHARE = 0.60
DIVERGENCE_MIN_RSI_DELTA = 4.0
CONFIRM_SLOPE = 3.0


def _inactive(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "active": False,
        "state": "inactive",
        "bias": "NEUTRAL",
        "score": 0,
        "value": None,
        "slope": 0.0,
        "volume_quality": "missing",
        "advisory_only": True,
        "reason": reason,
    }


def _rsi_from_weighted_moves(window: list[tuple]) -> float:
    weighted_gains = 0.0
    weighted_losses = 0.0
    for idx in range(1, len(window)):
        prev_close = float(window[idx - 1][4])
        close = float(window[idx][4])
        volume = max(float(window[idx][5]), 0.0)
        delta = close - prev_close
        if delta > 0:
            weighted_gains += delta * volume
        elif delta < 0:
            weighted_losses += abs(delta) * volume
    if weighted_gains == 0 and weighted_losses == 0:
        return 50.0
    if weighted_losses == 0:
        return 100.0
    if weighted_gains == 0:
        return 0.0
    rs = weighted_gains / weighted_losses
    return 100.0 - (100.0 / (1.0 + rs))


def _rolling_values(bars: list[tuple], period: int) -> list[float]:
    values = []
    for end in range(period, len(bars) + 1):
        values.append(_rsi_from_weighted_moves(bars[end - period : end]))
    return values


def _volume_quality(bars: list[tuple], period: int) -> tuple[bool, str, float]:
    recent = bars[-period:]
    if len(recent) < period:
        return False, "insufficient", 0.0
    positive = [bar for bar in recent if float(bar[5]) > 0]
    positive_share = len(positive) / len(recent)
    total_volume = sum(float(bar[5]) for bar in recent)
    if total_volume <= 0:
        return False, "missing", positive_share
    if positive_share < MIN_USABLE_VOLUME_SHARE:
        return False, "sparse", positive_share
    return True, "usable", positive_share


def _score_from_state(state: str, value: float, slope: float) -> int:
    if state in {"bullish_divergence", "bearish_divergence"}:
        return 72
    if state in {"confirming_up", "confirming_down"}:
        return 64
    if state in {"overbought", "oversold"}:
        return 58
    return max(35, min(60, int(50 + abs(value - 50) * 0.2 + abs(slope) * 0.5)))


def build_volume_weighted_rsi(
    bars: list[tuple],
    *,
    period: int = DEFAULT_PERIOD,
    divergence_lookback: int = DEFAULT_DIVERGENCE_LOOKBACK,
) -> dict[str, Any]:
    """Return a volume-weighted RSI advisory packet.

    Bars are cockpit tuples: (session_minute, open, high, low, close, volume).
    The output is deliberately advisory and volume-quality gated.
    """
    if len(bars) < period + 1:
        return _inactive(f"need at least {period + 1} bars for volume-weighted RSI")

    usable, quality, positive_share = _volume_quality(bars, period)
    if not usable:
        return _inactive(
            f"volume quality {quality}; do not trust volume-weighted oscillator"
        ) | {
            "volume_quality": quality,
            "positive_volume_share": round(positive_share, 4),
        }

    values = _rolling_values(bars, period)
    value = values[-1]
    prev_value = values[-2] if len(values) >= 2 else value
    slope = value - prev_value
    closes = [float(bar[4]) for bar in bars]
    recent_closes = closes[-divergence_lookback:]
    prior_values = values[-divergence_lookback:-1] or values[:-1]
    prior_low_price = min(recent_closes[:-1]) if len(recent_closes) > 1 else closes[-1]
    prior_high_price = max(recent_closes[:-1]) if len(recent_closes) > 1 else closes[-1]
    prior_low_rsi = min(prior_values) if prior_values else value
    prior_high_rsi = max(prior_values) if prior_values else value

    state = "neutral"
    bias = "NEUTRAL"
    reason = f"volume-weighted RSI {value:.1f}, slope {slope:+.1f}"
    if (
        closes[-1] <= prior_low_price
        and value >= prior_low_rsi + DIVERGENCE_MIN_RSI_DELTA
    ):
        state = "bullish_divergence"
        bias = "CALLS"
        reason = (
            f"price pressed a low, but volume-weighted RSI held higher "
            f"({prior_low_rsi:.1f}->{value:.1f})"
        )
    elif (
        closes[-1] >= prior_high_price
        and value <= prior_high_rsi - DIVERGENCE_MIN_RSI_DELTA
    ):
        state = "bearish_divergence"
        bias = "PUTS"
        reason = (
            f"price pressed a high, but volume-weighted RSI failed to confirm "
            f"({prior_high_rsi:.1f}->{value:.1f})"
        )
    elif slope >= CONFIRM_SLOPE and value >= 55:
        state = "confirming_up"
        bias = "CALLS"
        reason = f"volume-weighted RSI rising with upside pressure ({value:.1f}, {slope:+.1f})"
    elif slope <= -CONFIRM_SLOPE and value <= 45:
        state = "confirming_down"
        bias = "PUTS"
        reason = f"volume-weighted RSI falling with downside pressure ({value:.1f}, {slope:+.1f})"
    elif value >= 72:
        state = "overbought"
        reason = f"volume-weighted upside pressure is extended ({value:.1f})"
    elif value <= 28:
        state = "oversold"
        reason = f"volume-weighted downside pressure is extended ({value:.1f})"

    return {
        "schema": SCHEMA,
        "active": True,
        "state": state,
        "bias": bias,
        "score": _score_from_state(state, value, slope),
        "value": round(value, 2),
        "slope": round(slope, 2),
        "period": period,
        "divergence_lookback": divergence_lookback,
        "volume_quality": quality,
        "positive_volume_share": round(positive_share, 4),
        "advisory_only": True,
        "reason": reason,
    }


__all__ = ["build_volume_weighted_rsi"]
