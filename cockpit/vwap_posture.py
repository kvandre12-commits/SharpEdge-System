"""Canonical VWAP posture packet for SharpEdge."""

from __future__ import annotations

from typing import Any

HUGGING_BAND_PCT = 0.05
NEAR_BAND_PCT = 0.08
STRETCHED_BAND_PCT = 0.40
DEFAULT_ACCEPTANCE_WINDOW = 3
DEFAULT_MIN_ACCEPTANCE_CLOSES = 2


def _recent_close_acceptance(
    bars: list[tuple[Any, ...]] | list[list[Any]] | None,
    level: float | None,
    *,
    acceptance_window: int,
    min_acceptance_closes: int,
) -> tuple[bool, bool, int]:
    if not bars or not isinstance(level, (int, float)):
        return False, False, 0
    window = list(bars)[-int(acceptance_window) :]
    closes = [float(bar[4]) for bar in window]
    above_count = sum(close > float(level) for close in closes)
    below_count = sum(close < float(level) for close in closes)
    return (
        above_count >= int(min_acceptance_closes),
        below_count >= int(min_acceptance_closes),
        len(closes),
    )


def build_vwap_posture(
    pa: dict[str, Any] | None,
    bars: list[tuple[Any, ...]] | list[list[Any]] | None = None,
    *,
    hugging_band_pct: float = HUGGING_BAND_PCT,
    near_band_pct: float = NEAR_BAND_PCT,
    stretched_band_pct: float = STRETCHED_BAND_PCT,
    acceptance_window: int = DEFAULT_ACCEPTANCE_WINDOW,
    min_acceptance_closes: int = DEFAULT_MIN_ACCEPTANCE_CLOSES,
) -> dict[str, Any]:
    data = pa or {}
    spot = data.get("spot")
    vwap = data.get("vwap")
    raw_vs_vwap = data.get("vs_vwap")
    if isinstance(raw_vs_vwap, (int, float)):
        vs_vwap = float(raw_vs_vwap)
    elif isinstance(spot, (int, float)) and isinstance(vwap, (int, float)) and vwap:
        vs_vwap = (float(spot) - float(vwap)) / float(vwap) * 100
    else:
        vs_vwap = 0.0
    abs_vs_vwap = abs(vs_vwap)
    accepted_above, accepted_below, close_count = _recent_close_acceptance(
        bars,
        float(vwap) if isinstance(vwap, (int, float)) else None,
        acceptance_window=int(acceptance_window),
        min_acceptance_closes=int(min_acceptance_closes),
    )

    state = "hugging_vwap"
    posture = "magnet_chop"
    bias = "NEUTRAL"
    if abs_vs_vwap <= float(hugging_band_pct):
        state = "hugging_vwap"
        posture = "magnet_chop"
    elif abs_vs_vwap <= float(near_band_pct):
        state = "near_vwap"
        posture = "wait_for_acceptance"
        bias = "CALLS" if vs_vwap > 0 else "PUTS"
    elif vs_vwap > 0:
        bias = "CALLS"
        if abs_vs_vwap >= float(stretched_band_pct):
            state = "stretched_above"
            posture = "upside_extension"
        else:
            state = "above_vwap"
            posture = "upside_acceptance"
    else:
        bias = "PUTS"
        if abs_vs_vwap >= float(stretched_band_pct):
            state = "stretched_below"
            posture = "downside_extension"
        else:
            state = "below_vwap"
            posture = "downside_acceptance"

    return {
        "schema": "sharpedge.vwap_posture.v1",
        "state": state,
        "posture": posture,
        "bias": bias,
        "spot": spot,
        "vwap": vwap,
        "vs_vwap_pct": round(vs_vwap, 3),
        "abs_vs_vwap_pct": round(abs_vs_vwap, 3),
        "has_upside_control": state in {"above_vwap", "stretched_above"},
        "has_downside_control": state in {"below_vwap", "stretched_below"},
        "is_range_like": state in {"hugging_vwap", "near_vwap"},
        "is_stretched": state in {"stretched_above", "stretched_below"},
        "accepted_above_vwap": accepted_above,
        "accepted_below_vwap": accepted_below,
        "acceptance_window": int(acceptance_window),
        "recent_close_count": close_count,
        "min_acceptance_closes": int(min_acceptance_closes),
        "reason": f"spot is {vs_vwap:+.2f}% vs VWAP",
    }


__all__ = [
    "DEFAULT_ACCEPTANCE_WINDOW",
    "DEFAULT_MIN_ACCEPTANCE_CLOSES",
    "HUGGING_BAND_PCT",
    "NEAR_BAND_PCT",
    "STRETCHED_BAND_PCT",
    "build_vwap_posture",
]
