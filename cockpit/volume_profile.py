"""Move-aware volume profile for SharpEdge cockpit reads."""

from __future__ import annotations

from statistics import median

MOVE_LOOKBACK = 15
RECENT_LOOKBACK = 5
LOCAL_BASELINE_LOOKBACK = 30
MIN_MOVE_PCT = 0.03


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median_or_one(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None and value > 0]
    return float(median(clean)) if clean else 1.0


def _pct(delta: float, base: float) -> float:
    return delta / base * 100 if base else 0.0


def _move_direction(move_pct: float) -> str:
    if move_pct >= MIN_MOVE_PCT:
        return "up"
    if move_pct <= -MIN_MOVE_PCT:
        return "down"
    return "flat"


def _aligned_volume_share(window: list[tuple], direction: str) -> float:
    total = sum(float(bar[5]) for bar in window) or 1.0
    if direction == "flat":
        return 0.5
    aligned = 0.0
    for _minute, open_, _high, _low, close, volume in window:
        if direction == "up" and close > open_:
            aligned += float(volume)
        elif direction == "down" and close < open_:
            aligned += float(volume)
    return aligned / total


def _path_efficiency(window: list[tuple]) -> float:
    closes = [float(bar[4]) for bar in window]
    if len(closes) < 2:
        return 0.0
    net = abs(closes[-1] - closes[0])
    path = sum(abs(closes[idx] - closes[idx - 1]) for idx in range(1, len(closes)))
    return net / (path or 1e-9)


def _confirmation(local_mult: float, aligned_share: float, efficiency: float) -> str:
    if local_mult >= 1.15 and aligned_share >= 0.58 and efficiency >= 0.35:
        return "confirmed"
    if local_mult >= 0.90 and aligned_share >= 0.54 and efficiency >= 0.25:
        return "participating"
    if local_mult <= 0.70 or aligned_share <= 0.46:
        return "missing"
    return "mixed"


def build_volume_profile(bars: list[tuple]) -> dict:
    """Return volume context that follows the move, not just the session median."""
    if not bars:
        return {
            "schema": "sharpedge.volume_profile.v1",
            "confirmation": "missing",
            "move_direction": "flat",
            "move_pct": 0.0,
            "session_mult": 0.0,
            "local_mult": 0.0,
            "composite_mult": 0.0,
            "aligned_volume_share": 0.5,
            "path_efficiency": 0.0,
            "reason": "no bars for volume profile",
        }

    volumes = [float(bar[5]) for bar in bars]
    recent = volumes[-RECENT_LOOKBACK:]
    prior_end = max(len(volumes) - RECENT_LOOKBACK, 0)
    prior_start = max(prior_end - LOCAL_BASELINE_LOOKBACK, 0)
    local_baseline = volumes[prior_start:prior_end] or volumes[:prior_end] or volumes

    session_mult = _avg(recent) / _median_or_one(volumes)
    local_mult = _avg(recent) / _median_or_one(local_baseline)

    move_window = bars[-min(MOVE_LOOKBACK, len(bars)) :]
    start_close = float(move_window[0][4])
    end_close = float(move_window[-1][4])
    move_pct = _pct(end_close - start_close, start_close)
    direction = _move_direction(move_pct)
    aligned_share = _aligned_volume_share(move_window, direction)
    efficiency = _path_efficiency(move_window)
    confirmation = _confirmation(local_mult, aligned_share, efficiency)
    composite = (
        (local_mult * 0.55) + (session_mult * 0.25) + (aligned_share * 2.0 * 0.20)
    )

    reason = (
        f"{confirmation}: local {local_mult:.1f}x, session {session_mult:.1f}x, "
        f"aligned {aligned_share:.0%}, efficiency {efficiency:.0%}"
    )
    return {
        "schema": "sharpedge.volume_profile.v1",
        "confirmation": confirmation,
        "move_direction": direction,
        "move_pct": round(move_pct, 4),
        "session_mult": round(session_mult, 4),
        "local_mult": round(local_mult, 4),
        "composite_mult": round(composite, 4),
        "aligned_volume_share": round(aligned_share, 4),
        "path_efficiency": round(efficiency, 4),
        "reason": reason,
    }


__all__ = ["build_volume_profile"]
