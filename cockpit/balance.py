"""Balance-location helpers for the SharpEdge cockpit."""

from __future__ import annotations


BULLISH = "CALLS"
BEARISH = "PUTS"
NEUTRAL = "NEUTRAL"


def position_in_balance(
    close: float, balance_low: float, balance_high: float, *, clamp: bool = True
) -> float:
    """Return normalized location inside a balance range on a 0.0-1.0 scale."""
    width = balance_high - balance_low
    if width <= 0:
        return 0.5
    raw = (close - balance_low) / width
    if not clamp:
        return raw
    return max(0.0, min(1.0, raw))


def balance_label(position: float) -> str:
    """Return a plain-English location label for a normalized balance position."""
    if position >= 0.8:
        return "TOP"
    if position <= 0.2:
        return "BOTTOM"
    return "MIDDLE"


def _prior_rows(rows, window: int) -> list[tuple]:
    if not rows:
        return []
    ref_rows = rows[-(window + 1) : -1] if len(rows) > 1 else rows
    return ref_rows[-window:] if len(ref_rows) > window else ref_rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _snapshot_signal(snapshot: dict) -> tuple[str, str]:
    state = snapshot["balance_state"]
    pos = snapshot["position_in_balance"]
    if state == "above":
        return BULLISH, "accepted_above_balance"
    if state == "below":
        return BEARISH, "accepted_below_balance"
    if pos >= 0.8:
        return BEARISH, "top_of_balance"
    if pos <= 0.2:
        return BULLISH, "bottom_of_balance"
    if 0.4 <= pos <= 0.6:
        return NEUTRAL, "middle_of_balance"
    return NEUTRAL, "inside_balance"


def _balance_snapshot(
    close: float,
    balance_low: float,
    balance_high: float,
    *,
    reference: str,
    sample_size: int,
) -> dict:
    raw_position = position_in_balance(close, balance_low, balance_high, clamp=False)
    position = max(0.0, min(1.0, raw_position))
    state = (
        "above" if raw_position > 1.0 else "below" if raw_position < 0.0 else "inside"
    )
    width = balance_high - balance_low
    width_pct = width / close * 100 if close else 0.0
    snapshot = {
        "balance_high": round(balance_high, 2),
        "balance_low": round(balance_low, 2),
        "position_in_balance": round(position, 4),
        "balance_state": state,
        "balance_label": balance_label(position),
        "balance_width_pct": round(width_pct, 3),
        "balance_window_bars": sample_size,
        "balance_reference": reference,
    }
    bias, signal = _snapshot_signal(snapshot)
    snapshot["balance_bias"] = bias
    snapshot["balance_signal"] = signal
    return snapshot


def opening_balance_context_from_rows(rows, minutes: int = 30) -> dict:
    """Build opening-balance context from the first session chunk."""
    if not rows:
        return {}
    close = rows[-1][4]
    ref_rows = [bar for bar in rows[:-1] if bar[0] < minutes] or rows[:1]
    observed = min(len(ref_rows), minutes)
    return _balance_snapshot(
        close,
        min(bar[3] for bar in ref_rows),
        max(bar[2] for bar in ref_rows),
        reference=f"opening_{observed}m",
        sample_size=len(ref_rows),
    )


def recent_balance_context_from_rows(rows, window: int = 20) -> dict:
    """Build a rolling recent balance box from prior bars."""
    if not rows:
        return {}
    close = rows[-1][4]
    ref_rows = _prior_rows(rows, window) or rows[:1]
    return _balance_snapshot(
        close,
        min(bar[3] for bar in ref_rows),
        max(bar[2] for bar in ref_rows),
        reference=f"recent_{len(ref_rows)}_bar",
        sample_size=len(ref_rows),
    )


def value_balance_context_from_rows(rows, window: int = 60) -> dict:
    """Approximate session value with percentile bounds of prior closes."""
    if not rows:
        return {}
    close = rows[-1][4]
    ref_rows = _prior_rows(rows, window) or rows[:1]
    closes = [bar[4] for bar in ref_rows]
    low = _quantile(closes, 0.15)
    high = _quantile(closes, 0.85)
    if high <= low:
        low, high = min(closes), max(closes)
    return _balance_snapshot(
        close,
        low,
        high,
        reference=f"value_{len(ref_rows)}_bar",
        sample_size=len(ref_rows),
    )


def dominant_balance_name(rows) -> tuple[str, str]:
    """Choose which balance lens matters most for the current session phase."""
    minute = int(rows[-1][0]) if rows else 0
    if minute < 60:
        return "opening_balance", "first hour: opening balance controls the auction"
    if minute < 240:
        return "recent_balance", "mid-session: the active recent box matters most"
    return "value_balance", "late day: accepted value matters most"


def _balance_confluence(models: dict) -> dict:
    bias_map = {name: model.get("balance_bias", NEUTRAL) for name, model in models.items()}
    bullish = [name for name, bias in bias_map.items() if bias == BULLISH]
    bearish = [name for name, bias in bias_map.items() if bias == BEARISH]
    neutral = [name for name, bias in bias_map.items() if bias == NEUTRAL]
    if bullish and bearish:
        return {
            "state": "disagreement",
            "score": 28,
            "bias": NEUTRAL,
            "agreement_count": max(len(bullish), len(bearish)),
            "aligned_models": [],
            "reason": f"balance lenses disagree: bulls={', '.join(bullish)} bears={', '.join(bearish)}",
        }
    if bullish or bearish:
        aligned = bullish or bearish
        bias = BULLISH if bullish else BEARISH
        count = len(aligned)
        score = {1: 58, 2: 76}.get(count, 88)
        return {
            "state": "aligned" if count >= 2 else "lean",
            "score": score,
            "bias": bias,
            "agreement_count": count,
            "aligned_models": aligned,
            "reason": f"{count} balance lens(es) align {bias.lower()}: {', '.join(aligned)}",
        }
    return {
        "state": "neutral",
        "score": 42,
        "bias": NEUTRAL,
        "agreement_count": 0,
        "aligned_models": neutral,
        "reason": "balance lenses are centered; no edge confluence",
    }


def _balance_disagreement(models: dict) -> dict:
    bullish = [name for name, model in models.items() if model.get("balance_bias") == BULLISH]
    bearish = [name for name, model in models.items() if model.get("balance_bias") == BEARISH]
    neutral = [name for name, model in models.items() if model.get("balance_bias") == NEUTRAL]
    has_disagreement = bool(bullish and bearish)
    reason = (
        f"disagreement: bulls={', '.join(bullish)} bears={', '.join(bearish)}"
        if has_disagreement
        else "balance lenses are not fighting each other"
    )
    return {
        "has_disagreement": has_disagreement,
        "bullish_models": bullish,
        "bearish_models": bearish,
        "neutral_models": neutral,
        "reason": reason,
    }


def _dominant_balance_flip(rows, dominant_name: str) -> dict:
    if len(rows) < 2:
        return {"flipped": False, "from": dominant_name, "to": dominant_name, "reason": "no prior bar"}
    previous_name, _previous_reason = dominant_balance_name(rows[:-1])
    flipped = previous_name != dominant_name
    reason = (
        f"dominant balance lens flipped from {previous_name} to {dominant_name}"
        if flipped
        else f"dominant balance lens unchanged at {dominant_name}"
    )
    return {"flipped": flipped, "from": previous_name, "to": dominant_name, "reason": reason}


def build_balance_stack(rows) -> dict:
    """Return multi-lens balance context plus the dominant lens to trade against."""
    if not rows:
        return {}
    models = {
        "opening_balance": opening_balance_context_from_rows(rows),
        "recent_balance": recent_balance_context_from_rows(rows),
        "value_balance": value_balance_context_from_rows(rows),
    }
    dominant_name, dominant_reason = dominant_balance_name(rows)
    dominant = models.get(dominant_name) or models["recent_balance"]
    flip = _dominant_balance_flip(rows, dominant_name)
    closes = [bar[4] for bar in rows]
    session_low = min(closes)
    session_high = max(closes)
    return {
        **dominant,
        "dominant_balance_name": dominant_name,
        "dominant_balance_reason": dominant_reason,
        "dominant_balance_previous_name": flip["from"],
        "dominant_balance_flip": flip,
        "balance_models": models,
        "balance_confluence": _balance_confluence(models),
        "balance_disagreement": _balance_disagreement(models),
        "session_position_in_range": round(
            position_in_balance(closes[-1], session_low, session_high), 4
        ),
    }
