"""Derive explicit weekly/daily/intraday agreement for the cockpit."""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"
MIXED = "mixed"


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _title_bias(bias: str) -> str:
    return {
        BULLISH: "Bullish",
        BEARISH: "Bearish",
        NEUTRAL: "Neutral",
        MIXED: "Mixed",
    }.get(bias, "Neutral")


def _kind_for_row(bias: str, score: int, *, stance: str = "directional") -> str:
    if stance in {"caution", "stand_down"}:
        return "warn"
    if bias == BULLISH:
        return "ok"
    if bias == BEARISH and score >= 75:
        return "bad"
    if bias == BEARISH:
        return "warn"
    return "info"


def _row(
    timeframe: str,
    *,
    bias: str,
    score: int,
    detail: str,
    label: str | None = None,
    kind: str | None = None,
    stance: str = "directional",
    basis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    display = label or _title_bias(bias)
    return {
        "timeframe": timeframe,
        "bias": bias,
        "label": display,
        "score": int(score),
        "kind": kind or _kind_for_row(bias, score, stance=stance),
        "stance": stance,
        "detail": detail,
        "basis": basis or {},
    }


def build_weekly_timeframe(weekly_context: dict[str, Any] | None) -> dict[str, Any]:
    weekly_context = weekly_context or {}
    if not weekly_context.get("lookback_days"):
        return _row(
            "Weekly",
            bias=NEUTRAL,
            score=50,
            label="Unavailable",
            detail="Weekly carry map unavailable.",
            basis={"context_available": False},
        )

    range_pos = float(weekly_context.get("range_position_pct") or 50.0)
    kind = str(weekly_context.get("kind") or "info")
    headline = str(weekly_context.get("headline") or weekly_context.get("detail") or "")
    headline_lower = headline.lower()

    if "through h1" in headline_lower:
        bias = BULLISH
        score = 86
    elif "upper carry shelf" in headline_lower:
        bias = BULLISH
        score = 82
    elif "carry box" in headline_lower:
        bias = BULLISH if range_pos >= 60 else NEUTRAL
        score = 74 if bias == BULLISH else 60
    elif "lower carry shelf" in headline_lower:
        if range_pos >= 80:
            bias = BULLISH
            score = 72
        elif range_pos >= 55:
            bias = NEUTRAL
            score = 62
        else:
            bias = BEARISH
            score = 74
    elif "l1 washout low" in headline_lower:
        bias = BEARISH
        score = 84
    elif kind == "ok":
        bias = BULLISH
        score = 82 if range_pos >= 65 else 74
    elif kind == "bad":
        bias = BEARISH
        score = 84
    elif kind == "warn":
        if range_pos >= 80:
            bias = BULLISH
            score = 70
        elif range_pos < 50:
            bias = BEARISH
            score = 72
        else:
            bias = NEUTRAL
            score = 62
    else:
        if range_pos >= 60:
            bias = BULLISH
            score = 68
        elif range_pos <= 40:
            bias = BEARISH
            score = 68
        else:
            bias = NEUTRAL
            score = 58

    detail = (
        f"{headline} Range position {range_pos:.0f}% of the 5-day map."
        if headline
        else f"Range position {range_pos:.0f}% of the 5-day map."
    )
    row_kind = kind if kind in {"ok", "bad", "warn", "info"} else None
    return _row(
        "Weekly",
        bias=bias,
        score=score,
        detail=detail,
        kind=row_kind,
        basis={
            "range_position_pct": round(range_pos, 1),
            "kind": kind,
            "headline": headline,
        },
    )


def build_daily_timeframe(
    daily_rows: list[dict[str, Any]] | None,
    *,
    spot: float,
) -> dict[str, Any]:
    rows = [
        row
        for row in (daily_rows or [])
        if all(row.get(key) is not None for key in ("open", "high", "low", "close"))
    ]
    if len(rows) < 5:
        return _row(
            "Daily",
            bias=NEUTRAL,
            score=50,
            label="Unavailable",
            detail="Not enough daily bars for a trend read.",
            basis={"bar_count": len(rows)},
        )

    recent = rows[-20:]
    closes = [float(row["close"]) for row in recent]
    highs = [float(row["high"]) for row in recent]
    lows = [float(row["low"]) for row in recent]
    sma5 = sum(closes[-5:]) / min(5, len(closes))
    sma20 = sum(closes) / len(closes)
    prev_close = closes[-1]
    range_low = min(lows)
    range_high = max(highs)
    range_span = max(range_high - range_low, 1e-9)
    range_pos = max(0.0, min(100.0, (spot - range_low) / range_span * 100.0))

    bull_points = 0
    bear_points = 0
    if spot >= sma5:
        bull_points += 30
    else:
        bear_points += 30
    if sma5 >= sma20:
        bull_points += 25
    else:
        bear_points += 25
    if range_pos >= 60:
        bull_points += 20
    elif range_pos <= 40:
        bear_points += 20
    if spot >= prev_close:
        bull_points += 10
    else:
        bear_points += 10
    if prev_close >= sma20:
        bull_points += 10
    else:
        bear_points += 10

    if bull_points >= 60 and bull_points - bear_points >= 15:
        bias = BULLISH
        score = _clamp_score(55 + bull_points * 0.25)
    elif bear_points >= 60 and bear_points - bull_points >= 15:
        bias = BEARISH
        score = _clamp_score(55 + bear_points * 0.25)
    else:
        bias = NEUTRAL
        score = _clamp_score(
            52
            + abs(bull_points - bear_points) * 0.2
            + max(bull_points, bear_points) * 0.1
        )

    detail = (
        f"spot ${spot:.2f} vs 5d avg ${sma5:.2f} / 20d avg ${sma20:.2f}; "
        f"20d range position {range_pos:.0f}%."
    )
    return _row(
        "Daily",
        bias=bias,
        score=score,
        detail=detail,
        basis={
            "bar_count": len(recent),
            "sma5": round(sma5, 2),
            "sma20": round(sma20, 2),
            "previous_close": round(prev_close, 2),
            "range_position_pct": round(range_pos, 1),
            "bull_points": bull_points,
            "bear_points": bear_points,
        },
    )


def build_intraday_timeframe(
    permission: dict[str, Any] | None,
) -> dict[str, Any]:
    permission = permission or {}
    spine = permission.get("bucket_conditioned_spine") or {}
    gate = str(spine.get("gate") or permission.get("trade_gate") or "BLOCK")
    score = int(spine.get("score") or permission.get("trade_permission_score") or 0)
    raw_bias = str(spine.get("bias") or permission.get("bias") or "NEUTRAL")
    posture = str(spine.get("diagnostic_posture") or "watch_only_context")
    reason = str(spine.get("reason") or permission.get("reason") or "")

    if gate == "BLOCK" or posture in {"stand_down", "stand_down_context_only"}:
        return _row(
            "Intraday",
            bias=NEUTRAL,
            score=score,
            label="Stand Down",
            kind="warn",
            stance="stand_down",
            detail=f"{gate} | {posture.replace('_', ' ')}. {reason}".strip(),
            basis={"gate": gate, "posture": posture, "raw_bias": raw_bias},
        )
    if gate == "CAUTION" or posture in {"watch_edges", "watch_edges_context_only"}:
        return _row(
            "Intraday",
            bias=NEUTRAL,
            score=score,
            label="Neutral/Caution",
            kind="warn",
            stance="caution",
            detail=f"{gate} | {posture.replace('_', ' ')}. {reason}".strip(),
            basis={"gate": gate, "posture": posture, "raw_bias": raw_bias},
        )

    bias = (
        BULLISH if raw_bias == "CALLS" else BEARISH if raw_bias == "PUTS" else NEUTRAL
    )
    return _row(
        "Intraday",
        bias=bias,
        score=score,
        detail=f"{gate} | {posture.replace('_', ' ')}. {reason}".strip(),
        basis={"gate": gate, "posture": posture, "raw_bias": raw_bias},
    )


def _higher_timeframe_bias(weekly: dict[str, Any], daily: dict[str, Any]) -> str:
    weekly_bias = str(weekly.get("bias") or NEUTRAL)
    daily_bias = str(daily.get("bias") or NEUTRAL)
    if weekly_bias == daily_bias and weekly_bias != NEUTRAL:
        return weekly_bias
    if weekly_bias == NEUTRAL and daily_bias in {BULLISH, BEARISH}:
        return daily_bias
    if daily_bias == NEUTRAL and weekly_bias in {BULLISH, BEARISH}:
        return weekly_bias
    return MIXED


def _caution_clause(pa: dict[str, Any], higher_bias: str) -> str:
    posture = build_range_posture(pa)
    upside = bool(posture.get("is_upper_half")) or str(posture.get("vwap_state")) in {
        "above_vwap",
        "stretched_above",
    }
    downside = bool(posture.get("is_lower_half")) or str(posture.get("vwap_state")) in {
        "below_vwap",
        "stretched_below",
    }

    if higher_bias == BULLISH:
        if upside:
            return "but intraday conditions favor fading extensions into resistance until participation or momentum improves."
        if downside:
            return "but intraday conditions favor buying flushes into support only after reclaim and acceptance improve."
        return "but intraday conditions still argue for patience until participation or momentum improves."

    if higher_bias == BEARISH:
        if downside:
            return "but intraday conditions favor fading breakdown extensions into support until participation or momentum improves."
        if upside:
            return "but intraday conditions favor selling strength into resistance only if it stalls cleanly."
        return "but intraday conditions still argue for patience until participation or momentum improves."

    return "and intraday conditions still argue for patience until participation or momentum improves."


def _summary(
    pa: dict[str, Any],
    weekly: dict[str, Any],
    daily: dict[str, Any],
    intraday: dict[str, Any],
) -> str:
    higher_bias = _higher_timeframe_bias(weekly, daily)
    stance = str(intraday.get("stance") or "directional")
    intraday_bias = str(intraday.get("bias") or NEUTRAL)

    if higher_bias == BULLISH:
        prefix = "Higher-timeframe trend remains bullish"
    elif higher_bias == BEARISH:
        prefix = "Higher-timeframe trend remains bearish"
    else:
        prefix = "Higher-timeframe signals are mixed"

    if stance == "caution":
        return f"{prefix}, {_caution_clause(pa, higher_bias)}".replace(", but", ", but")
    if stance == "stand_down":
        return f"{prefix}, but intraday conditions still argue for standing down until structure and participation stabilize."
    if higher_bias == BULLISH and intraday_bias == BULLISH:
        return f"{prefix}, and intraday conditions are supportive enough for bullish continuation."
    if higher_bias == BEARISH and intraday_bias == BEARISH:
        return f"{prefix}, and intraday conditions are supportive enough for bearish continuation."
    if higher_bias in {BULLISH, BEARISH} and intraday_bias in {BULLISH, BEARISH}:
        return f"{prefix}, but intraday conditions are leaning countertrend, so execution should stay selective."
    return f"{prefix}, and intraday conditions are still balanced enough to demand patience."


def build_timeframe_agreement(
    pa: dict[str, Any],
    weekly_context: dict[str, Any] | None,
    daily_rows: list[dict[str, Any]] | None,
    permission: dict[str, Any] | None,
) -> dict[str, Any]:
    weekly = build_weekly_timeframe(weekly_context)
    daily = build_daily_timeframe(daily_rows, spot=float(pa.get("spot") or 0.0))
    intraday = build_intraday_timeframe(permission)
    higher_bias = _higher_timeframe_bias(weekly, daily)
    alignment = [weekly.get("bias"), daily.get("bias"), intraday.get("bias")]
    directional_count = sum(1 for bias in alignment if bias in {BULLISH, BEARISH})
    aligned_count = (
        alignment.count(higher_bias) if higher_bias in {BULLISH, BEARISH} else 0
    )
    agreement_score = _clamp_score(
        (weekly["score"] + daily["score"] + intraday["score"]) / 3
    )

    return {
        "schema": "sharpedge.timeframe_agreement.v1",
        "higher_timeframe_bias": higher_bias,
        "agreement_state": f"{higher_bias}_{intraday.get('stance')}",
        "agreement_score": agreement_score,
        "alignment": {
            "directional_count": directional_count,
            "aligned_count": aligned_count,
        },
        "summary": _summary(pa, weekly, daily, intraday),
        "timeframes": {
            "weekly": weekly,
            "daily": daily,
            "intraday": intraday,
        },
    }


__all__ = [
    "build_daily_timeframe",
    "build_intraday_timeframe",
    "build_timeframe_agreement",
    "build_weekly_timeframe",
]
