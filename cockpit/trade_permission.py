"""Discretionary-to-programmatic trade permission scoring.

This module turns the operator's SPY tape-reading checklist into explicit,
explainable scores. It does not submit orders. It does not pretend to know the
future. It answers one boring-but-useful question:

    Are current conditions clean enough to consider a trade?

Inputs are the cockpit's existing OHLCV bars, price-action dict, reference
levels, setup cards, options read, gamma profile, and magnitude read.
"""

from __future__ import annotations

from dataclasses import dataclass

BULLISH = 1
BEARISH = -1
NEUTRAL = 0


@dataclass(frozen=True)
class ScorePart:
    score: int
    bias: int = NEUTRAL
    reason: str = ""


def _clamp(value, low=0, high=100):
    return int(max(low, min(high, round(value))))


def _safe_pct(num, den):
    return num / den * 100 if den else 0.0



def _buffer(price):
    return max(0.10, price * 0.0003) if price else 0.10


def _last_close(bars):
    return bars[-1][4] if bars else 0.0


def _last_minute(bars):
    return int(bars[-1][0]) if bars else 0


def _bar_personality(bar):
    _minute, open_, high, low, close, _volume = bar
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    close_pos = (close - low) / rng
    return body, upper_wick, lower_wick, close_pos


def _nearest_level(spot, levels):
    clean = {name: value for name, value in levels.items() if value is not None}
    if not clean or not spot:
        return None, None, None
    name, value = min(clean.items(), key=lambda item: abs(item[1] - spot))
    return name, value, abs(value - spot) / spot * 100


def _level_map(pa, levels, op):
    mapped = dict(levels or {})
    if pa.get("vwap"):
        mapped["VWAP"] = pa["vwap"]
    if op.get("call_wall"):
        mapped["CALL_WALL"] = op["call_wall"]
    if op.get("put_wall"):
        mapped["PUT_WALL"] = op["put_wall"]
    return mapped


def _recent_closes(bars, n=3):
    return [bar[4] for bar in bars[-n:]]


def _ema(values, length=20):
    if not values:
        return 0.0
    alpha = 2 / (length + 1)
    value = values[0]
    for item in values[1:]:
        value = item * alpha + value * (1 - alpha)
    return value


def _swing_points(bars, window=2):
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


def _score_structure(bars):
    highs, lows = _swing_points(bars)
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1][1] > highs[-2][1]
        higher_low = lows[-1][1] > lows[-2][1]
        lower_high = highs[-1][1] < highs[-2][1]
        lower_low = lows[-1][1] < lows[-2][1]
        if higher_high and higher_low:
            return ScorePart(82, BULLISH, "HH/HL structure intact")
        if lower_high and lower_low:
            return ScorePart(82, BEARISH, "LH/LL structure intact")
        return ScorePart(46, NEUTRAL, "mixed swing structure")

    if len(bars) < 8:
        return ScorePart(45, NEUTRAL, "not enough bars for structure")
    first = bars[: len(bars) // 2]
    last = bars[len(bars) // 2 :]
    hi_up = max(bar[2] for bar in last) > max(bar[2] for bar in first)
    lo_up = min(bar[3] for bar in last) > min(bar[3] for bar in first)
    hi_down = max(bar[2] for bar in last) < max(bar[2] for bar in first)
    lo_down = min(bar[3] for bar in last) < min(bar[3] for bar in first)
    if hi_up and lo_up:
        return ScorePart(70, BULLISH, "session halves show higher high/higher low")
    if hi_down and lo_down:
        return ScorePart(70, BEARISH, "session halves show lower high/lower low")
    return ScorePart(45, NEUTRAL, "range structure, no clean sequence")


def _score_acceptance(bars, pa, levels):
    closes = _recent_closes(bars, 3)
    if len(closes) < 3:
        return ScorePart(45, NEUTRAL, "need 3 closes for acceptance")
    candidates = sorted(levels.items(), key=lambda item: abs(item[1] - closes[-1]))
    for name, level in candidates:
        buf = _buffer(level)
        if all(close > level + buf for close in closes):
            return ScorePart(78, BULLISH, f"3 closes accepted above {name} {level:.2f}")
        if all(close < level - buf for close in closes):
            return ScorePart(78, BEARISH, f"3 closes accepted below {name} {level:.2f}")
    if pa.get("vs_vwap", 0) > 0.05:
        return ScorePart(60, BULLISH, "above VWAP, but acceptance is not stacked")
    if pa.get("vs_vwap", 0) < -0.05:
        return ScorePart(60, BEARISH, "below VWAP, but acceptance is not stacked")
    return ScorePart(35, NEUTRAL, "no clean level acceptance")


def _score_rejection(bars, setups):
    for setup in setups or []:
        tag = setup.get("tag", "")
        if tag == "FAILED BREAKDOWN":
            return ScorePart(88, BULLISH, "bear trap: support break reclaimed")
        if tag == "FAILED BREAKOUT":
            return ScorePart(88, BEARISH, "bull trap: resistance break rejected")
    if not bars:
        return ScorePart(35, NEUTRAL, "no bar data for rejection")
    body, upper_wick, lower_wick, close_pos = _bar_personality(bars[-1])
    if lower_wick > body * 2 and close_pos > 0.6:
        return ScorePart(70, BULLISH, "last candle rejected lower prices")
    if upper_wick > body * 2 and close_pos < 0.4:
        return ScorePart(70, BEARISH, "last candle rejected higher prices")
    return ScorePart(35, NEUTRAL, "no obvious rejection/trap")


def _score_trend(bars, pa):
    if len(bars) < 6:
        return ScorePart(45, NEUTRAL, "need more bars for trend")
    closes = [bar[4] for bar in bars]
    recent = closes[-6:]
    slope = recent[-1] - recent[0]
    vs_vwap = pa.get("vs_vwap", 0)
    mom15 = pa.get("mom15", 0)
    if vs_vwap > 0.05 and mom15 > 0 and slope > 0:
        return ScorePart(82, BULLISH, "above VWAP with positive short-term momentum")
    if vs_vwap < -0.05 and mom15 < 0 and slope < 0:
        return ScorePart(82, BEARISH, "below VWAP with negative short-term momentum")
    if abs(vs_vwap) <= 0.05:
        return ScorePart(38, NEUTRAL, "hugging VWAP: chop risk")
    bias = BULLISH if vs_vwap > 0 else BEARISH
    side = "above" if bias == BULLISH else "below"
    return ScorePart(58, bias, f"price is {side} VWAP, momentum not fully aligned")


def _score_volume(pa):
    mult = pa.get("vol_mult", 0)
    if mult >= 1.5:
        return ScorePart(85, NEUTRAL, f"volume confirms move at {mult:.1f}x normal")
    if mult >= 1.0:
        return ScorePart(62, NEUTRAL, f"volume acceptable at {mult:.1f}x normal")
    if mult >= 0.7:
        return ScorePart(42, NEUTRAL, f"volume thin at {mult:.1f}x normal")
    return ScorePart(25, NEUTRAL, f"volume missing at {mult:.1f}x normal")


def _score_location(pa, levels):
    spot = pa.get("spot") or 0
    name, value, dist = _nearest_level(spot, levels)
    rng_pos = pa.get("rng_pos", 50)
    if name and dist is not None and dist <= 0.08:
        return ScorePart(82, NEUTRAL, f"at decision level {name} {value:.2f}")
    if name and dist is not None and dist <= 0.20:
        return ScorePart(68, NEUTRAL, f"near {name} {value:.2f} ({dist:.2f}% away)")
    if rng_pos >= 80 or rng_pos <= 20:
        return ScorePart(58, NEUTRAL, f"at range extreme ({rng_pos:.0f}% of day range)")
    return ScorePart(34, NEUTRAL, "middle of nowhere; location is not helping")


def _score_pressure(setups):
    for setup in setups or []:
        tag = setup.get("tag", "")
        if tag == "FAILED BREAKDOWN":
            return ScorePart(86, BULLISH, "sellers trapped below support")
        if tag == "FAILED BREAKOUT":
            return ScorePart(86, BEARISH, "buyers trapped above resistance")
        if tag == "DOWNSIDE EXHAUSTION":
            return ScorePart(64, BULLISH, "downside pressure may be exhausted")
        if tag == "UPSIDE EXHAUSTION":
            return ScorePart(64, BEARISH, "upside pressure may be exhausted")
    return ScorePart(35, NEUTRAL, "no clear trapped side")


def _score_time_of_day(bars):
    minute = _last_minute(bars)
    clock = minute + 570
    if minute < 30:
        return ScorePart(52, NEUTRAL, "opening auction: price discovery")
    if minute < 120:
        return ScorePart(74, NEUTRAL, "morning continuation window")
    if 120 <= minute < 240:
        return ScorePart(42, NEUTRAL, "midday chop window")
    if minute >= 330:
        return ScorePart(68, NEUTRAL, "power hour positioning window")
    hour = clock // 60
    mins = clock % 60
    return ScorePart(58, NEUTRAL, f"neutral time window around {hour}:{mins:02d}")


def _score_volatility(op, magnitude):
    atm_iv = op.get("atm_iv") or 0
    premium = (magnitude or {}).get("premium_read")
    if not atm_iv:
        return ScorePart(50, NEUTRAL, "no volatility read")
    base = 62
    if atm_iv < 0.12:
        base = 52
        reason = "low IV: favor acceptance less, mean reversion more"
    elif atm_iv <= 0.28:
        base = 70
        reason = "normal/high-enough IV for intraday follow-through"
    else:
        base = 60
        reason = "very high IV: moves work, but slippage/whipsaw risk rises"
    if premium == "cheap":
        base += 8
        reason += "; realized move looks cheap vs implied"
    elif premium == "rich":
        base -= 5
        reason += "; options look rich vs realized move"
    return ScorePart(_clamp(base), NEUTRAL, reason)


def _score_candle(bars):
    if not bars:
        return ScorePart(45, NEUTRAL, "no candle data")
    body, upper_wick, lower_wick, close_pos = _bar_personality(bars[-1])
    if close_pos > 0.8 and lower_wick <= body:
        return ScorePart(76, BULLISH, "strong bull candle: close near high")
    if close_pos < 0.2 and upper_wick <= body:
        return ScorePart(76, BEARISH, "strong bear candle: close near low")
    if lower_wick > body * 2:
        return ScorePart(66, BULLISH, "lower-wick rejection candle")
    if upper_wick > body * 2:
        return ScorePart(66, BEARISH, "upper-wick rejection candle")
    return ScorePart(45, NEUTRAL, "ordinary candle personality")


def _score_exhaustion(bars, pa, levels):
    if not bars:
        return ScorePart(45, NEUTRAL, "no bars for exhaustion read")
    closes = [bar[4] for bar in bars]
    spot = pa.get("spot") or closes[-1]
    vwap = pa.get("vwap") or spot
    ema20 = _ema(closes[-20:], 20)
    rng_pos = pa.get("rng_pos", 50)
    dist_vwap = abs(spot - vwap) / spot * 100 if spot else 0
    dist_ema = abs(spot - ema20) / spot * 100 if spot and ema20 else 0
    orh = levels.get("ORH")
    orl = levels.get("ORL")
    orb_dist = min(
        [abs(spot - level) / spot * 100 for level in (orh, orl) if level],
        default=0.0,
    )
    stretched = dist_vwap >= 0.35 or dist_ema >= 0.25
    extreme = rng_pos >= 82 or rng_pos <= 18
    body, upper_wick, lower_wick, close_pos = _bar_personality(bars[-1])
    wick_reject = upper_wick > body * 1.8 or lower_wick > body * 1.8
    score = 35 + min(dist_vwap * 55, 22) + min(dist_ema * 70, 20)
    score += 12 if extreme else 0
    score += 10 if wick_reject else 0
    score -= 10 if orb_dist <= 0.12 else 0
    if stretched and rng_pos >= 82:
        return ScorePart(
            _clamp(score),
            BEARISH,
            f"stretched high: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
        )
    if stretched and rng_pos <= 18:
        return ScorePart(
            _clamp(score),
            BULLISH,
            f"stretched low: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
        )
    return ScorePart(
        _clamp(score),
        NEUTRAL,
        f"not exhausted: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
    )


def _score_trap(bars, levels, setups):
    for setup in setups or []:
        tag = setup.get("tag", "")
        if tag == "FAILED BREAKDOWN":
            return ScorePart(92, BULLISH, "failed breakdown probability high")
        if tag == "FAILED BREAKOUT":
            return ScorePart(92, BEARISH, "failed breakout probability high")
    if not bars:
        return ScorePart(35, NEUTRAL, "no bars for trap read")
    close = bars[-1][4]
    for name, level in (levels or {}).items():
        if name not in {"ORH", "ORL", "PDH", "PDL"} or level is None:
            continue
        buf = _buffer(level)
        recent = bars[-6:]
        broke_up = any(bar[2] > level + buf for bar in recent)
        broke_down = any(bar[3] < level - buf for bar in recent)
        if name in {"ORH", "PDH"} and broke_up and close < level:
            return ScorePart(78, BEARISH, f"buyers trapped above {name} {level:.2f}")
        if name in {"ORL", "PDL"} and broke_down and close > level:
            return ScorePart(78, BULLISH, f"sellers trapped below {name} {level:.2f}")
    return ScorePart(35, NEUTRAL, "no failed-break trap detected")


def _score_dealer_gamma(pa, op, gp, magnitude):
    spot = pa.get("spot") or 0
    regime = (gp or {}).get("regime")
    pin = (gp or {}).get("pin")
    premium = (magnitude or {}).get("premium_read")
    pin_dist = abs(spot - pin) / spot * 100 if spot and pin else None
    call_wall = op.get("call_wall")
    put_wall = op.get("put_wall")
    wall_bias = NEUTRAL
    wall_reason = ""
    if spot and call_wall and abs(spot - call_wall) / spot * 100 <= 0.20:
        wall_bias = BEARISH
        wall_reason = f"near call wall {call_wall:g}; upside resistance"
    elif spot and put_wall and abs(spot - put_wall) / spot * 100 <= 0.20:
        wall_bias = BULLISH
        wall_reason = f"near put wall {put_wall:g}; downside support"
    if regime == "positive" and pin_dist is not None and pin_dist <= 0.25:
        return ScorePart(38, wall_bias, f"pinning near gamma pin {pin:g}; {wall_reason or 'chop risk'}")
    if regime == "negative":
        score = 72 if premium == "cheap" else 62
        reason = "negative gamma: expansion/squeeze risk"
        if pin_dist is not None:
            reason += f", pin {pin_dist:.2f}% away"
        if wall_reason:
            reason += f"; {wall_reason}"
        return ScorePart(score, wall_bias, reason)
    if pin_dist is not None:
        return ScorePart(55, wall_bias, f"dealer gravity neutral; pin {pin_dist:.2f}% away")
    return ScorePart(50, wall_bias, "no dealer/gamma read")


def _score_regime(bars, pa, setups):
    if len(bars) < 10:
        return ScorePart(45, NEUTRAL, "need more bars for regime")
    vs_vwap = pa.get("vs_vwap", 0)
    mom15 = pa.get("mom15", 0)
    vol_mult = pa.get("vol_mult", 0)
    rng_pos = pa.get("rng_pos", 50)
    closes = [bar[4] for bar in bars]
    first_half = closes[: len(closes) // 2]
    second_half = closes[len(closes) // 2 :]
    drift = _safe_pct(second_half[-1] - first_half[0], first_half[0])
    has_trap = any((setup.get("tag") or "").startswith("FAILED") for setup in setups or [])
    if abs(vs_vwap) <= 0.05 and abs(mom15) < 0.08:
        return ScorePart(38, NEUTRAL, "balance day: VWAP magnet, mean reversion likely")
    if has_trap:
        bias = BULLISH if any(s.get("tag") == "FAILED BREAKDOWN" for s in setups or []) else BEARISH
        return ScorePart(70, bias, "mean-reversion/trap regime")
    if vol_mult >= 1.0 and abs(drift) >= 0.35 and (rng_pos >= 70 or rng_pos <= 30):
        bias = BULLISH if vs_vwap > 0 else BEARISH
        return ScorePart(82, bias, "trend day regime: VWAP control + directional drift")
    if rng_pos >= 78 or rng_pos <= 22:
        bias = BULLISH if rng_pos <= 22 else BEARISH
        return ScorePart(58, bias, "range extreme: continuation needs proof, fade risk exists")
    return ScorePart(48, NEUTRAL, "unclear regime; do not overpay for mediocre reads")


def _score_opening_auction(bars, levels):
    pdc = levels.get("PDC")
    if not pdc or not bars:
        return ScorePart(50, NEUTRAL, "no prior close for gap read")
    open_ = bars[0][1]
    close = _last_close(bars)
    gap_pct = _safe_pct(open_ - pdc, pdc)
    if abs(gap_pct) < 0.15:
        return ScorePart(55, NEUTRAL, f"flat open ({gap_pct:+.2f}% gap)")
    if gap_pct > 0 and close > open_:
        return ScorePart(72, BULLISH, f"gap up accepting ({gap_pct:+.2f}%)")
    if gap_pct > 0 and close < pdc:
        return ScorePart(36, BEARISH, f"gap up failed and filled ({gap_pct:+.2f}%)")
    if gap_pct < 0 and close < open_:
        return ScorePart(72, BEARISH, f"gap down accepting ({gap_pct:+.2f}%)")
    if gap_pct < 0 and close > pdc:
        return ScorePart(66, BULLISH, f"gap down reclaimed ({gap_pct:+.2f}%)")
    return ScorePart(48, NEUTRAL, f"gap is unresolved ({gap_pct:+.2f}%)")


def _bias_label(score):
    if score >= 0.25:
        return "CALLS"
    if score <= -0.25:
        return "PUTS"
    return "NEUTRAL"


def _gate(score):
    if score >= 72:
        return "PERMIT"
    if score >= 58:
        return "CAUTION"
    return "BLOCK"


def _weighted_score(parts):
    setup_score = max(
        parts["acceptance_score"].score,
        parts["rejection_score"].score,
        parts["pressure_score"].score,
    )
    weighted = (
        parts["structure_score"].score * 0.08
        + parts["trend_score"].score * 0.10
        + setup_score * 0.16
        + parts["volume_score"].score * 0.08
        + parts["location_score"].score * 0.10
        + parts["time_of_day_score"].score * 0.05
        + parts["volatility_score"].score * 0.05
        + parts["candle_score"].score * 0.05
        + parts["opening_auction_score"].score * 0.04
        + parts["exhaustion_score"].score * 0.09
        + parts["trap_score"].score * 0.10
        + parts["dealer_gamma_score"].score * 0.05
        + parts["regime_score"].score * 0.05
    )
    return _clamp(weighted)


def _weighted_bias(parts):
    weights = {
        "structure_score": 0.10,
        "acceptance_score": 0.12,
        "rejection_score": 0.14,
        "trend_score": 0.12,
        "pressure_score": 0.14,
        "candle_score": 0.08,
        "opening_auction_score": 0.07,
        "exhaustion_score": 0.08,
        "trap_score": 0.10,
        "dealer_gamma_score": 0.07,
        "regime_score": 0.10,
    }
    total = 0.0
    weight_total = 0.0
    for name, weight in weights.items():
        part = parts[name]
        total += part.bias * weight * (part.score / 100)
        weight_total += weight
    return total / max(weight_total, 1e-9)


def _reasons(parts):
    ordered = sorted(parts.items(), key=lambda item: item[1].score, reverse=True)
    best = [
        f"{name.replace('_score', '')}: {part.reason}" for name, part in ordered[:3]
    ]
    worst = [
        f"{name.replace('_score', '')}: {part.reason}" for name, part in ordered[-2:]
    ]
    return {"supporting": best, "warnings": worst}


def _serialize(parts):
    return {
        name: {
            "score": part.score,
            "bias": _bias_label(part.bias),
            "reason": part.reason,
        }
        for name, part in parts.items()
    }


def score_trade_permission(
    bars,
    pa,
    levels,
    setups=None,
    op=None,
    gp=None,
    magnitude=None,
):
    """Return an explainable trade-permission card for cockpit + signal.json."""
    op = op or {}
    gp = gp or {}
    full_levels = _level_map(pa or {}, levels or {}, op)
    parts = {
        "structure_score": _score_structure(bars),
        "acceptance_score": _score_acceptance(bars, pa or {}, full_levels),
        "rejection_score": _score_rejection(bars, setups),
        "trend_score": _score_trend(bars, pa or {}),
        "volume_score": _score_volume(pa or {}),
        "location_score": _score_location(pa or {}, full_levels),
        "pressure_score": _score_pressure(setups),
        "time_of_day_score": _score_time_of_day(bars),
        "volatility_score": _score_volatility(op, magnitude or {}),
        "candle_score": _score_candle(bars),
        "opening_auction_score": _score_opening_auction(bars, full_levels),
        "exhaustion_score": _score_exhaustion(bars, pa or {}, full_levels),
        "trap_score": _score_trap(bars, full_levels, setups),
        "dealer_gamma_score": _score_dealer_gamma(pa or {}, op, gp, magnitude or {}),
        "regime_score": _score_regime(bars, pa or {}, setups),
    }
    permission = _weighted_score(parts)
    bias_value = _weighted_bias(parts)
    bias = _bias_label(bias_value)
    reasons = _reasons(parts)
    return {
        "schema": "sharpedge.trade_permission.v1",
        "trade_permission_score": permission,
        "trade_gate": _gate(permission),
        "bias": bias,
        "bias_strength": round(abs(bias_value), 3),
        "scores": _serialize(parts),
        "supporting_reasons": reasons["supporting"],
        "warning_reasons": reasons["warnings"],
    }


__all__ = ["score_trade_permission"]
