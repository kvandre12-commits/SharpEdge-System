"""Setup detection for the SharpEdge cockpit.

Detects the trades the operator actually takes:
  - FAILED BREAKDOWN (bear trap) -> reclaim of a support level -> CALLS bias
  - FAILED BREAKOUT  (bull trap) -> rejection of a resistance level -> PUTS bias
  - EXHAUSTION at the extremes (volume climax / VWAP overextension /
    momentum deceleration / wick rejection)
  - POST-SELLOFF COIL (impulse down -> volatility compression -> tight channel)

Everything is computed from 1-min OHLCV bars + reference levels. Each result
is a dict the cockpit renders as a number-backed card. Detection is
deliberately conservative and "recent" (only flags setups still actionable).
"""

from __future__ import annotations

from level_state_engine import build_level_state_map
from market_data_sources import fetch_yahoo_prior_day_levels
from range_posture import build_range_posture
from vwap_posture import build_vwap_posture

RECENT_BARS = 6  # a setup is "actionable" only if it triggered this recently
OR_MINUTES = 30  # opening-range window
AGGREGATE_MINUTES = 5  # coil detector runs on 5m bars built from 1m tape
BB_WINDOW = 20
BB_WIDTH_HISTORY = 50
CHANNEL_LOOKBACK = 8
IMPULSE_LOOKBACK = 8
IMPULSE_SEARCH = 24
COMPRESSION_QUANTILE = 0.2
NARROW_CHANNEL_PCT = 0.003
IMPULSE_DOWN_PCT = 0.007
CONTINUATION_MIN_VS_VWAP = 0.05
CONTINUATION_MIN_MOM = 0.05
CONTINUATION_MIN_VOL_MULT = 1.2
WALL_PROXIMITY_PCT = 0.20
PIN_PROXIMITY_PCT = 0.10
EXHAUSTION_STRETCH_MIN_PCT = 0.40
HANDOFF_LOOKBACK_BARS = 45
HANDOFF_ACCEPTANCE_WINDOW = 3
HANDOFF_MIN_ACCEPTANCE_CLOSES = 2
HANDOFF_MIN_RNG_POS = 45


# ----------------------------- reference levels -----------------------------
def opening_range(bars):
    """(ORH, ORL) from the first OR_MINUTES of the session. bars: (m,o,h,l,c,v)."""
    early = [b for b in bars if b[0] < OR_MINUTES]
    if not early:
        early = bars[:6]
    orh = max(b[2] for b in early)
    orl = min(b[3] for b in early)
    return orh, orl


def prior_day():
    """Prior-day high/low/close. Returns {} on any failure (fail-soft)."""
    try:
        levels, _source = fetch_yahoo_prior_day_levels("SPY")
        return levels
    except Exception:
        return {}


def reference_levels(bars):
    orh, orl = opening_range(bars)
    levels = {"ORH": orh, "ORL": orl}
    levels.update(prior_day())
    return levels


# ----------------------------- failed breaks -----------------------------
def _failed_breakdown(level_state):
    """Price broke BELOW support then reclaimed it recently -> bull trap."""
    if not level_state or level_state.get("event_state") != "failed_break_reclaimed":
        return None
    facts = level_state.get("facts") or {}
    reclaim_idx = facts.get("reclaim_above_level_index")
    bars_ago = facts.get("bars_since_reclaim_above_level")
    deepest = facts.get("breach_below_deepest_low")
    depth = facts.get("breach_below_depth_pct")
    level = level_state.get("level_price")
    name = level_state.get("level_name")
    if (
        reclaim_idx is None
        or bars_ago is None
        or bars_ago > RECENT_BARS
        or deepest is None
        or depth is None
        or level is None
    ):
        return None
    return {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": (
            f"reclaimed {name} ${level:.2f} {bars_ago}m ago after "
            f"stabbing ${deepest:.2f} (-{depth:.2f}% below) - bear trap"
        ),
        "score": depth + (RECENT_BARS - bars_ago),
        "level_name": name,
        "level_price": round(level, 2),
        "trigger_price": round(deepest, 2),
        "bars_ago": bars_ago,
    }


def _failed_breakout(level_state):
    """Price broke ABOVE resistance then rejected back below recently -> bull trap."""
    if not level_state or level_state.get("event_state") != "failed_break_rejected":
        return None
    facts = level_state.get("facts") or {}
    reject_idx = facts.get("reject_below_level_index")
    bars_ago = facts.get("bars_since_reject_below_level")
    highest = facts.get("breach_above_highest_high")
    ext = facts.get("breach_above_extension_pct")
    level = level_state.get("level_price")
    name = level_state.get("level_name")
    if (
        reject_idx is None
        or bars_ago is None
        or bars_ago > RECENT_BARS
        or highest is None
        or ext is None
        or level is None
    ):
        return None
    return {
        "tag": "FAILED BREAKOUT",
        "bias": "PUTS (bearish)",
        "kind": "bad",
        "detail": (
            f"rejected {name} ${level:.2f} {bars_ago}m ago after "
            f"poking ${highest:.2f} (+{ext:.2f}% above) - bull trap"
        ),
        "score": ext + (RECENT_BARS - bars_ago),
        "level_name": name,
        "level_price": round(level, 2),
        "trigger_price": round(highest, 2),
        "bars_ago": bars_ago,
    }


def detect_failed_breaks(bars, levels):
    """Return canonical failed-break setup-event identity cards.

    Contract doctrine:
    - this is the canonical setup detector for FAILED BREAKDOWN / FAILED
      BREAKOUT identity in the live cockpit pipeline.
    - downstream vector/authority layers may corroborate or ignore that setup,
      but they should not redefine what setup card was emitted here.
    """
    out = []
    level_states = build_level_state_map(
        bars,
        levels,
        level_names=("ORL", "PDL", "ORH", "PDH"),
        recent_window=RECENT_BARS,
    )
    for name in ("ORL", "PDL"):
        r = _failed_breakdown(level_states.get(name))
        if r:
            out.append(r)
    for name in ("ORH", "PDH"):
        r = _failed_breakout(level_states.get(name))
        if r:
            out.append(r)
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


# ----------------------------- exhaustion -----------------------------
def detect_exhaustion(bars, pa):
    """Flag possible exhaustion at the day's extremes. Returns list of cards."""
    out = []
    closes = [b[4] for b in bars]
    vols = [b[5] for b in bars]
    posture = build_range_posture(pa)
    side = str(posture.get("side") or "center")
    at_low = bool(posture.get("is_pressing_edge")) and side == "downside"
    at_high = bool(posture.get("is_pressing_edge")) and side == "upside"
    abs_vs_vwap = float(posture.get("abs_vs_vwap_pct") or 0.0)
    if not (at_low or at_high):
        return out

    # signals
    body = sorted(vols)
    med = body[len(body) // 2] or 1
    climax = (vols[-1] / med) if med else 0
    o, h, low, c = bars[-1][1], bars[-1][2], bars[-1][3], bars[-1][4]
    rng = (h - low) or 1e-9
    lower_wick = (min(o, c) - low) / rng
    upper_wick = (h - max(o, c)) / rng

    # momentum deceleration: last 5 vs prior 5
    def rate(seq):
        return (seq[-1] / seq[0] - 1) * 100 if len(seq) > 1 and seq[0] else 0

    r_now = rate(closes[-5:])
    r_prev = rate(closes[-10:-5]) if len(closes) >= 10 else r_now
    decel = abs(r_now) < abs(r_prev) * 0.6

    signals = []
    if climax >= 2.5:
        signals.append(f"volume climax {climax:.1f}x")
    if decel:
        signals.append(f"momentum fading ({r_prev:+.2f}%->{r_now:+.2f}%)")

    if at_low:
        if lower_wick >= 0.5:
            signals.append(f"long lower wick ({lower_wick * 100:.0f}% of bar)")
        if abs_vs_vwap >= EXHAUSTION_STRETCH_MIN_PCT:
            signals.append(f"stretched {pa['vs_vwap']:+.2f}% from VWAP")
        if len(signals) >= 2:
            out.append(
                {
                    "tag": "DOWNSIDE EXHAUSTION",
                    "bias": "watch for reversal UP (calls)",
                    "kind": "warn",
                    "detail": "at day lows + " + " + ".join(signals),
                    "score": 50,
                }
            )
    if at_high:
        if upper_wick >= 0.5:
            signals.append(f"long upper wick ({upper_wick * 100:.0f}% of bar)")
        if abs_vs_vwap >= EXHAUSTION_STRETCH_MIN_PCT:
            signals.append(f"stretched {pa['vs_vwap']:+.2f}% from VWAP")
        if len(signals) >= 2:
            out.append(
                {
                    "tag": "UPSIDE EXHAUSTION",
                    "bias": "watch for reversal DOWN (puts)",
                    "kind": "warn",
                    "detail": "at day highs + " + " + ".join(signals),
                    "score": 50,
                }
            )
    return out


# ----------------------------- volatility compression -----------------------------
def _aggregate_bars(bars, group_size=AGGREGATE_MINUTES):
    grouped = []
    for start in range(0, len(bars), group_size):
        chunk = bars[start : start + group_size]
        if len(chunk) < group_size:
            continue
        grouped.append(
            (
                chunk[-1][0],
                chunk[0][1],
                max(bar[2] for bar in chunk),
                min(bar[3] for bar in chunk),
                chunk[-1][4],
                sum(bar[5] for bar in chunk),
            )
        )
    return grouped


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _stdev(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5


def _quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _bollinger_widths(closes, window=BB_WINDOW):
    widths = []
    for end in range(window, len(closes) + 1):
        sample = closes[end - window : end]
        middle = _mean(sample)
        if not middle:
            widths.append(0.0)
            continue
        width = (4 * _stdev(sample)) / middle
        widths.append(width)
    return widths


def read_volatility_structure(bars, pa=None):
    """Describe post-impulse volatility state on 5m bars built from 1m tape."""
    agg = _aggregate_bars(bars)
    state = {
        "timeframe": f"{AGGREGATE_MINUTES}m",
        "source_bars": len(bars),
        "aggregate_bars": len(agg),
        "volatility_state": "unknown",
        "structure_state": "unknown",
        "bias": "neutral",
        "compression": False,
        "narrow_channel": False,
        "impulse_down": False,
        "coil": False,
    }
    if len(agg) < CHANNEL_LOOKBACK + IMPULSE_LOOKBACK:
        return state

    closes = [bar[4] for bar in agg]
    channel = agg[-CHANNEL_LOOKBACK:]
    channel_high = max(bar[2] for bar in channel)
    channel_low = min(bar[3] for bar in channel)
    channel_mid = (channel_high + channel_low) / 2
    last_close = closes[-1]
    channel_pct = (channel_high - channel_low) / last_close if last_close else 0.0
    narrow_channel = channel_pct < NARROW_CHANNEL_PCT

    prev_channel = agg[-2 * CHANNEL_LOOKBACK : -CHANNEL_LOOKBACK]
    prev_mid = (
        (max(bar[2] for bar in prev_channel) + min(bar[3] for bar in prev_channel)) / 2
        if prev_channel
        else channel_mid
    )
    channel_slope_pct = (
        ((channel_mid - prev_mid) / last_close * 100) if last_close else 0.0
    )

    impulse_window = agg[-(CHANNEL_LOOKBACK + IMPULSE_SEARCH) : -CHANNEL_LOOKBACK]
    if not impulse_window:
        impulse_window = agg[:-CHANNEL_LOOKBACK]
    impulse_anchor = (
        max(bar[4] for bar in impulse_window) if impulse_window else last_close
    )
    channel_floor = min(bar[4] for bar in channel)
    impulse_down_pct = (
        ((impulse_anchor - channel_floor) / impulse_anchor) if impulse_anchor else 0.0
    )
    impulse_down = impulse_down_pct > IMPULSE_DOWN_PCT

    widths = _bollinger_widths(closes)
    width_now = widths[-1] if widths else 0.0
    history = widths[-BB_WIDTH_HISTORY:] if widths else []
    width_threshold = _quantile(history, COMPRESSION_QUANTILE) if history else 0.0
    compression = bool(history) and width_now <= width_threshold
    width_slope = width_now - widths[-4] if len(widths) >= 4 else 0.0

    vwap_posture = build_vwap_posture(pa)
    below_vwap = bool(vwap_posture.get("has_downside_control"))
    above_vwap = bool(vwap_posture.get("has_upside_control"))

    if compression and narrow_channel:
        volatility_state = "squeeze"
    elif width_slope < -0.0005:
        volatility_state = "contraction"
    elif width_slope > 0.0005:
        volatility_state = "expansion"
    else:
        volatility_state = "normal"

    if impulse_down and compression and narrow_channel:
        structure_state = "channel_breakout_setup"
    elif narrow_channel:
        structure_state = "narrow_channel"
    elif impulse_down:
        structure_state = "pullback"
    else:
        structure_state = "trend"

    bias = "neutral"
    if impulse_down and (below_vwap or channel_slope_pct < 0):
        bias = "neutral_to_bearish"
    elif impulse_down and (above_vwap or channel_slope_pct > 0):
        bias = "neutral_to_bullish"

    state.update(
        {
            "volatility_state": volatility_state,
            "structure_state": structure_state,
            "bias": bias,
            "compression": compression,
            "bollinger_width_pct": round(width_now * 100, 3),
            "bollinger_width_p20_pct": round(width_threshold * 100, 3),
            "narrow_channel": narrow_channel,
            "channel_high": round(channel_high, 2),
            "channel_low": round(channel_low, 2),
            "channel_pct": round(channel_pct * 100, 3),
            "channel_slope_pct": round(channel_slope_pct, 3),
            "impulse_down": impulse_down,
            "prior_impulse_down_pct": round(impulse_down_pct * 100, 3),
            "coil": impulse_down and compression and narrow_channel,
            "trigger_high": round(channel_high, 2),
            "trigger_low": round(channel_low, 2),
        }
    )
    return state


def _distance_pct(spot, level):
    if not isinstance(spot, (int, float)) or spot <= 0:
        return None
    if not isinstance(level, (int, float)):
        return None
    return abs(spot - level) / spot * 100


def _near_wall(spot, op):
    for key, label in (("call_wall", "call wall"), ("put_wall", "put wall")):
        dist = _distance_pct(spot, op.get(key))
        if dist is not None and dist < WALL_PROXIMITY_PCT:
            return label, op.get(key), dist
    return None, None, None


def _recent_downside_exhaustion_pivot(bars):
    if len(bars) < HANDOFF_ACCEPTANCE_WINDOW + 2:
        return None
    lows = [bar[3] for bar in bars]
    highs = [bar[2] for bar in bars]
    vols = [bar[5] for bar in bars]
    session_low = min(lows)
    session_range = max(max(highs) - session_low, 1e-9)
    ordered = sorted(vols)
    med = ordered[len(ordered) // 2] or 1
    best = None
    start = max(0, len(bars) - HANDOFF_LOOKBACK_BARS)
    stop = len(bars) - HANDOFF_ACCEPTANCE_WINDOW
    for idx in range(start, stop):
        _minute, open_, high, low, close, volume = bars[idx]
        rng = (high - low) or 1e-9
        lower_wick = (min(open_, close) - low) / rng
        climax = (volume / med) if med else 0.0
        low_pos = ((low - session_low) / session_range) * 100
        if low_pos > 18:
            continue
        if lower_wick < 0.5 and climax < 2.0:
            continue
        bars_ago = len(bars) - 1 - idx
        candidate = {
            "low": low,
            "high": high,
            "close": close,
            "lower_wick": lower_wick,
            "climax": climax,
            "bars_ago": bars_ago,
            "score": (lower_wick * 100) + (climax * 10) - low_pos - bars_ago,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def detect_negative_gamma_continuation(pa, op, gp, bars=None):
    """Runner-style continuation card that mirrors the execution worldview.

    Conservative by design: negative gamma, above VWAP, positive 15m momentum,
    confirming volume, and clear of immediate wall friction. When a recent
    downside-exhaustion pivot has already reclaimed value, promote the read into
    an explicit handoff card so the operator does not manage a runner like a mere
    VWAP fade.
    """
    if (gp or {}).get("regime") != "negative":
        return None
    spot = float((pa or {}).get("spot") or 0)
    vwap_posture = build_vwap_posture(
        pa,
        bars,
        acceptance_window=HANDOFF_ACCEPTANCE_WINDOW,
        min_acceptance_closes=HANDOFF_MIN_ACCEPTANCE_CLOSES,
    )
    vs_vwap = float(vwap_posture.get("vs_vwap_pct") or 0)
    mom = float((pa or {}).get("mom15") or 0)
    vol = float((pa or {}).get("vol_mult") or 0)
    wall_label, wall_price, wall_dist = _near_wall(spot, op or {})
    if wall_label is not None:
        return None
    if not (
        bool(vwap_posture.get("has_upside_control"))
        and mom > CONTINUATION_MIN_MOM
        and vol >= CONTINUATION_MIN_VOL_MULT
    ):
        return None
    pin = (gp or {}).get("pin")
    pin_dist = _distance_pct(spot, pin)
    pivot = _recent_downside_exhaustion_pivot(bars or []) if bars else None
    accepted_above_vwap = bool(vwap_posture.get("accepted_above_vwap"))
    rng_pos = float((pa or {}).get("rng_pos") or 0)
    if (
        pivot
        and accepted_above_vwap
        and rng_pos >= HANDOFF_MIN_RNG_POS
        and spot > pivot["high"]
    ):
        pivot_notes = []
        if pivot["lower_wick"] >= 0.5:
            pivot_notes.append(f"wick {pivot['lower_wick'] * 100:.0f}%")
        if pivot["climax"] >= 2.0:
            pivot_notes.append(f"volume {pivot['climax']:.1f}x")
        detail = (
            f"recent downside exhaustion {pivot['bars_ago']}m ago near ${pivot['low']:.2f}"
            f" ({', '.join(pivot_notes)}) | now {vs_vwap:+.2f}% above VWAP"
            f" | 15m momentum {mom:+.2f}% | volume {vol:.1f}x confirms"
            f" | reclaimed pivot high ${pivot['high']:.2f}"
        )
        if pin_dist is not None:
            detail += f" | pin ${pin:g} is {pin_dist:.2f}% away"
        return {
            "tag": "EXHAUSTION -> RUNNER HANDOFF",
            "bias": "CALLS (reversal promoted to runner)",
            "kind": "ok",
            "detail": detail + " | fade has graduated into continuation",
            "score": 72 + min(int(round(vol * 5)), 10),
        }
    detail = (
        f"negative gamma continuation candidate | price {vs_vwap:+.2f}% above VWAP "
        f"| 15m momentum {mom:+.2f}% | volume {vol:.1f}x confirms"
    )
    if pin_dist is not None:
        detail += f" | pin ${pin:g} is {pin_dist:.2f}% away"
    return {
        "tag": "NEGATIVE GAMMA CONTINUATION",
        "bias": "CALLS (runner continuation)",
        "kind": "ok",
        "detail": detail + " | expansion odds > mean reversion odds",
        "score": 60 + min(int(round(vol * 5)), 10),
    }


def detect_sticky_noise(pa, op, gp):
    """Context card for sticky, low-quality tape that should not be confused
    with a real continuation setup.
    """
    if (gp or {}).get("regime") != "positive":
        return None
    spot = float((pa or {}).get("spot") or 0)
    vwap_posture = build_vwap_posture(pa)
    vs_vwap = float(vwap_posture.get("vs_vwap_pct") or 0)
    mom = float((pa or {}).get("mom15") or 0)
    vol = float((pa or {}).get("vol_mult") or 0)
    pin = (gp or {}).get("pin")
    wall_label, wall_price, wall_dist = _near_wall(spot, op or {})
    pin_dist = _distance_pct(spot, pin)

    conditions = []
    if not bool(vwap_posture.get("has_upside_control")):
        conditions.append(f"{vs_vwap:+.2f}% vs VWAP ({vwap_posture.get('state')})")
    if abs(mom) < CONTINUATION_MIN_MOM:
        conditions.append(f"15m momentum {mom:+.2f}% is flat")
    if vol < CONTINUATION_MIN_VOL_MULT:
        conditions.append(f"volume {vol:.1f}x is not confirming")
    if wall_label is not None:
        conditions.append(
            f"price is {wall_dist:.2f}% from {wall_label} ${wall_price:g}"
        )
    if pin_dist is not None and pin_dist < PIN_PROXIMITY_PCT:
        conditions.append(f"pin ${pin:g} is only {pin_dist:.2f}% away")

    if len(conditions) < 3:
        return None
    return {
        "tag": "STICKY NOISE",
        "bias": "stand down / mean reversion only",
        "kind": "warn",
        "detail": "positive gamma chop context | " + " | ".join(conditions),
        "score": 42 + len(conditions),
    }


def detect_volatility_coil(bars, pa=None, state=None):
    """Detect post-selloff compression that is waiting for channel resolution."""
    state = state or read_volatility_structure(bars, pa)
    if not state.get("coil"):
        return None
    width_pct = state.get("bollinger_width_pct", 0.0)
    width_floor = state.get("bollinger_width_p20_pct", 0.0)
    impulse_pct = state.get("prior_impulse_down_pct", 0.0)
    channel_pct = state.get("channel_pct", 0.0)
    channel_high = state.get("channel_high")
    channel_low = state.get("channel_low")
    vwap_note = ""
    if pa and pa.get("vs_vwap") is not None:
        vwap_note = f" | vs VWAP {pa['vs_vwap']:+.2f}%"
    score = 45 + min(int(round(impulse_pct * 10)), 18)
    if state.get("compression"):
        score += 5
    if state.get("narrow_channel"):
        score += 5
    return {
        "tag": "POST-SELLOFF COIL",
        "bias": "NEUTRAL-to-BEARISH | break high = reclaim, lose low = continuation",
        "kind": "warn",
        "detail": (
            f"5m compression after impulse down {impulse_pct:.2f}% | BB width {width_pct:.3f}% "
            f"vs p20 {width_floor:.3f}% | channel ${channel_low:.2f}-${channel_high:.2f} "
            f"({channel_pct:.3f}%) | trigger above ${channel_high:.2f} | "
            f"trigger below ${channel_low:.2f}{vwap_note}"
        ),
        "score": score,
    }
