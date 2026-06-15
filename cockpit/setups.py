"""Setup detection for the SharpEdge cockpit.

Detects the trades the operator actually takes:
  - FAILED BREAKDOWN (bear trap) -> reclaim of a support level -> CALLS bias
  - FAILED BREAKOUT  (bull trap) -> rejection of a resistance level -> PUTS bias
  - EXHAUSTION at the extremes (volume climax / VWAP overextension /
    momentum deceleration / wick rejection)

Everything is computed from 1-min OHLCV bars + reference levels. Each result
is a dict the cockpit renders as a number-backed card. Detection is
deliberately conservative and "recent" (only flags setups still actionable).
"""

from __future__ import annotations

import datetime as dt

import requests

UA = {"User-Agent": "Mozilla/5.0"}
DAILY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?interval=1d&range=5d"
)
RECENT_BARS = 6        # a setup is "actionable" only if it triggered this recently
OR_MINUTES = 30        # opening-range window


def buffer(price):
    """Level-break tolerance: ~3 bps of price, floor 10 cents."""
    return max(0.10, price * 0.0003)


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
        r = requests.get(DAILY_URL, headers=UA, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        highs = [h for h in q["high"] if h is not None]
        lows = [low for low in q["low"] if low is not None]
        closes = [c for c in q["close"] if c is not None]
        if len(closes) < 2:
            return {}
        # second to last completed day = prior day
        return {"PDH": highs[-2], "PDL": lows[-2], "PDC": closes[-2]}
    except Exception:
        return {}


def reference_levels(bars):
    orh, orl = opening_range(bars)
    levels = {"ORH": orh, "ORL": orl}
    levels.update(prior_day())
    return levels


# ----------------------------- failed breaks -----------------------------
def _failed_breakdown(bars, level, name):
    """Price broke BELOW level then reclaimed it recently -> bull trap."""
    if level is None:
        return None
    buf = buffer(level)
    broke_idx = None
    deepest = level
    for i, (_, _, _, low, _, _) in enumerate(bars):
        if low < level - buf:
            broke_idx = i
            deepest = min(deepest, low)
    if broke_idx is None:
        return None
    # reclaimed = a later close back above the level, and still above now
    last_close = bars[-1][4]
    if last_close <= level:
        return None
    # find the bar where it reclaimed (first close above level after break)
    reclaim_idx = None
    for i in range(broke_idx, len(bars)):
        if bars[i][4] > level:
            reclaim_idx = i
            break
    if reclaim_idx is None:
        return None
    bars_ago = len(bars) - 1 - reclaim_idx
    if bars_ago > RECENT_BARS:
        return None  # old news, not actionable now
    depth = (level - deepest) / level * 100
    return {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "detail": (f"reclaimed {name} ${level:.2f} {bars_ago}m ago after "
                   f"stabbing ${deepest:.2f} (-{depth:.2f}% below) - bear trap"),
        "score": depth + (RECENT_BARS - bars_ago),
    }


def _failed_breakout(bars, level, name):
    """Price broke ABOVE level then rejected back below recently -> bull trap."""
    if level is None:
        return None
    buf = buffer(level)
    broke_idx = None
    highest = level
    for i, (_, _, high, _, _, _) in enumerate(bars):
        if high > level + buf:
            broke_idx = i
            highest = max(highest, high)
    if broke_idx is None:
        return None
    last_close = bars[-1][4]
    if last_close >= level:
        return None
    reject_idx = None
    for i in range(broke_idx, len(bars)):
        if bars[i][4] < level:
            reject_idx = i
            break
    if reject_idx is None:
        return None
    bars_ago = len(bars) - 1 - reject_idx
    if bars_ago > RECENT_BARS:
        return None
    ext = (highest - level) / level * 100
    return {
        "tag": "FAILED BREAKOUT",
        "bias": "PUTS (bearish)",
        "kind": "bad",
        "detail": (f"rejected {name} ${level:.2f} {bars_ago}m ago after "
                   f"poking ${highest:.2f} (+{ext:.2f}% above) - bull trap"),
        "score": ext + (RECENT_BARS - bars_ago),
    }


def detect_failed_breaks(bars, levels):
    out = []
    for name in ("ORL", "PDL"):
        r = _failed_breakdown(bars, levels.get(name), name)
        if r:
            out.append(r)
    for name in ("ORH", "PDH"):
        r = _failed_breakout(bars, levels.get(name), name)
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
    rng_pos = pa["rng_pos"]
    at_low = rng_pos <= 22
    at_high = rng_pos >= 78
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
            signals.append(f"long lower wick ({lower_wick*100:.0f}% of bar)")
        if abs(pa["vs_vwap"]) >= 0.4:
            signals.append(f"stretched {pa['vs_vwap']:+.2f}% from VWAP")
        if len(signals) >= 2:
            out.append({
                "tag": "DOWNSIDE EXHAUSTION",
                "bias": "watch for reversal UP (calls)",
                "kind": "warn",
                "detail": "at day lows + " + " + ".join(signals),
                "score": 50,
            })
    if at_high:
        if upper_wick >= 0.5:
            signals.append(f"long upper wick ({upper_wick*100:.0f}% of bar)")
        if abs(pa["vs_vwap"]) >= 0.4:
            signals.append(f"stretched {pa['vs_vwap']:+.2f}% from VWAP")
        if len(signals) >= 2:
            out.append({
                "tag": "UPSIDE EXHAUSTION",
                "bias": "watch for reversal DOWN (puts)",
                "kind": "warn",
                "detail": "at day highs + " + " + ".join(signals),
                "score": 50,
            })
    return out
