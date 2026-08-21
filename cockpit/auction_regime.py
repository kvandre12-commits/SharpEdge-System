"""Canonical daily auction-regime classifier for SharpEdge.

This is the SINGLE SOURCE OF TRUTH for classifying a completed daily session
into an auction bucket. The logic here is ported verbatim from
``scripts/build_liquidity_regime_daily.py`` so the LIVE cockpit read and the
historical/backtest table (``liquidity_regime_events``) can never diverge.

Buckets: FAILED_BREAKDOWN, FAILED_BREAKOUT, CLEAN_BREAKOUT, CLEAN_BREAKDOWN,
RANGE_COMPRESSION, UNCLASSIFIED.

Two distinct meanings that must stay separated (see repo decisions):
  * prior_key_high / prior_key_low  -> today's TRUE key levels (Lane A)
  * regime_type / confidence / ratio -> the COMPLETED auction bucket, which is
    inherited forward as the NEXT session's context (Lane B)

``build_inherited_auction_context()`` computes Lane B live: it classifies the
most recent COMPLETED daily session and returns it as today's inherited bucket,
with a session_date proof + freshness so nobody mistakes stale data for live.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Optional

# Canonical thresholds — identical defaults to build_liquidity_regime_daily.py.
FAILED_MIN_RANGE_ATR = float(os.getenv("FAILED_MIN_RANGE_ATR", "1.25"))
CLEAN_MIN_RANGE_ATR = float(os.getenv("CLEAN_MIN_RANGE_ATR", "1.00"))
COMPRESSION_MAX_RANGE_ATR = float(os.getenv("COMPRESSION_MAX_RANGE_ATR", "0.75"))
ATR_LOOKBACK = int(os.getenv("ATR_LOOKBACK", "14"))

REGIME_STORY = {
    "FAILED_BREAKDOWN": (
        "Prior session swept below its low and reclaimed. Sellers failed. "
        "Inherited context leans constructive/long until the reclaim is lost."
    ),
    "FAILED_BREAKOUT": (
        "Prior session swept above its high and was rejected. Buyers failed. "
        "Inherited context leans heavy/short until the rejection is reclaimed."
    ),
    "CLEAN_BREAKOUT": (
        "Prior session broke and held above its prior high. Trend up carried. "
        "Inherited context favors continuation-up while support holds."
    ),
    "CLEAN_BREAKDOWN": (
        "Prior session broke and held below its prior low. Trend down carried. "
        "Inherited context favors continuation-down while resistance caps."
    ),
    "RANGE_COMPRESSION": (
        "Prior session compressed inside prior range. Coiled auction. "
        "Inherited context is expansion risk — direction unproven."
    ),
    "UNCLASSIFIED": "Prior session showed no strong auction signature. No inherited edge.",
}


# --------------------------- canonical math ---------------------------
def compute_true_range(prev_close: Optional[float], high: float, low: float) -> float:
    if prev_close is None:
        return float(high - low)
    return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))


def rolling_sma(values: list[float], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out


def classify_regime(
    prior_high: float,
    prior_low: float,
    o: float,
    h: float,
    l: float,
    c: float,
    tr: float,
    atr: Optional[float],
) -> tuple[str, dict[str, int], float, str]:
    """VERBATIM canonical classifier. Keep in sync with build_liquidity_regime_daily."""
    flags = {
        "broke_above_high": int(h > prior_high),
        "broke_below_low": int(l < prior_low),
        "failed_breakout": 0,
        "failed_breakdown": 0,
        "reclaimed_level": 0,
        "rejected_level": 0,
    }

    if atr is None or atr <= 0:
        return ("UNCLASSIFIED", flags, 0.0, "ATR unavailable; need lookback sessions")

    ratio = tr / atr

    broke_above = h > prior_high
    broke_below = l < prior_low
    close_back_below_prior_high = c < prior_high
    close_back_above_prior_low = c > prior_low

    reclaimed = broke_below and close_back_above_prior_low
    rejected = broke_above and close_back_below_prior_high

    flags["reclaimed_level"] = int(reclaimed)
    flags["rejected_level"] = int(rejected)

    notes: list[str] = []
    confidence = 0.0

    if reclaimed and ratio >= FAILED_MIN_RANGE_ATR:
        flags["failed_breakdown"] = 1
        confidence = 60.0
        if ratio >= 1.5:
            confidence += 15.0
        if abs(c - prior_low) / max(1e-6, atr) >= 0.25:
            confidence += 10.0
        notes.append("swept prior low and reclaimed")
        return ("FAILED_BREAKDOWN", flags, min(confidence, 100.0), "; ".join(notes))

    if rejected and ratio >= FAILED_MIN_RANGE_ATR:
        flags["failed_breakout"] = 1
        confidence = 60.0
        if ratio >= 1.5:
            confidence += 15.0
        if abs(prior_high - c) / max(1e-6, atr) >= 0.25:
            confidence += 10.0
        notes.append("swept prior high and rejected")
        return ("FAILED_BREAKOUT", flags, min(confidence, 100.0), "; ".join(notes))

    if broke_above and (c > prior_high) and ratio >= CLEAN_MIN_RANGE_ATR:
        notes.append("broke and held above prior high")
        return ("CLEAN_BREAKOUT", flags, 50.0 + min((ratio - 1.0) * 20.0, 30.0), "; ".join(notes))

    if broke_below and (c < prior_low) and ratio >= CLEAN_MIN_RANGE_ATR:
        notes.append("broke and held below prior low")
        return ("CLEAN_BREAKDOWN", flags, 50.0 + min((ratio - 1.0) * 20.0, 30.0), "; ".join(notes))

    if ratio <= COMPRESSION_MAX_RANGE_ATR and (not broke_above) and (not broke_below):
        notes.append("range compression")
        return ("RANGE_COMPRESSION", flags, 40.0, "; ".join(notes))

    return ("UNCLASSIFIED", flags, 20.0, "no strong regime match")


# --------------------------- live Lane B builder ---------------------------
def _today_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).date().isoformat()


def classify_regime_series(daily_bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify every session i>=1 in a daily OHLC series.

    daily_bars: ascending list of dicts with keys date, open, high, low, close.
    Returns per-session dicts mirroring liquidity_regime_events columns.
    """
    days = [
        b
        for b in (daily_bars or [])
        if all(b.get(k) is not None for k in ("date", "open", "high", "low", "close"))
    ]
    if len(days) < 2:
        return []

    trs: list[float] = []
    prev_close: Optional[float] = None
    for d in days:
        tr = compute_true_range(prev_close, float(d["high"]), float(d["low"]))
        trs.append(tr)
        prev_close = float(d["close"])
    atrs = rolling_sma(trs, ATR_LOOKBACK)

    out: list[dict[str, Any]] = []
    for i in range(1, len(days)):
        cur = days[i]
        prev = days[i - 1]
        prior_high = float(prev["high"])
        prior_low = float(prev["low"])
        o = float(cur["open"])
        h = float(cur["high"])
        l = float(cur["low"])
        c = float(cur["close"])
        tr = float(trs[i])
        atr = atrs[i]
        ratio = (tr / atr) if (atr is not None and atr > 0) else None
        regime_type, flags, conf, notes = classify_regime(prior_high, prior_low, o, h, l, c, tr, atr)
        out.append(
            {
                "session_date": cur["date"],
                "prior_key_high": prior_high,
                "prior_key_low": prior_low,
                "session_open": o,
                "session_high": h,
                "session_low": l,
                "session_close": c,
                "true_range": tr,
                "atr_14": float(atr) if atr is not None else None,
                "range_atr_ratio": float(ratio) if ratio is not None else None,
                "regime_type": regime_type,
                "regime_confidence": round(float(conf), 2),
                "notes": notes,
                **{k: int(v) for k, v in flags.items()},
            }
        )
    return out


def build_inherited_auction_context(
    daily_bars: list[dict[str, Any]],
    *,
    today: Optional[str] = None,
    symbol: str = "SPY",
) -> dict[str, Any]:
    """Compute today's INHERITED auction bucket from the most recent COMPLETED session.

    Yahoo daily bars may include a partial in-progress bar for `today`; we skip
    any bar whose date >= today so the inherited bucket is always a completed
    session. Returns an ``sharpedge.auction_context.v1`` packet with proof and
    freshness so stale/backtest data can never masquerade as live.
    """
    today = today or _today_iso()
    series = classify_regime_series(daily_bars)
    completed = [row for row in series if str(row["session_date"]) < today]

    if not completed:
        return {
            "schema": "sharpedge.auction_context.v1",
            "symbol": symbol,
            "available": False,
            "bucket": "UNCLASSIFIED",
            "confidence": 0,
            "range_atr_ratio": None,
            "inherited_from_session": None,
            "sessions_stale": None,
            "story": "No completed daily session available to inherit.",
            "source": "live:yahoo_daily_bars",
        }

    row = completed[-1]
    inherited_date = str(row["session_date"])
    # freshness: trading-day-ish staleness by calendar days as a cheap proxy
    try:
        d_inherited = dt.date.fromisoformat(inherited_date)
        d_today = dt.date.fromisoformat(today)
        stale_days = (d_today - d_inherited).days
    except ValueError:
        stale_days = None

    bucket = str(row["regime_type"])
    return {
        "schema": "sharpedge.auction_context.v1",
        "symbol": symbol,
        "available": True,
        "bucket": bucket,
        "confidence": int(round(float(row["regime_confidence"] or 0))),
        "range_atr_ratio": row["range_atr_ratio"],
        "inherited_from_session": inherited_date,
        "calendar_days_stale": stale_days,
        "story": REGIME_STORY.get(bucket, REGIME_STORY["UNCLASSIFIED"]),
        "flags": {
            "failed_breakdown": row["failed_breakdown"],
            "failed_breakout": row["failed_breakout"],
            "reclaimed_level": row["reclaimed_level"],
            "rejected_level": row["rejected_level"],
            "broke_above_high": row["broke_above_high"],
            "broke_below_low": row["broke_below_low"],
        },
        "prior_session_ohlc": {
            "open": row["session_open"],
            "high": row["session_high"],
            "low": row["session_low"],
            "close": row["session_close"],
        },
        "source": "live:yahoo_daily_bars",
        "classifier": "canonical:auction_regime.classify_regime",
    }


__all__ = [
    "classify_regime",
    "classify_regime_series",
    "build_inherited_auction_context",
    "compute_true_range",
    "rolling_sma",
    "REGIME_STORY",
]
