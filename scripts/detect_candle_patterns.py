#!/usr/bin/env python3
"""
Detect Candle Patterns (SharpEdge 2.0 - Pattern Engine).

Reads daily bars from `bars_daily` and detects 16 classic candlestick
patterns using deterministic shape rules (no ML, no ambiguity). Trend-gated
patterns (hammer vs hanging man, etc.) use a 10-day SMA of closes.

Writes:
    candle_patterns(date, symbol, pattern, direction_hint, strength)
    PRIMARY KEY (date, symbol, pattern)

direction_hint is the textbook bias (bullish/bearish/neutral) -- folklore
until `build_pattern_expectancy.py` measures it. strength is a 0..1 shape
quality score.

Idempotent: the symbol's rows are rebuilt from scratch each run.

Usage:
    python scripts/detect_candle_patterns.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/detect_candle_patterns.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
TREND_N = int(os.getenv("PATTERN_TREND_N", "10"))

BULLISH = {"hammer", "inverted_hammer", "engulfing_bull", "harami_bull",
           "morning_star", "marubozu_bull", "soldiers_3white"}
BEARISH = {"hanging_man", "shooting_star", "engulfing_bear", "harami_bear",
           "evening_star", "marubozu_bear", "crows_3black"}


def direction_hint(pattern: str) -> str:
    if pattern in BULLISH:
        return "bullish"
    if pattern in BEARISH:
        return "bearish"
    return "neutral"


def metrics(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    return {
        "rng": rng,
        "body": body,
        "upper": h - max(o, c),
        "lower": min(o, c) - l,
        "bull": c > o,
        "bear": c < o,
    }


def sma(values, n):
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def single_patterns(m, trend):
    """Single-candle patterns. Returns list of (pattern, strength)."""
    out = []
    rng, body = m["rng"], m["body"]
    if body <= 0.10 * rng:
        out.append(("doji", round(1.0 - body / (0.10 * rng + 1e-9), 2)))
    long_lower = m["lower"] >= 2 * max(body, 1e-9) and m["upper"] <= 0.30 * rng
    long_upper = m["upper"] >= 2 * max(body, 1e-9) and m["lower"] <= 0.30 * rng
    if long_lower and body <= 0.35 * rng and trend == "down":
        out.append(("hammer", round(min(1.0, m["lower"] / (2.5 * max(body, 1e-9))), 2)))
    if long_lower and body <= 0.35 * rng and trend == "up":
        out.append(("hanging_man", round(min(1.0, m["lower"] / (2.5 * max(body, 1e-9))), 2)))
    if long_upper and body <= 0.35 * rng and trend == "down":
        out.append(("inverted_hammer", round(min(1.0, m["upper"] / (2.5 * max(body, 1e-9))), 2)))
    if long_upper and body <= 0.35 * rng and trend == "up":
        out.append(("shooting_star", round(min(1.0, m["upper"] / (2.5 * max(body, 1e-9))), 2)))
    if (body <= 0.30 * rng and m["upper"] >= body and m["lower"] >= body
            and body > 0.10 * rng):
        out.append(("spinning_top", round(1.0 - body / (0.30 * rng + 1e-9), 2)))
    if body >= 0.95 * rng and m["bull"]:
        out.append(("marubozu_bull", round(body / rng, 2)))
    if body >= 0.95 * rng and m["bear"]:
        out.append(("marubozu_bear", round(body / rng, 2)))
    return out


def two_patterns(p, c):
    """Two-candle patterns from (prev, curr) OHLC dicts."""
    out = []
    pm, cm = metrics(*p), metrics(*c)
    if not pm or not cm:
        return out
    po, pc = p[0], p[3]
    co, cc = c[0], c[3]
    # engulfing
    if pm["bear"] and cm["bull"] and co <= pc and cc >= po:
        out.append(("engulfing_bull", round(min(1.0, cm["body"] / (pm["body"] + 1e-9) / 2), 2)))
    if pm["bull"] and cm["bear"] and co >= pc and cc <= po:
        out.append(("engulfing_bear", round(min(1.0, cm["body"] / (pm["body"] + 1e-9) / 2), 2)))
    # harami
    if (pm["bear"] and pm["body"] >= 0.6 * pm["rng"] and cm["bull"]
            and co >= pc and cc <= po and cm["body"] <= 0.5 * pm["body"]):
        out.append(("harami_bull", round(1.0 - cm["body"] / (pm["body"] + 1e-9), 2)))
    if (pm["bull"] and pm["body"] >= 0.6 * pm["rng"] and cm["bear"]
            and co <= pc and cc >= po and cm["body"] <= 0.5 * pm["body"]):
        out.append(("harami_bear", round(1.0 - cm["body"] / (pm["body"] + 1e-9), 2)))
    return out


def three_patterns(a, b, c):
    """Three-candle patterns from three OHLC dicts (oldest first)."""
    out = []
    am, bm, cm = metrics(*a), metrics(*b), metrics(*c)
    if not am or not bm or not cm:
        return out
    # morning star
    if (am["bear"] and am["body"] >= 0.6 * am["rng"]
            and bm["body"] <= 0.3 * bm["rng"]
            and cm["bull"] and cm["body"] >= 0.6 * cm["rng"]
            and c[3] > (a[0] + a[3]) / 2):
        out.append(("morning_star", 0.9))
    # evening star
    if (am["bull"] and am["body"] >= 0.6 * am["rng"]
            and bm["body"] <= 0.3 * bm["rng"]
            and cm["bear"] and cm["body"] >= 0.6 * cm["rng"]
            and c[3] < (a[0] + a[3]) / 2):
        out.append(("evening_star", 0.9))
    # three white soldiers
    bars = [(a, am), (b, bm), (c, cm)]
    if all(m["bull"] and m["body"] >= 0.5 * m["rng"] for _, m in bars) \
            and b[3] > a[3] > 0 and c[3] > b[3]:
        out.append(("soldiers_3white", 0.9))
    if all(m["bear"] and m["body"] >= 0.5 * m["rng"] for _, m in bars) \
            and b[3] < a[3] and c[3] < b[3]:
        out.append(("crows_3black", 0.9))
    return out


def detect(bars):
    """bars: list of (date, o, h, l, c) ascending. Returns list of
    (date, pattern, direction_hint, strength)."""
    out = []
    closes = []
    for i, (date, o, h, l, c) in enumerate(bars):
        if None in (o, h, l, c):
            continue
        closes.append(c)
        m = metrics(o, h, l, c)
        if not m:
            continue
        trend = "unknown"
        s = sma(closes[:-1], TREND_N)
        if s is not None:
            trend = "up" if c > s else "down"
        for pat, strength in single_patterns(m, trend):
            out.append((date, pat, direction_hint(pat), strength))
        if i >= 1:
            prev = bars[i - 1]
            if None not in prev[1:]:
                for pat, strength in two_patterns(
                        (prev[1], prev[2], prev[3], prev[4]), (o, h, l, c)):
                    out.append((date, pat, direction_hint(pat), strength))
        if i >= 2:
            a, b = bars[i - 2], bars[i - 1]
            if None not in a[1:] and None not in b[1:]:
                for pat, strength in three_patterns(
                        (a[1], a[2], a[3], a[4]),
                        (b[1], b[2], b[3], b[4]), (o, h, l, c)):
                    out.append((date, pat, direction_hint(pat), strength))
    return out


def table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "bars_daily"):
            print("WARN: no bars_daily table; nothing to detect")
            return 0
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS candle_patterns (
              date           TEXT NOT NULL,
              symbol         TEXT NOT NULL,
              pattern        TEXT NOT NULL,
              direction_hint TEXT NOT NULL,
              strength       REAL NOT NULL,
              PRIMARY KEY (date, symbol, pattern)
            );
            """
        )
        bars = con.execute(
            """
            SELECT date, open, high, low, close
            FROM bars_daily
            WHERE symbol = ?
            ORDER BY date
            """,
            (SYMBOL,),
        ).fetchall()
        found = detect(bars)
        con.execute("DELETE FROM candle_patterns WHERE symbol = ?", (SYMBOL,))
        con.executemany(
            """
            INSERT INTO candle_patterns
              (date, symbol, pattern, direction_hint, strength)
            VALUES (?,?,?,?,?)
            """,
            [(d, SYMBOL, p, dh, s) for d, p, dh, s in found],
        )
        con.commit()
        by_pat = {}
        for _, p, _, _ in found:
            by_pat[p] = by_pat.get(p, 0) + 1
        top = sorted(by_pat.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"candle_patterns: {len(found)} detections over {len(bars)} bars "
              f"for {SYMBOL}; top={top}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
