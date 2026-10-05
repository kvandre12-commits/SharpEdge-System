#!/usr/bin/env python3
"""
Seed Candle Pattern Library (SharpEdge 2.0 - Pattern Engine).

The "teaches" half of the pattern engine: plain-English definitions for every
pattern the detector knows. The other half is measured, not memorized --
`build_pattern_expectancy.py` records what actually happened after each
pattern on the symbol's own history.

Table: candle_pattern_library(pattern, aka, description, textbook_bias,
                               candles_involved)
       PRIMARY KEY (pattern)

textbook_bias is what the textbooks claim (bullish/bearish/neutral). Treat it
as folklore until the expectancy table confirms or denies it.

Usage:
    python scripts/seed_pattern_library.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")

# (pattern, aka, description, textbook_bias, candles_involved)
PATTERNS = [
    ("doji", "Doji",
     "Open and close nearly equal -- the day ended in a draw between buyers "
     "and sellers. Indecision, not direction.",
     "neutral", 1),
    ("hammer", "Hammer",
     "Small body near the high with a long lower wick: sellers pushed the "
     "price down, buyers took it all back before the close. Appears after a "
     "decline.",
     "bullish", 1),
    ("hanging_man", "Hanging Man",
     "Same shape as a hammer but appearing after a rally: the long lower wick "
     "shows sellers are starting to test the downside.",
     "bearish", 1),
    ("inverted_hammer", "Inverted Hammer",
     "Long upper wick with a small body near the low, after a decline: buyers "
     "tried to take control intraday but couldn't hold it. Needs confirmation.",
     "bullish", 1),
    ("shooting_star", "Shooting Star",
     "Long upper wick with a small body near the low, after a rally: buyers "
     "pushed up and got rejected. A warning, not a verdict.",
     "bearish", 1),
    ("spinning_top", "Spinning Top",
     "Small body with long wicks on both sides: nobody won the day. Pure "
     "indecision -- often appears at turning points, but points nowhere.",
     "neutral", 1),
    ("marubozu_bull", "Bullish Marubozu",
     "A full-range green candle with almost no wicks: buyers owned the entire "
     "session from open to close. Strength, but watch for exhaustion.",
     "bullish", 1),
    ("marubozu_bear", "Bearish Marubozu",
     "A full-range red candle with almost no wicks: sellers owned the entire "
     "session. Strength to the downside, but watch for exhaustion.",
     "bearish", 1),
    ("engulfing_bull", "Bullish Engulfing",
     "A green body that completely swallows the prior red body: control "
     "flipped from sellers to buyers in one session.",
     "bullish", 2),
    ("engulfing_bear", "Bearish Engulfing",
     "A red body that completely swallows the prior green body: control "
     "flipped from buyers to sellers in one session.",
     "bearish", 2),
    ("harami_bull", "Bullish Harami",
     "A small green body tucked inside the prior long red body: selling "
     "pressure is weakening. Quieter than an engulfing -- also easier to "
     "ignore, which is the risk.",
     "bullish", 2),
    ("harami_bear", "Bearish Harami",
     "A small red body tucked inside the prior long green body: buying "
     "pressure is weakening.",
     "bearish", 2),
    ("morning_star", "Morning Star",
     "Three-candle bottom: a long red candle, a small indecision candle, then "
     "a long green candle closing above the first candle's midpoint. The "
     "textbook reversal -- still demands confirmation.",
     "bullish", 3),
    ("evening_star", "Evening Star",
     "Three-candle top: a long green candle, a small indecision candle, then "
     "a long red candle closing below the first candle's midpoint.",
     "bearish", 3),
    ("soldiers_3white", "Three White Soldiers",
     "Three consecutive strong green closes, each higher than the last: "
     "sustained buying. Powerful -- and prone to arriving late.",
     "bullish", 3),
    ("crows_3black", "Three Black Crows",
     "Three consecutive strong red closes, each lower than the last: "
     "sustained selling.",
     "bearish", 3),
]


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS candle_pattern_library (
          pattern           TEXT NOT NULL PRIMARY KEY,
          aka               TEXT NOT NULL,
          description       TEXT NOT NULL,
          textbook_bias     TEXT NOT NULL,
          candles_involved  INTEGER NOT NULL
        );
        """
    )
    con.commit()


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        before = con.execute("SELECT COUNT(*) FROM candle_pattern_library").fetchone()[0]
        con.executemany(
            """
            INSERT OR IGNORE INTO candle_pattern_library
              (pattern, aka, description, textbook_bias, candles_involved)
            VALUES (?,?,?,?,?)
            """,
            PATTERNS,
        )
        con.commit()
        after = con.execute("SELECT COUNT(*) FROM candle_pattern_library").fetchone()[0]
        print(f"candle_pattern_library: +{after - before} new, {after} total")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
