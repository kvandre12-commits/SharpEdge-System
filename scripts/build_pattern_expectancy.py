#!/usr/bin/env python3
"""
Build Pattern Expectancy (SharpEdge 2.0 - Pattern Engine).

The "measured, not memorized" half: for every detected candlestick pattern,
record what the symbol actually did afterwards -- mean forward returns at
1/3/5 sessions and the 5-session hit rate -- over the full bar history.

This is where the library's textbook folklore meets the data. A hammer is
"bullish" because a book says so; it is worth trading only if this table
says so.

Reads:
    candle_patterns, bars_daily

Writes:
    pattern_expectancy(pattern, symbol, n,
        fwd_1d_mean, fwd_3d_mean, fwd_5d_mean, hit_rate_5d)
    PRIMARY KEY (pattern, symbol)

Idempotent (INSERT OR REPLACE). Patterns with fewer than MIN_N occurrences
are still recorded -- consumers decide their own sample-size bar.

Usage:
    python scripts/build_pattern_expectancy.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/build_pattern_expectancy.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
MIN_N = int(os.getenv("PATTERN_MIN_N", "10"))
HORIZONS = (1, 3, 5)


def table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "candle_patterns") or not table_exists(con, "bars_daily"):
            print("WARN: need candle_patterns + bars_daily; nothing to build")
            return 0
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_expectancy (
              pattern     TEXT NOT NULL,
              symbol      TEXT NOT NULL,
              n           INTEGER NOT NULL,
              fwd_1d_mean REAL,
              fwd_3d_mean REAL,
              fwd_5d_mean REAL,
              hit_rate_5d REAL,
              PRIMARY KEY (pattern, symbol)
            );
            """
        )
        closes = {r[0]: r[1] for r in con.execute(
            "SELECT date, close FROM bars_daily WHERE symbol=? ORDER BY date",
            (SYMBOL,))}
        dates = sorted(closes)
        idx = {d: i for i, d in enumerate(dates)}

        patterns = [r[0] for r in con.execute(
            "SELECT DISTINCT pattern FROM candle_patterns WHERE symbol=?",
            (SYMBOL,))]
        n_wrote = 0
        for pat in patterns:
            occ = [r[0] for r in con.execute(
                "SELECT date FROM candle_patterns WHERE symbol=? AND pattern=?",
                (SYMBOL, pat))]
            fwds = {h: [] for h in HORIZONS}
            for d in occ:
                i = idx.get(d)
                if i is None:
                    continue
                base = closes[d]
                if not base:
                    continue
                for h in HORIZONS:
                    if i + h < len(dates):
                        fwd = closes[dates[i + h]]
                        if fwd:
                            fwds[h].append(fwd / base - 1.0)
            row = {"pattern": pat, "n": len(occ)}
            for h in HORIZONS:
                vals = fwds[h]
                row[f"fwd_{h}d_mean"] = (sum(vals) / len(vals)) if vals else None
            vals5 = fwds[5]
            row["hit_rate_5d"] = (
                sum(1 for v in vals5 if v > 0) / len(vals5)) if vals5 else None
            con.execute(
                """
                INSERT OR REPLACE INTO pattern_expectancy
                  (pattern, symbol, n, fwd_1d_mean, fwd_3d_mean, fwd_5d_mean, hit_rate_5d)
                VALUES (?,?,?,?,?,?,?)
                """,
                (pat, SYMBOL, row["n"], row["fwd_1d_mean"], row["fwd_3d_mean"],
                 row["fwd_5d_mean"], row["hit_rate_5d"]),
            )
            n_wrote += 1
        con.commit()
        big = con.execute(
            "SELECT pattern, n, ROUND(fwd_5d_mean,4) FROM pattern_expectancy "
            "WHERE symbol=? AND n>=? ORDER BY fwd_5d_mean DESC LIMIT 3",
            (SYMBOL, MIN_N)).fetchall()
        print(f"pattern_expectancy: {n_wrote} patterns for {SYMBOL}; "
              f"top-5d (n>={MIN_N}): {big}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
