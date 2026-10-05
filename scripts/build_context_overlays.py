#!/usr/bin/env python3
"""
Build Context Overlays (SharpEdge 2.0 - Context Layer).

Materializes the two context sources into the existing `overlays_daily`
table so downstream consumers (regime builders, signal strength, trade card)
pick up context with zero changes to their queries:

  - event_library  -> overlay_type 'event_<type>' (strength from the library)
  - news_daily     -> overlay_type 'news_tone'   (daily mean tone in [-1, 1])

Idempotent: rows this script manages are deleted and re-inserted each run
(scoped to the symbol), so re-running never duplicates.

Usage:
    python scripts/build_context_overlays.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/build_context_overlays.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

NEWS_TONE_TYPE = "news_tone"
EVENT_PREFIX = "event_"


def ensure_overlays_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS overlays_daily (
          date             TEXT NOT NULL,
          symbol           TEXT NOT NULL,
          overlay_type     TEXT NOT NULL,
          overlay_strength REAL NOT NULL,
          notes            TEXT,
          PRIMARY KEY (symbol, date, overlay_type)
        );
        """
    )
    con.commit()


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def clear_managed(con: sqlite3.Connection, symbol: str) -> int:
    cur = con.execute(
        """
        DELETE FROM overlays_daily
        WHERE symbol = ?
          AND (overlay_type = ? OR overlay_type LIKE ? ESCAPE '\\')
        """,
        (symbol, NEWS_TONE_TYPE, EVENT_PREFIX.replace("_", "\\_") + "%"),
    )
    return cur.rowcount


def materialize_events(con: sqlite3.Connection, symbol: str) -> int:
    if not table_exists(con, "event_library"):
        return 0
    rows = con.execute(
        """
        SELECT date, event_type, title, strength
        FROM event_library
        WHERE symbol = ?
        """,
        (symbol,),
    ).fetchall()
    con.executemany(
        """
        INSERT OR REPLACE INTO overlays_daily
          (date, symbol, overlay_type, overlay_strength, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(d, symbol, f"{EVENT_PREFIX}{t}", s, title) for d, t, title, s in rows],
    )
    return len(rows)


def materialize_news_tone(con: sqlite3.Connection, symbol: str) -> int:
    if not table_exists(con, "news_daily"):
        return 0
    rows = con.execute(
        """
        SELECT date, AVG(tone) AS mean_tone, COUNT(*) AS n
        FROM news_daily
        WHERE symbol = ?
        GROUP BY date
        """,
        (symbol,),
    ).fetchall()
    con.executemany(
        """
        INSERT OR REPLACE INTO overlays_daily
          (date, symbol, overlay_type, overlay_strength, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(d, symbol, NEWS_TONE_TYPE, round(m, 3),
          f"{n} headlines, mean tone {m:+.2f}") for d, m, n in rows],
    )
    return len(rows)


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_overlays_table(con)
        cleared = clear_managed(con, SYMBOL)
        n_events = materialize_events(con, SYMBOL)
        n_tone = materialize_news_tone(con, SYMBOL)
        con.commit()
        print(f"overlays_daily: cleared {cleared} managed rows; "
              f"+{n_events} event overlays, +{n_tone} news_tone days for {SYMBOL}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
