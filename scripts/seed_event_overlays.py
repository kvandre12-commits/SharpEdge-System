#!/usr/bin/env python3
"""Seed scheduled macro-catalyst events into overlays_daily.

Writes FOMC / Jobs (NFP) / Treasury refunding dates as overlay rows so the
research/backtest layer (overlays_daily.overlay_type) has the event context that
backtest_same_day_breakouts.OVERLAY_WEIGHTS already reserves slots for
(fomc/tariff/... — this adds fomc/jobs/treasury).

SINGLE SOURCE: event dates come from cockpit/event_calendar.py, the same list
the live cockpit EVENT RADAR reads, so the seeded rows and the cockpit flag
cannot disagree.

Seeds the range [SEED_START, SEED_END] (default: 2 years back to 1 year ahead).
Idempotent: INSERT OR IGNORE, so re-running is safe.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys

# Canonical event calendar (shared with the live cockpit radar).
_COCKPIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cockpit")
if _COCKPIT_DIR not in sys.path:
    sys.path.insert(0, _COCKPIT_DIR)
from event_calendar import all_events_in_range, event_label  # noqa: E402

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
SEED_START = os.getenv("EVENT_SEED_START", "")
SEED_END = os.getenv("EVENT_SEED_END", "")


def _default_range() -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    return today - dt.timedelta(days=730), today + dt.timedelta(days=365)


def main() -> None:
    start = dt.date.fromisoformat(SEED_START) if SEED_START else _default_range()[0]
    end = dt.date.fromisoformat(SEED_END) if SEED_END else _default_range()[1]

    events = all_events_in_range(start, end)

    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS overlays_daily (
              date TEXT NOT NULL,
              symbol TEXT NOT NULL,
              overlay_type TEXT NOT NULL,
              overlay_strength REAL NOT NULL,
              notes TEXT,
              PRIMARY KEY (symbol, date, overlay_type)
            )
            """
        )
        rows = [
            (
                e["date"],
                SYMBOL,
                e["type"],
                1.0,
                event_label(e),
            )
            for e in events
        ]
        con.executemany(
            """
            INSERT OR IGNORE INTO overlays_daily
              (date, symbol, overlay_type, overlay_strength, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()

    by_type: dict[str, int] = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
    print(
        f"OK: seeded {len(events)} event overlays into overlays_daily "
        f"[{start} .. {end}] ({breakdown})"
    )


if __name__ == "__main__":
    main()
