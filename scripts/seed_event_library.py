#!/usr/bin/env python3
"""
Seed Curated Historical Event Library (SharpEdge 2.0 - Context Layer).

Creates the `event_library` table (if missing) and inserts a curated set of
high-signal historical market events. Idempotent: re-running changes nothing
(INSERT OR IGNORE on the natural key).

The library is the "historical context" half of the 2.0 context layer. The
"current events" half is `ingest_current_events.py`. Both are materialized
into the existing `overlays_daily` table by `build_context_overlays.py`, so
downstream consumers (regime, signal strength, trade card) pick them up with
no changes to their queries.

Schema:
    event_library(date, symbol, event_type, title, strength, notes)
    PRIMARY KEY (date, symbol, event_type)

event_type values: market_stress | policy_rates | geopolitical | election
                   | banking | trade | medical

Usage:
    python scripts/seed_event_library.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/seed_event_library.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# (date, event_type, title, strength 0..1, notes)
EVENTS = [
    # --- 2020: COVID crash & response ---
    ("2020-02-24", "market_stress", "COVID-19 selloff begins", 0.7,
     "S&P 500 -3.35%; pandemic repricing starts"),
    ("2020-03-09", "market_stress", "Black Monday I (-7.6%, circuit breaker)", 1.0,
     "Oil price war + COVID panic; trading halted"),
    ("2020-03-12", "market_stress", "Black Thursday (-10%)", 1.0,
     "Worst day since 1987 at the time"),
    ("2020-03-16", "market_stress", "Black Monday II (-12%)", 1.0,
     "Worst day since 1987; Fed emergency cut to zero"),
    ("2020-03-23", "policy_rates", "Fed announces unlimited QE", 0.9,
     "'Whatever it takes' moment; market bottom within days"),
    ("2020-11-09", "medical", "Pfizer vaccine efficacy announced", 0.7,
     "Rotation: cyclicals rip, stay-at-home unwinds"),
    # --- 2021 ---
    ("2021-01-27", "market_stress", "Meme-stock squeeze volatility peak", 0.6,
     "GME/AMC gamma squeeze; broad-market de-risking"),
    # --- 2022: inflation shock & hiking cycle ---
    ("2022-01-05", "policy_rates", "Hawkish FOMC minutes spark growth selloff", 0.7,
     "QT signal; Nasdaq enters correction"),
    ("2022-02-24", "geopolitical", "Russia invades Ukraine", 0.8,
     "Energy shock; European risk repriced"),
    ("2022-03-16", "policy_rates", "Fed first hike (+25bp), ends zero-rate era", 0.7,
     "Dot plot guides 6 more hikes"),
    ("2022-06-13", "market_stress", "S&P 500 confirms bear market", 0.9,
     "Closes -20%+ from January high on hot CPI"),
    ("2022-06-15", "policy_rates", "Fed hikes 75bp (largest since 1994)", 0.8,
     "Front-loaded tightening; volatility spike"),
    ("2022-10-13", "market_stress", "Hot CPI then historic intraday reversal", 0.7,
     "+2.6% close; bear-market bottoming process begins"),
    # --- 2023: banking stress ---
    ("2023-03-10", "banking", "SVB collapse", 0.8,
     "Second-largest US bank failure; regional-bank contagion fear"),
    ("2023-03-12", "banking", "SVB/Signature backstop announced", 0.7,
     "Systemic risk exception; futures rally"),
    # --- 2024 ---
    ("2024-04-03", "trade", "Tariff enforcement announcement", 0.7,
     "Trade-policy risk repriced"),
    ("2024-04-11", "trade", "Presidential tariff threat", 0.6,
     "Escalation rhetoric"),
    ("2024-04-24", "trade", "Trade policy remarks", 0.5,
     "Ongoing tariff headlines"),
    ("2024-08-05", "market_stress", "Yen carry-trade unwind (VIX 65)", 0.9,
     "S&P -3%; global deleveraging scare"),
    ("2024-09-18", "policy_rates", "Fed cuts 50bp (first cut of cycle)", 0.7,
     "'Recalibration'; soft-landing bet"),
    ("2024-11-06", "election", "US presidential election: Trump wins", 0.8,
     "S&P +2.5%; deregulation/tax-cut repricing"),
    # --- 2025 ---
    ("2025-04-02", "trade", "'Liberation Day' tariff announcement", 0.9,
     "Blanket tariff regime unveiled after close"),
    ("2025-04-03", "market_stress", "Tariff selloff (-4.8%)", 0.9,
     "Growth scare; recession odds repriced"),
    ("2025-04-09", "trade", "Tariff pause announced (+9.5% rally)", 0.9,
     "One of the largest single-day S&P rallies on record"),
]


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS event_library (
          date       TEXT NOT NULL,
          symbol     TEXT NOT NULL,
          event_type TEXT NOT NULL,
          title      TEXT NOT NULL,
          strength   REAL NOT NULL,
          notes      TEXT,
          PRIMARY KEY (date, symbol, event_type)
        );
        """
    )
    con.commit()


def seed(con: sqlite3.Connection, symbol: str) -> int:
    """Insert curated events. Returns number of rows newly inserted."""
    before = con.execute("SELECT COUNT(*) FROM event_library").fetchone()[0]
    con.executemany(
        """
        INSERT OR IGNORE INTO event_library
          (date, symbol, event_type, title, strength, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(d, symbol, t, title, s, n) for d, t, title, s, n in EVENTS],
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM event_library").fetchone()[0]
    return after - before


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        added = seed(con, SYMBOL)
        total = con.execute("SELECT COUNT(*) FROM event_library").fetchone()[0]
        print(f"event_library: +{added} new rows, {total} total for {SYMBOL}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
