#!/usr/bin/env python3
"""Shared persistence for ``options_chain_snapshots``.

Both the (legacy, OI-less) Alpaca ingester and the CBOE ingester write the same
table, so the connection / schema / upsert / date helpers live here once instead
of being copy-pasted. One row = one (expiry, strike) with call+put fields merged.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - Termux/minimal envs may lack tzdata
    # Fixed US-Eastern fallback. Good enough for date derivation: market hours
    # (14:30-21:00 UTC) map to the same NY calendar date under either EST/EDT.
    NY = timezone(timedelta(hours=-5), name="EST-fallback")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ny_session_date(dt_utc: datetime | None = None) -> str:
    dt_utc = dt_utc or datetime.now(timezone.utc)
    return dt_utc.astimezone(NY).date().isoformat()


def dte(session_date: str, expiry_date: str) -> int:
    sd = datetime.fromisoformat(session_date).date()
    ed = datetime.fromisoformat(expiry_date).date()
    return (ed - sd).days


def connect(db_path: str | None = None) -> sqlite3.Connection:
    db_path = db_path or os.getenv("SPY_DB_PATH", "data/spy_truth.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS options_chain_snapshots (
      snapshot_ts   TEXT NOT NULL,
      session_date  TEXT NOT NULL,
      underlying    TEXT NOT NULL,
      expiry_date   TEXT NOT NULL,
      dte           INTEGER NOT NULL,
      strike        REAL NOT NULL,
      call_oi       INTEGER,
      put_oi        INTEGER,
      call_volume   INTEGER,
      put_volume    INTEGER,
      call_gamma    REAL,
      put_gamma     REAL,
      call_iv       REAL,
      put_iv        REAL,
      source        TEXT DEFAULT 'cboe',
      PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike)
    );
    """)
    # Safe migrations for pre-existing tables (no-op if column present).
    existing = [r[1] for r in con.execute("PRAGMA table_info(options_chain_snapshots)")]
    for col in ("call_iv", "put_iv"):
        if col not in existing:
            con.execute(f"ALTER TABLE options_chain_snapshots ADD COLUMN {col} REAL")
    con.commit()


def upsert_row(con: sqlite3.Connection, row: tuple, source: str = "cboe") -> None:
    """Insert/update one merged (expiry, strike) row.

    ``row`` = (snapshot_ts, session_date, underlying, expiry_date, dte, strike,
               call_oi, put_oi, call_volume, put_volume, call_gamma, put_gamma,
               call_iv, put_iv)
    COALESCE keeps an existing real value when a later writer supplies NULL,
    so a single-sided (call-only or put-only) update never wipes the other leg.
    """
    con.execute(
        """
      INSERT INTO options_chain_snapshots (
        snapshot_ts, session_date, underlying, expiry_date, dte, strike,
        call_oi, put_oi, call_volume, put_volume, call_gamma, put_gamma,
        call_iv, put_iv, source
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(snapshot_ts, underlying, expiry_date, strike) DO UPDATE SET
        session_date = excluded.session_date,
        dte          = excluded.dte,
        call_oi      = COALESCE(excluded.call_oi, options_chain_snapshots.call_oi),
        put_oi       = COALESCE(excluded.put_oi, options_chain_snapshots.put_oi),
        call_volume  = COALESCE(excluded.call_volume, options_chain_snapshots.call_volume),
        put_volume   = COALESCE(excluded.put_volume, options_chain_snapshots.put_volume),
        call_gamma   = COALESCE(excluded.call_gamma, options_chain_snapshots.call_gamma),
        put_gamma    = COALESCE(excluded.put_gamma, options_chain_snapshots.put_gamma),
        call_iv      = COALESCE(excluded.call_iv, options_chain_snapshots.call_iv),
        put_iv       = COALESCE(excluded.put_iv, options_chain_snapshots.put_iv),
        source       = excluded.source
    """,
        (*row, source),
    )
