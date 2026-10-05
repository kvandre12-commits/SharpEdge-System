#!/usr/bin/env python3
"""
Diff Options Snapshots -> Flow Events (SharpEdge 2.0 - Options Snapshots).

Compares the two most recent options chain snapshots for the latest session
and flags unusual positioning changes ("flow"): strikes/expiries where open
interest moved the most between snapshots.

This is the "snapshot diffing" half of 2.0 options: one snapshot is a photo,
two snapshots are a story.

Reads:
    options_chain_snapshots  (written by ingest_alpaca_options_chain_snapshots.py)

Writes:
    options_flow_events(snapshot_ts, session_date, underlying,
                         expiry_date, dte, strike, side,
                         d_oi, d_volume, oi_now, volume_now, flow_rank)
    PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike, side)

Selection rule: rank strikes by |d_call_oi| + |d_put_oi| descending, keep the
top FLOW_TOP_N with combined |d_oi| >= FLOW_MIN_OI. Both sides are recorded
for each selected strike.

Idempotent: managed rows for the snapshot are deleted before re-insert.
Fail-soft: fewer than 2 snapshots for the latest session -> no events, exit 0.

Usage:
    python scripts/diff_options_snapshots.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/diff_options_snapshots.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
FLOW_MIN_OI = int(os.getenv("FLOW_MIN_OI", "1000"))
FLOW_TOP_N = int(os.getenv("FLOW_TOP_N", "15"))


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_flow_events (
          snapshot_ts  TEXT NOT NULL,
          session_date TEXT NOT NULL,
          underlying   TEXT NOT NULL,
          expiry_date  TEXT NOT NULL,
          dte          INTEGER NOT NULL,
          strike       REAL NOT NULL,
          side         TEXT NOT NULL,
          d_oi         INTEGER NOT NULL,
          d_volume     INTEGER NOT NULL,
          oi_now       INTEGER,
          volume_now   INTEGER,
          flow_rank    INTEGER NOT NULL,
          PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike, side)
        );
        """
    )
    con.commit()


def latest_two_snapshots(con: sqlite3.Connection, symbol: str):
    """Return (prev_ts, latest_ts, session_date) or None if < 2 snapshots."""
    row = con.execute(
        """
        SELECT session_date, MAX(snapshot_ts)
        FROM options_chain_snapshots
        WHERE underlying = ?
        GROUP BY session_date
        ORDER BY session_date DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if not row:
        return None
    session_date = row[0]
    ts_rows = con.execute(
        """
        SELECT DISTINCT snapshot_ts
        FROM options_chain_snapshots
        WHERE underlying = ? AND session_date = ?
        ORDER BY snapshot_ts DESC
        LIMIT 2
        """,
        (symbol, session_date),
    ).fetchall()
    if len(ts_rows) < 2:
        return None
    return ts_rows[1][0], ts_rows[0][0], session_date


def diff_snapshots_sqlite(con, symbol, prev_ts, latest_ts):
    rows = con.execute(
        """
        SELECT expiry_date, strike, dte,
               SUM(d_call_oi), SUM(d_put_oi), SUM(d_call_vol), SUM(d_put_vol),
               MAX(call_oi_now), MAX(put_oi_now),
               MAX(call_vol_now), MAX(put_vol_now)
        FROM (
          SELECT l.expiry_date, l.strike, l.dte,
                 COALESCE(l.call_oi,0)-COALESCE(p.call_oi,0) AS d_call_oi,
                 COALESCE(l.put_oi,0)-COALESCE(p.put_oi,0) AS d_put_oi,
                 COALESCE(l.call_volume,0)-COALESCE(p.call_volume,0) AS d_call_vol,
                 COALESCE(l.put_volume,0)-COALESCE(p.put_volume,0) AS d_put_vol,
                 l.call_oi AS call_oi_now, l.put_oi AS put_oi_now,
                 l.call_volume AS call_vol_now, l.put_volume AS put_vol_now
          FROM options_chain_snapshots l
          LEFT JOIN options_chain_snapshots p
            ON p.underlying=? AND p.snapshot_ts=?
           AND p.expiry_date=l.expiry_date AND p.strike=l.strike
          WHERE l.underlying=? AND l.snapshot_ts=?
          UNION ALL
          SELECT p.expiry_date, p.strike, p.dte,
                 COALESCE(l.call_oi,0)-COALESCE(p.call_oi,0),
                 COALESCE(l.put_oi,0)-COALESCE(p.put_oi,0),
                 COALESCE(l.call_volume,0)-COALESCE(p.call_volume,0),
                 COALESCE(l.put_volume,0)-COALESCE(p.put_volume,0),
                 l.call_oi, l.put_oi, l.call_volume, l.put_volume
          FROM options_chain_snapshots p
          LEFT JOIN options_chain_snapshots l
            ON l.underlying=? AND l.snapshot_ts=?
           AND l.expiry_date=p.expiry_date AND l.strike=p.strike
          WHERE p.underlying=? AND p.snapshot_ts=?
            AND l.snapshot_ts IS NULL
        )
        GROUP BY expiry_date, strike, dte
        """,
        (symbol, prev_ts, symbol, latest_ts,
         symbol, latest_ts, symbol, prev_ts),
    ).fetchall()
    return rows


def select_flow(rows, top_n: int, min_oi: int):
    scored = []
    for (expiry, strike, dte, d_call_oi, d_put_oi, d_call_vol, d_put_vol,
         call_oi_now, put_oi_now, call_vol_now, put_vol_now) in rows:
        mag = abs(d_call_oi or 0) + abs(d_put_oi or 0)
        scored.append((mag, {
            "expiry_date": expiry, "strike": strike, "dte": dte,
            "d_call_oi": d_call_oi or 0, "d_put_oi": d_put_oi or 0,
            "d_call_vol": d_call_vol or 0, "d_put_vol": d_put_vol or 0,
            "call_oi_now": call_oi_now, "put_oi_now": put_oi_now,
            "call_vol_now": call_vol_now, "put_vol_now": put_vol_now,
        }))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [s for mag, s in scored if mag >= min_oi][:top_n]


def store(con: sqlite3.Connection, symbol: str, snapshot_ts: str,
          session_date: str, selected: list) -> int:
    con.execute(
        "DELETE FROM options_flow_events WHERE snapshot_ts=? AND underlying=?",
        (snapshot_ts, symbol),
    )
    n = 0
    for rank, s in enumerate(selected, start=1):
        for side, d_oi, d_vol, oi_now, vol_now in (
            ("call", s["d_call_oi"], s["d_call_vol"], s["call_oi_now"], s["call_vol_now"]),
            ("put", s["d_put_oi"], s["d_put_vol"], s["put_oi_now"], s["put_vol_now"]),
        ):
            con.execute(
                """
                INSERT OR REPLACE INTO options_flow_events
                  (snapshot_ts, session_date, underlying, expiry_date, dte,
                   strike, side, d_oi, d_volume, oi_now, volume_now, flow_rank)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (snapshot_ts, session_date, symbol, s["expiry_date"], s["dte"],
                 s["strike"], side, d_oi, d_vol, oi_now, vol_now, rank),
            )
            n += 1
    con.commit()
    return n


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "options_chain_snapshots"):
            print("WARN: no options_chain_snapshots table; nothing to diff")
            return 0
        ensure_table(con)
        pair = latest_two_snapshots(con, SYMBOL)
        if not pair:
            print("no snapshot pair to diff (need 2+ snapshots); nothing to do")
            return 0
        prev_ts, latest_ts, session_date = pair
        rows = diff_snapshots_sqlite(con, SYMBOL, prev_ts, latest_ts)
        selected = select_flow(rows, FLOW_TOP_N, FLOW_MIN_OI)
        n = store(con, SYMBOL, latest_ts, session_date, selected)
        net_call = sum(s["d_call_oi"] for s in selected)
        net_put = sum(s["d_put_oi"] for s in selected)
        print(f"flow: {latest_ts} vs {prev_ts}: {len(selected)} strikes, {n} rows; "
              f"net d_oi calls {net_call:+d}, puts {net_put:+d}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
