#!/usr/bin/env python3
"""
Build Options Snapshot Features (SharpEdge 2.0 - Options Snapshots).

Turns raw chain snapshots into decision-ready per-session features:

  - n_snapshots        how many intraday snapshots the session has
  - pcr_oi / pcr_vol   put/call open-interest and volume ratios (0-1 DTE,
                       from the existing positioning metrics)
  - max_pain           strike minimizing total option-holder payout
  - d_max_pain_1d      max-pain drift vs previous session (velocity)
  - d_pcr_oi_1d        put/call OI ratio change vs previous session
  - net_d_oi_calls / net_d_oi_puts   intraday OI build from flow events
  - top_flow_strike / top_flow_d_oi  where the biggest positioning change hit

Reads:
    options_chain_snapshots, options_positioning_metrics, options_flow_events

Writes:
    options_snapshot_features(session_date, underlying, n_snapshots,
        pcr_oi, pcr_vol, max_pain, d_max_pain_1d, d_pcr_oi_1d,
        net_d_oi_calls, net_d_oi_puts, top_flow_strike, top_flow_d_oi)
    PRIMARY KEY (session_date, underlying)

Idempotent (INSERT OR REPLACE). Sessions with no usable data are skipped.

Usage:
    python scripts/build_options_snapshot_features.py
    SPY_DB_PATH=data/spy_truth.db SYMBOL=SPY python scripts/build_options_snapshot_features.py
"""

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
LOOKBACK_SESSIONS = int(os.getenv("SNAPSHOT_FEATURE_LOOKBACK", "60"))
DTE_MAX = int(os.getenv("SNAPSHOT_FEATURE_DTE_MAX", "3"))


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_snapshot_features (
          session_date   TEXT NOT NULL,
          underlying     TEXT NOT NULL,
          n_snapshots    INTEGER,
          pcr_oi         REAL,
          pcr_vol        REAL,
          max_pain       REAL,
          d_max_pain_1d  REAL,
          d_pcr_oi_1d    REAL,
          net_d_oi_calls INTEGER,
          net_d_oi_puts  INTEGER,
          top_flow_strike REAL,
          top_flow_d_oi  INTEGER,
          PRIMARY KEY (session_date, underlying)
        );
        """
    )
    con.commit()


def sessions(con: sqlite3.Connection, symbol: str):
    rows = con.execute(
        """
        SELECT session_date, COUNT(DISTINCT snapshot_ts) AS n,
               MAX(snapshot_ts) AS latest_ts
        FROM options_chain_snapshots
        WHERE underlying = ?
        GROUP BY session_date
        ORDER BY session_date DESC
        LIMIT ?
        """,
        (symbol, LOOKBACK_SESSIONS),
    ).fetchall()
    return rows


def latest_positioning(con: sqlite3.Connection, symbol: str, session_date: str):
    """Latest 0-1 DTE positioning row for the session (pcr, spot)."""
    if not table_exists(con, "options_positioning_metrics"):
        return None
    return con.execute(
        """
        SELECT pcr_oi, pcr_vol, spot
        FROM options_positioning_metrics
        WHERE underlying = ? AND session_date = ?
          AND dte_min = 0 AND dte_max = 1
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (symbol, session_date),
    ).fetchone()


def max_pain(con: sqlite3.Connection, symbol: str, latest_ts: str):
    """Strike minimizing total option-holder payout (classic max pain)."""
    rows = con.execute(
        """
        SELECT strike,
               SUM(COALESCE(call_oi, 0)) AS c_oi,
               SUM(COALESCE(put_oi, 0)) AS p_oi
        FROM options_chain_snapshots
        WHERE underlying = ? AND snapshot_ts = ? AND dte <= ?
        GROUP BY strike
        HAVING c_oi > 0 OR p_oi > 0
        ORDER BY strike
        """,
        (symbol, latest_ts, DTE_MAX),
    ).fetchall()
    if len(rows) < 3:
        return None
    strikes = [r[0] for r in rows]
    best, best_pain = None, None
    for s in strikes:
        pain = 0.0
        for k, c_oi, p_oi in rows:
            if s > k:
                pain += c_oi * (s - k)
            elif s < k:
                pain += p_oi * (k - s)
        if best_pain is None or pain < best_pain:
            best, best_pain = s, pain
    return best


def flow_summary(con: sqlite3.Connection, symbol: str, latest_ts: str):
    """Net OI deltas + top flow strike for a snapshot."""
    if not table_exists(con, "options_flow_events"):
        return None
    rows = con.execute(
        """
        SELECT side, strike, d_oi
        FROM options_flow_events
        WHERE underlying = ? AND snapshot_ts = ?
        ORDER BY flow_rank, side
        """,
        (symbol, latest_ts),
    ).fetchall()
    if not rows:
        return None
    net_call = sum(r[2] for r in rows if r[0] == "call")
    net_put = sum(r[2] for r in rows if r[0] == "put")
    top = max(rows, key=lambda r: abs(r[2]))
    return net_call, net_put, top[1], top[2]


def build(con: sqlite3.Connection, symbol: str) -> int:
    ensure_table(con)
    sess = sessions(con, symbol)
    prev = {}  # session_date -> (max_pain, pcr_oi) of previous session, ascending
    feats = []
    for session_date, n_snaps, latest_ts in sorted(sess):
        pos = latest_positioning(con, symbol, session_date)
        pcr_oi = pos[0] if pos else None
        pcr_vol = pos[1] if pos else None
        mp = max_pain(con, symbol, latest_ts)
        flow = flow_summary(con, symbol, latest_ts)
        net_call, net_put, top_strike, top_d_oi = flow if flow else (None, None, None, None)
        feats.append({
            "session_date": session_date, "n_snapshots": n_snaps,
            "pcr_oi": pcr_oi, "pcr_vol": pcr_vol, "max_pain": mp,
            "net_d_oi_calls": net_call, "net_d_oi_puts": net_put,
            "top_flow_strike": top_strike, "top_flow_d_oi": top_d_oi,
        })
        prev[session_date] = (mp, pcr_oi)

    ordered = sorted(feats, key=lambda f: f["session_date"])
    n = 0
    for i, f in enumerate(ordered):
        d_mp = d_pcr = None
        if i > 0:
            p = ordered[i - 1]
            if f["max_pain"] is not None and p["max_pain"] is not None:
                d_mp = round(f["max_pain"] - p["max_pain"], 2)
            if f["pcr_oi"] is not None and p["pcr_oi"] is not None:
                d_pcr = round(f["pcr_oi"] - p["pcr_oi"], 4)
        con.execute(
            """
            INSERT OR REPLACE INTO options_snapshot_features
              (session_date, underlying, n_snapshots, pcr_oi, pcr_vol,
               max_pain, d_max_pain_1d, d_pcr_oi_1d,
               net_d_oi_calls, net_d_oi_puts, top_flow_strike, top_flow_d_oi)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (f["session_date"], symbol, f["n_snapshots"], f["pcr_oi"], f["pcr_vol"],
             f["max_pain"], d_mp, d_pcr,
             f["net_d_oi_calls"], f["net_d_oi_puts"],
             f["top_flow_strike"], f["top_flow_d_oi"]),
        )
        n += 1
    con.commit()
    return n


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "options_chain_snapshots"):
            print("WARN: no options_chain_snapshots table; nothing to build")
            return 0
        n = build(con, SYMBOL)
        print(f"options_snapshot_features: {n} sessions for {SYMBOL}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
