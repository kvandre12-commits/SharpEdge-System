#!/usr/bin/env python3
"""
Aggregate options positioning metrics for SPY (or SYMBOL) into SQLite.

Key design:
- Gamma geometry + volumes come from options_chain_snapshots (intraday market-data snapshots)
- OI walls + PCR(OI) come from options_open_interest_daily (contracts/OI ingest)
  because snapshot OI can be missing/zero.

This file is a drop-in replacement for your aggregate_options_positioning_metrics script.
"""

import os
import sqlite3
from typing import Optional, List, Tuple, Dict
import numpy as np

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")
DTE_MIN = int(os.getenv("DTE_MIN", "0"))
DTE_MAX = int(os.getenv("DTE_MAX", "3"))

# Optional: restrict to a single snapshot_ts (useful for debugging)
ONLY_SNAPSHOT_TS = os.getenv("SNAPSHOT_TS", "").strip()

# Placeholder knobs (kept for compatibility; state/gamma_proxy not computed yet)
COMPUTE_STATE = os.getenv("COMPUTE_DEALER_STATE", "1").strip() == "1"
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))

# ---------------- DB helpers ----------------

def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def column_names(con: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


# ---------------- spot lookup ----------------

def pick_daily_table(con: sqlite3.Connection) -> Optional[str]:
    # Your repo uses a few different daily tables; this keeps it flexible.
    for t in ["bars_daily", "truth_daily", "ohlc_daily", "spy_daily"]:
        if table_exists(con, t):
            cols = column_names(con, t)
            if {"date", "symbol", "close"}.issubset(cols):
                return t
    return None


def get_spot_for_session(con: sqlite3.Connection, session_date: str) -> Optional[float]:
    t = pick_daily_table(con)
    if not t:
        return None

    row = con.execute(
        f"SELECT close FROM {t} WHERE symbol=? AND date=? LIMIT 1",
        (UNDERLYING, session_date),
    ).fetchone()

    return float(row[0]) if row and row[0] is not None else None


# ---------------- gamma flip ----------------

def compute_flip(strikes: List[float], net: List[float], spot: Optional[float]) -> Optional[float]:
    if spot is None or len(strikes) < 2:
        return None

    s = np.array(strikes, dtype=float)
    g = np.array(net, dtype=float)

    order = np.argsort(s)
    s, g = s[order], g[order]

    # Find any sign change (simple linear interpolation)
    for a in range(len(s) - 1):
        if np.sign(g[a]) != np.sign(g[a + 1]):
            x1, x2 = float(s[a]), float(s[a + 1])
            g1, g2 = float(g[a]), float(g[a + 1])
            # Guard divide by zero
            if g2 == g1:
                return float(x1)
            return float(x1 + (0.0 - g1) * (x2 - x1) / (g2 - g1))

    return None


# ---------------- OI table helpers ----------------

def pick_oi_expiry_col(con: sqlite3.Connection) -> Optional[str]:
    cols = column_names(con, "options_open_interest_daily")
    # Common names across scripts/exports
    for c in ["expiration_date", "expiry", "expiration", "exp_date", "expirationDate", "expiration_date_utc"]:
        if c in cols:
            return c
    return None


def pick_oi_underlying_col(con: sqlite3.Connection) -> str:
    cols = column_names(con, "options_open_interest_daily")
    return "underlying" if "underlying" in cols else ("symbol" if "symbol" in cols else "underlying")


def pick_oi_session_col(con: sqlite3.Connection) -> str:
    cols = column_names(con, "options_open_interest_daily")
    return "session_date" if "session_date" in cols else ("date" if "date" in cols else "session_date")


def fetch_oi_by_strike(
    con: sqlite3.Connection,
    session_date: str,
    dte_min: int,
    dte_max: int,
) -> Tuple[Dict[float, int], Dict[float, int]]:
    """
    Returns (call_oi_by_strike, put_oi_by_strike) filtered to a DTE window.
    Uses options_open_interest_daily when available, otherwise empty dicts.
    """
    call_oi_by: Dict[float, int] = {}
    put_oi_by: Dict[float, int] = {}

    if not table_exists(con, "options_open_interest_daily"):
        return call_oi_by, put_oi_by

    exp_col = pick_oi_expiry_col(con)
    if not exp_col:
        return call_oi_by, put_oi_by

    und_col = pick_oi_underlying_col(con)
    sess_col = pick_oi_session_col(con)

    # DTE computed as integer days between expiry and session_date
    q = f"""
        SELECT
          CAST(strike AS REAL) AS strike,
          COALESCE(call_oi, 0) AS call_oi,
          COALESCE(put_oi, 0)  AS put_oi
        FROM options_open_interest_daily
        WHERE {sess_col}=? AND {und_col}=?
          AND CAST((julianday({exp_col}) - julianday(?)) AS INT) BETWEEN ? AND ?
    """
    oi_rows = con.execute(q, (session_date, UNDERLYING, session_date, dte_min, dte_max)).fetchall()

    for k, co, po in oi_rows:
        if k is None:
            continue
        strike = float(k)
        call_oi_by[strike] = call_oi_by.get(strike, 0) + int(co or 0)
        put_oi_by[strike] = put_oi_by.get(strike, 0) + int(po or 0)

    return call_oi_by, put_oi_by


def argmax_positive(d: Dict[float, int]) -> Optional[float]:
    if not d:
        return None
    k = max(d.keys(), key=lambda x: d[x])
    return float(k) if d[k] > 0 else None


# ---------------- compute metrics ----------------

def compute_metrics(con: sqlite3.Connection, snapshot_ts: str):
    """
    One row per (snapshot_ts, underlying, dte_min, dte_max).
    Pulls gamma/volume from snapshots and OI walls from contracts OI table.
    """

    rows = con.execute(
        """
        SELECT session_date, strike, call_oi, put_oi,
               call_volume, put_volume, call_gamma, put_gamma
        FROM options_chain_snapshots
        WHERE underlying=? AND snapshot_ts=? AND dte BETWEEN ? AND ?
        """,
        (UNDERLYING, snapshot_ts, DTE_MIN, DTE_MAX),
    ).fetchall()

    if not rows:
        return None

    session_date = rows[0][0]
    spot = get_spot_for_session(con, session_date)

    # Volumes from snapshots (usually reliable)
    total_call_vol = sum(r[4] or 0 for r in rows)
    total_put_vol = sum(r[5] or 0 for r in rows)
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol else None

    # ---- OI walls from contracts OI table ----
    call_oi_by, put_oi_by = fetch_oi_by_strike(con, session_date, DTE_MIN, DTE_MAX)

    total_call_oi = sum(call_oi_by.values())
    total_put_oi = sum(put_oi_by.values())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else None

    strikes_oi = sorted(set(call_oi_by.keys()) | set(put_oi_by.keys()))

    max_call_oi_strike = argmax_positive(call_oi_by)
    max_put_oi_strike = argmax_positive(put_oi_by)
    total_oi_by = {k: call_oi_by.get(k, 0) + put_oi_by.get(k, 0) for k in strikes_oi}
    max_total_oi_strike = argmax_positive(total_oi_by)

    # ATM strike (closest listed strike to spot): prefer OI strikes (stable ladder);
    # if empty, fall back to snapshot strikes.
    atm_strike = None
    if spot is not None:
        if strikes_oi:
            atm_strike = float(min(strikes_oi, key=lambda s: abs(s - spot)))
        else:
            snap_strikes = sorted({float(r[1]) for r in rows if r[1] is not None})
            if snap_strikes:
                atm_strike = float(min(snap_strikes, key=lambda s: abs(s - spot)))

    # ---- gamma geometry from snapshots ----
    by_strike: Dict[float, float] = {}
    for _, k, co, po, _, _, cg, pg in rows:
        if k is None:
            continue
        strike = float(k)
        net = (cg or 0) * (co or 0) - (pg or 0) * (po or 0)
        by_strike[strike] = by_strike.get(strike, 0.0) + float(net)

    strikes = sorted(by_strike.keys())
    net = [by_strike[k] for k in strikes]

    gamma_wall = max(strikes, key=lambda k: abs(by_strike[k])) if strikes else None
    gamma_pos = max(strikes, key=lambda k: by_strike[k]) if strikes else None
    gamma_neg = min(strikes, key=lambda k: by_strike[k]) if strikes else None
    gamma_flip = compute_flip(strikes, net, spot) if strikes else None

    # Placeholders (kept for compatibility with downstream fusion)
    gamma_proxy = None
    dealer_hint = None

    return (
        snapshot_ts,
        session_date,
        UNDERLYING,
        DTE_MIN,
        DTE_MAX,
        spot,
        atm_strike,
        max_total_oi_strike, max_call_oi_strike, max_put_oi_strike,
        gamma_wall,
        gamma_pos,
        gamma_neg,
        gamma_flip,
        total_call_oi,
        total_put_oi,
        pcr_oi,
        total_call_vol,
        total_put_vol,
        pcr_vol,
        gamma_proxy,
        dealer_hint,
    )


def ensure_schema(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS options_positioning_metrics (
      snapshot_ts TEXT NOT NULL,
      session_date TEXT NOT NULL,
      underlying TEXT NOT NULL,
      dte_min INTEGER NOT NULL,
      dte_max INTEGER NOT NULL,
      PRIMARY KEY (snapshot_ts, underlying, dte_min, dte_max)
    );
    """)

    want = {
        "spot": "REAL",
        "atm_strike": "REAL",

        "max_total_oi_strike": "REAL",
        "max_call_oi_strike": "REAL",
        "max_put_oi_strike": "REAL",

        "gamma_wall_strike": "REAL",
        "gamma_pos_wall_strike": "REAL",
        "gamma_neg_wall_strike": "REAL",
        "gamma_flip_strike": "REAL",

        "total_call_oi": "REAL",
        "total_put_oi": "REAL",
        "pcr_oi": "REAL",

        "total_call_vol": "REAL",
        "total_put_vol": "REAL",
        "pcr_vol": "REAL",

        "gamma_proxy": "REAL",
        "dealer_state_hint": "TEXT",
    }

    have = {r[1] for r in con.execute("PRAGMA table_info(options_positioning_metrics)")}
    for col, typ in want.items():
        if col not in have:
            con.execute(f"ALTER TABLE options_positioning_metrics ADD COLUMN {col} {typ};")


def upsert(con: sqlite3.Connection, row):
    con.execute(
        """
        INSERT INTO options_positioning_metrics (
          snapshot_ts, session_date, underlying,
          dte_min, dte_max,
          spot, atm_strike,
          max_total_oi_strike, max_call_oi_strike, max_put_oi_strike,
          gamma_wall_strike, gamma_pos_wall_strike, gamma_neg_wall_strike, gamma_flip_strike,
          total_call_oi, total_put_oi, pcr_oi,
          total_call_vol, total_put_vol, pcr_vol,
          gamma_proxy, dealer_state_hint
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(snapshot_ts, underlying, dte_min, dte_max) DO UPDATE SET
          session_date           = excluded.session_date,
          spot                   = excluded.spot,
          atm_strike             = excluded.atm_strike,
          max_total_oi_strike    = excluded.max_total_oi_strike,
          max_call_oi_strike     = excluded.max_call_oi_strike,
          max_put_oi_strike      = excluded.max_put_oi_strike,
          gamma_wall_strike      = excluded.gamma_wall_strike,
          gamma_pos_wall_strike  = excluded.gamma_pos_wall_strike,
          gamma_neg_wall_strike  = excluded.gamma_neg_wall_strike,
          gamma_flip_strike      = excluded.gamma_flip_strike,
          total_call_oi          = excluded.total_call_oi,
          total_put_oi           = excluded.total_put_oi,
          pcr_oi                 = excluded.pcr_oi,
          total_call_vol         = excluded.total_call_vol,
          total_put_vol          = excluded.total_put_vol,
          pcr_vol                = excluded.pcr_vol,
          gamma_proxy            = excluded.gamma_proxy,
          dealer_state_hint      = excluded.dealer_state_hint
        """,
        row,
    )


# ---------------- main ----------------

def main():
    con = connect()
    ensure_schema(con)

    snaps_q = "SELECT DISTINCT snapshot_ts FROM options_chain_snapshots"
    params = ()
    if ONLY_SNAPSHOT_TS:
        snaps_q += " WHERE snapshot_ts=?"
        params = (ONLY_SNAPSHOT_TS,)
    snaps_q += " ORDER BY snapshot_ts"

    snaps = [r[0] for r in con.execute(snaps_q, params).fetchall()]

    wrote = 0
    for s in snaps:
        row = compute_metrics(con, s)
        if row:
            upsert(con, row)
            wrote += 1

    con.commit()

    # Debug sample (safe; keeps CI introspection easy)
    sample = con.execute("""
        SELECT session_date, spot, max_total_oi_strike, max_call_oi_strike, max_put_oi_strike, pcr_oi
        FROM options_positioning_metrics
        WHERE underlying=? AND dte_min=? AND dte_max=?
        ORDER BY snapshot_ts DESC
        LIMIT 5
    """, (UNDERLYING, DTE_MIN, DTE_MAX)).fetchall()

    print("DEBUG positioning_metrics sample (session_date, spot, max_total, max_call, max_put, pcr_oi):")
    for r in sample:
        print(r)

    con.close()
    print(f"OK: gamma geometry integrated. upserted {wrote} snapshot rows.")


if __name__ == "__main__":
    main()
