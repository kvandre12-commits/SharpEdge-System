#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
from typing import Optional, List

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")
DTE_MIN = int(os.getenv("DTE_MIN", "0"))
DTE_MAX = int(os.getenv("DTE_MAX", "3"))

ONLY_SNAPSHOT_TS = os.getenv("SNAPSHOT_TS", "").strip()

COMPUTE_STATE = os.getenv("COMPUTE_DEALER_STATE", "1").strip() == "1"
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))


# ---------------- DB helpers ----------------

def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def table_exists(con, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def column_names(con, table: str) -> set:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


# ---------------- spot lookup ----------------

def pick_daily_table(con) -> Optional[str]:
    for t in ["bars_daily", "truth_daily", "ohlc_daily", "spy_daily"]:
        if table_exists(con, t):
            cols = column_names(con, t)
            if {"date", "symbol", "close"}.issubset(cols):
                return t
    return None


def get_spot_for_session(con, session_date: str) -> Optional[float]:
    t = pick_daily_table(con)
    if not t:
        return None

    row = con.execute(
        f"SELECT close FROM {t} WHERE symbol=? AND date=? LIMIT 1",
        (UNDERLYING, session_date),
    ).fetchone()

    return float(row[0]) if row and row[0] is not None else None


# ---------------- gamma flip ----------------

def compute_flip(strikes: List[float], net: List[float], spot: float):
    if spot is None or len(strikes) < 2:
        return None

    s = np.array(strikes)
    g = np.array(net)

    order = np.argsort(s)
    s, g = s[order], g[order]

    idx = np.searchsorted(s, spot)

    for a in range(len(s) - 1):
        if np.sign(g[a]) != np.sign(g[a + 1]):
            x1, x2 = s[a], s[a + 1]
            g1, g2 = g[a], g[a + 1]
            return float(x1 + (0 - g1) * (x2 - x1) / (g2 - g1))

    return None


# ---------------- compute metrics ----------------

def compute_metrics(con, snapshot_ts: str):

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

    total_call_oi = sum(r[2] or 0 for r in rows)
    total_put_oi = sum(r[3] or 0 for r in rows)

    total_call_vol = sum(r[4] or 0 for r in rows)
    total_put_vol = sum(r[5] or 0 for r in rows)

    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else None
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol else None

    # -------- gamma geometry --------
    by_strike = {}
    for _, k, co, po, _, _, cg, pg in rows:
        net = (cg or 0) * (co or 0) - (pg or 0) * (po or 0)
        by_strike[k] = by_strike.get(k, 0) + net

    strikes = sorted(by_strike.keys())
    net = [by_strike[k] for k in strikes]

    gamma_wall = max(strikes, key=lambda k: abs(by_strike[k])) if strikes else None
    gamma_pos = max(strikes, key=lambda k: by_strike[k]) if strikes else None
    gamma_neg = min(strikes, key=lambda k: by_strike[k]) if strikes else None
    gamma_flip = compute_flip(strikes, net, spot) if strikes else None

    return (
        snapshot_ts,
        session_date,
        UNDERLYING,
        DTE_MIN,
        DTE_MAX,
        spot,
        None,  # atm placeholder
        None, None, None,  # OI walls placeholder
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
        None,  # gamma proxy placeholder
        None,  # dealer hint placeholder
    )


def upsert(con, row):
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
          session_date        = excluded.session_date,
          spot                = excluded.spot,
          atm_strike          = excluded.atm_strike,
          max_total_oi_strike = excluded.max_total_oi_strike,
          max_call_oi_strike  = excluded.max_call_oi_strike,
          max_put_oi_strike   = excluded.max_put_oi_strike,
          gamma_wall_strike     = excluded.gamma_wall_strike,
          gamma_pos_wall_strike = excluded.gamma_pos_wall_strike,
          gamma_neg_wall_strike = excluded.gamma_neg_wall_strike,
          gamma_flip_strike     = excluded.gamma_flip_strike,
          total_call_oi       = excluded.total_call_oi,
          total_put_oi        = excluded.total_put_oi,
          pcr_oi              = excluded.pcr_oi,
          total_call_vol      = excluded.total_call_vol,
          total_put_vol       = excluded.total_put_vol,
          pcr_vol             = excluded.pcr_vol,
          gamma_proxy         = excluded.gamma_proxy,
          dealer_state_hint   = excluded.dealer_state_hint
        """,
        row,
    )
# ---------------- main ----------------

def main():
    con = connect()

    snaps = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT snapshot_ts FROM options_chain_snapshots ORDER BY snapshot_ts"
        )
    ]

    for s in snaps:
        row = compute_metrics(con, s)
        if row:
            upsert(con, row)

    con.commit()
    con.close()
    print("OK: gamma geometry integrated.")


if __name__ == "__main__":
    main()
