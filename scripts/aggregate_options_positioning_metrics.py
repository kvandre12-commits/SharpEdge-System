#!/usr/bin/env python3
"""
Aggregate options positioning metrics for SPY (or SYMBOL) into SQLite.
"""

import os
import sqlite3
from typing import Optional, List, Tuple, Dict
import numpy as np

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")
DTE_MIN = int(os.getenv("DTE_MIN", "0"))
DTE_MAX = int(os.getenv("DTE_MAX", "3"))
ONLY_SNAPSHOT_TS = os.getenv("SNAPSHOT_TS", "").strip()
COMPUTE_STATE = os.getenv("COMPUTE_DEALER_STATE", "1").strip() == "1"
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))


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


def pick_daily_table(con: sqlite3.Connection) -> Optional[str]:
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


def compute_flip(strikes: List[float], net: List[float], spot: Optional[float]) -> Optional[float]:
    if spot is None or len(strikes) < 2:
        return None

    s = np.array(strikes, dtype=float)
    g = np.array(net, dtype=float)

    order = np.argsort(s)
    s, g = s[order], g[order]

    for a in range(len(s) - 1):
        if np.sign(g[a]) != np.sign(g[a + 1]):
            x1, x2 = float(s[a]), float(s[a + 1])
            g1, g2 = float(g[a]), float(g[a + 1])
            if g2 == g1:
                return float(x1)
            return float(x1 + (0.0 - g1) * (x2 - x1) / (g2 - g1))

    return None


def compute_dealer_state(
    spot: Optional[float],
    gamma_flip: Optional[float],
    max_total_oi_strike: Optional[float],
    pcr_oi: Optional[float],
    pcr_vol: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:

    if spot is None:
        return None, None

    gamma_proxy = None
    dealer_hint = "NEUTRAL"

    if gamma_flip is not None:
        gamma_proxy = float(spot - gamma_flip)

    wall_distance_pct = None
    if max_total_oi_strike is not None:
        wall_distance_pct = abs(spot - max_total_oi_strike) / spot

    if gamma_proxy is not None:
        if gamma_proxy > 0:
            dealer_hint = "LONG_GAMMA"
        elif gamma_proxy < 0:
            dealer_hint = "SHORT_GAMMA"

    if wall_distance_pct is not None and wall_distance_pct <= PIN_THRESH_PCT:
        dealer_hint = "PINNED"

    if pcr_oi is not None:
        if pcr_oi > 1.4:
            dealer_hint = "DEFENSIVE"
        elif pcr_oi < 0.7 and dealer_hint != "PINNED":
            dealer_hint = "CHASE"

    if pcr_vol is not None and pcr_vol > 1.8:
        dealer_hint = "UNWIND_RISK"

    return gamma_proxy, dealer_hint


def pick_oi_expiry_col(con: sqlite3.Connection) -> Optional[str]:
    cols = column_names(con, "options_open_interest_daily")
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


def fetch_oi_by_strike(con, session_date, dte_min, dte_max):
    call_oi_by = {}
    put_oi_by = {}

    if not table_exists(con, "options_open_interest_daily"):
        return call_oi_by, put_oi_by

    exp_col = pick_oi_expiry_col(con)
    if not exp_col:
        return call_oi_by, put_oi_by

    und_col = pick_oi_underlying_col(con)
    sess_col = pick_oi_session_col(con)

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


def argmax_positive(d):
    if not d:
        return None
    k = max(d.keys(), key=lambda x: d[x])
    return float(k) if d[k] > 0 else None


def compute_metrics(con, snapshot_ts):
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

    total_call_vol = sum(r[4] or 0 for r in rows)
    total_put_vol = sum(r[5] or 0 for r in rows)
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol else None

    call_oi_by, put_oi_by = fetch_oi_by_strike(con, session_date, DTE_MIN, DTE_MAX)

    total_call_oi = sum(call_oi_by.values())
    total_put_oi = sum(put_oi_by.values())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else None

    strikes_oi = sorted(set(call_oi_by.keys()) | set(put_oi_by.keys()))

    max_call_oi_strike = argmax_positive(call_oi_by)
    max_put_oi_strike = argmax_positive(put_oi_by)
    total_oi_by = {k: call_oi_by.get(k, 0) + put_oi_by.get(k, 0) for k in strikes_oi}
    max_total_oi_strike = argmax_positive(total_oi_by)

    atm_strike = None
    if spot is not None:
        if strikes_oi:
            atm_strike = float(min(strikes_oi, key=lambda s: abs(s - spot)))

    by_strike = {}
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

    gamma_proxy, dealer_hint = compute_dealer_state(
        spot,
        gamma_flip,
        max_total_oi_strike,
        pcr_oi,
        pcr_vol,
    )

    return (
        snapshot_ts,
        session_date,
        UNDERLYING,
        DTE_MIN,
        DTE_MAX,
        spot,
        atm_strike,
        max_total_oi_strike,
        max_call_oi_strike,
        max_put_oi_strike,
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
