#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = os.getenv("OUTDIR", "outputs")

MIN_N = int(os.getenv("EDGE_MIN_N", "25"))          # minimum samples per bucket
ANNUALIZE = int(os.getenv("EDGE_ANNUALIZE", "252")) # daily annualization factor

OUT_CSV = os.path.join(OUTDIR, "edge_regime_pressure_dte.csv")
OUT_MD  = os.path.join(OUTDIR, "edge_regime_pressure_dte_top.md")


# -------------------------
# helpers
# -------------------------
def q1(con: sqlite3.Connection, sql: str, params=()) -> Optional[tuple]:
    cur = con.execute(sql, params)
    return cur.fetchone()

def table_exists(con: sqlite3.Connection, name: str) -> bool:
    r = q1(con, "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return r is not None

def cols(con: sqlite3.Connection, name: str) -> List[str]:
    # works for tables; views may return empty in older sqlite builds
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]
    except Exception:
        return []

def pick_first_existing(con: sqlite3.Connection, candidates: List[str]) -> Optional[str]:
    for t in candidates:
        if table_exists(con, t):
            return t
    return None

def safe_sharpe(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.sqrt(ANNUALIZE) * float(r.mean()) / sd)

def t_stat(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(mu / (sd / np.sqrt(len(r))))

def max_drawdown_from_returns(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())

def fail(msg: str):
    raise SystemExit(f"\nEXPECTANCY ENGINE FAIL: {msg}\n")


# -------------------------
# core: build joined frame
# -------------------------
def build_joined_df(con: sqlite3.Connection) -> pd.DataFrame:
    # Required base tables
    if not table_exists(con, "signals_daily"):
        fail("Missing table: signals_daily")
    if not table_exists(con, "features_daily"):
        fail("Missing table: features_daily")

    # Regime table naming mismatch (your common wiring bug)
    regime_table = pick_first_existing(con, ["spy_regime_daily", "regime_daily"])
    if regime_table is None:
        # Allow running without it, but label will be UNKNOWN
        regime_table = None

    # Liquidity regimes are often in liquidity_regime_events
    liq_table = "liquidity_regime_events" if table_exists(con, "liquidity_regime_events") else None

    # DTE bucket could live in multiple places depending on your build phase
    exec_table = "execution_state_daily" if table_exists(con, "execution_state_daily") else None
    trade_signals_table = "trade_signals" if table_exists(con, "trade_signals") else None

    # Figure out where DTE bucket exists
    dte_source = None
    dte_col = None

    # 1) execution_state_daily
    if exec_table:
        c = cols(con, exec_table)
        for cand in ["dte_bucket", "dte_recommendation", "dte_choice", "dte_selected"]:
            if cand in c:
                dte_source = exec_table
                dte_col = cand
                break

    # 2) trade_signals (if you persist)
    if dte_source is None and trade_signals_table:
        c = cols(con, trade_signals_table)
        if "dte_bucket" in c:
            dte_source = trade_signals_table
            dte_col = "dte_bucket"

    # 3) signals_daily (fallback)
    if dte_source is None:
        c = cols(con, "signals_daily")
        for cand in ["dte_bucket", "dte_recommendation", "dte_choice", "dte_selected"]:
            if cand in c:
                dte_source = "signals_daily"
                dte_col = cand
                break

    # Build query
