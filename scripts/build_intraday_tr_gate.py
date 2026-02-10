#!/usr/bin/env python3
"""
Build intraday-evolving TR-lift gate using session-so-far range and
time-of-day percentiles.

Inputs:
  - spy_bars_15m (ts, session_date, symbol, open, high, low, close, volume)
    from ingest_spy_intraday_alpaca.py
Outputs:
  - intraday_tr_gate

Gate definition:
  - tr_so_far_pct(ts) = (session_high_so_far - session_low_so_far) / session_open
  - For each time bucket (minutes_from_open rounded down to 15m),
    compute percentile rank vs last LOOKBACK_SESS prior sessions at that same bucket
  - tr_lift_gate_75 = 1 if percentile >= 0.75

This is designed for "check at mid-day and 3:30pm" behavior.
"""

import os
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
OUT_TABLE  = os.getenv("INTRADAY_TR_GATE_TABLE", "intraday_tr_gate")

BUCKET_MINUTES = int(os.getenv("TR_GATE_BUCKET_MINUTES", "15"))   # 15m bars
LOOKBACK_SESS  = int(os.getenv("TR_GATE_LOOKBACK_SESS", "20"))    # you chose 20
PCTL_THRESH    = float(os.getenv("TR_GATE_PCTL_THRESH", "0.75"))  # you chose 0.75
MIN_HIST       = int(os.getenv("TR_GATE_MIN_HIST", "10"))         # require some history

NY = ZoneInfo("America/New_York")

def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def table_exists(con, name: str) -> bool:
    r = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)).fetchone()
    return r is not None

def ensure_out_table(con):
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
      ts TEXT NOT NULL,
      session_date TEXT NOT NULL,
      symbol TEXT NOT NULL,

      minutes_from_open INTEGER NOT NULL,
      bucket_min INTEGER NOT NULL,

      session_open REAL,
      session_high_so_far REAL,
      session_low_so_far REAL,

      tr_so_far_pct REAL,
      tr_so_far_pctile REAL,
      hist_n INTEGER,
      tr_lift_gate_75 INTEGER,

      PRIMARY KEY (symbol, ts)
    )
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_sess ON {OUT_TABLE}(symbol, session_date)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_bucket ON {OUT_TABLE}(symbol, bucket_min, session_date)")
    con.commit()

def load_bars(con) -> pd.DataFrame:
    if not table_exists(con, BARS_TABLE):
        raise RuntimeError(f"Missing intraday bars table {BARS_TABLE}. Run your intraday ingest first.")
    df = pd.read_sql_query(
        f"""
        SELECT ts, session_date, symbol, open, high, low, close, volume
        FROM {BARS_TABLE}
        WHERE symbol = ?
        ORDER BY ts ASC
        """,
        con,
        params=(SYMBOL,),
    )
    if df.empty:
        raise RuntimeError(f"{BARS_TABLE} has 0 rows for {SYMBOL}.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    return df

def compute_minutes_from_open(ts_utc: pd.Series) -> pd.Series:
    # minutes from 09:30 NY for that session date
    ts_ny = ts_utc.dt.tz_convert(NY)
    sess_date = ts_ny.dt.date
    open_dt = pd.to_datetime(sess_date.astype(str) + " 09:30:00").dt.tz_localize(NY)
    mins = ((ts_ny - open_dt).dt.total_seconds() / 60.0).astype(int)
    return mins

def percentile_against_history(values_by_session: pd.Series, lookback: int, min_hist: int) -> pd.DataFrame:
    """
    values_by_session: indexed by session_date (string), sorted ascending
    Returns df with columns: pctile, hist_n
    pctile = fraction of historical values <= current (using last 'lookback' prior sessions)
    """
    sessions = values_by_session.index.to_list()
    vals = values_by_session.to_numpy(dtype=float)

    out_pct = np.full(len(vals), np.nan, dtype=float)
    out_n = np.zeros(len(vals), dtype=int)

    for i in range(len(vals)):
        # prior window
        start = max(0, i - lookback)
        hist = vals[start:i]  # prior only
        hist = hist[np.isfinite(hist)]
        n = len(hist)
        out_n[i] = n
        if n < min_hist or not np.isfinite(vals[i]):
            continue
        # percentile rank: P(hist <= current)
        out_pct[i] = float(np.sum(hist <= vals[i]) / n)

    return pd.DataFrame({"tr_so_far_pctile": out_pct, "hist_n": out_n}, index=values_by_session.index)

def main():
    con = connect()
    try:
        ensure_out_table(con)
        bars = load_bars(con)

        # --- compute intraday evolving TR so far ---
        bars["minutes_from_open"] = compute_minutes_from_open(bars["ts"])
        bars["bucket_min"] = (bars["minutes_from_open"] // BUCKET_MINUTES) * BUCKET_MINUTES

        # session_open = first open of the session
        bars["session_open"] = bars.groupby(["symbol", "session_date"])["open"].transform("first")
        bars["session_high_so_far"] = bars.groupby(["symbol", "session_date"])["high"].cummax()
        bars["session_low_so_far"]  = bars.groupby(["symbol", "session_date"])["low"].cummin()

        bars["tr_so_far_pct"] = (bars["session_high_so_far"] - bars["session_low_so_far"]) / bars["session_open"].replace(0, np.nan)

        # --- reduce to per-session, per-bucket (take last bar in bucket) ---
        # (with 15m bars this is naturally one bar per bucket, but this keeps it robust)
        bars = bars.sort_values(["session_date", "ts"]).reset_index(drop=True)
        per_bucket = (
            bars.groupby(["symbol", "session_date", "bucket_min"], as_index=False)
                .tail(1)
                .copy()
        )

        # --- compute time-of-day percentile vs last LOOKBACK_SESS sessions ---
        per_bucket["tr_so_far_pctile"] = np.nan
        per_bucket["hist_n"] = 0

        # For each bucket_min separately, percentile across sessions
        per_bucket = per_bucket.sort_values(["bucket_min", "session_date"]).reset_index(drop=True)
        for b, g in per_bucket.groupby("bucket_min", sort=True):
            # index by session_date for stable ordering
            s = g.set_index("session_date")["tr_so_far_pct"].sort_index()
            p = percentile_against_history(s, lookback=LOOKBACK_SESS, min_hist=MIN_HIST)
            # write back
            idx = per_bucket.index[per_bucket["bucket_min"] == b]
            # align by session_date
            per_bucket.loc[idx, "tr_so_far_pctile"] = per_bucket.loc[idx, "session_date"].map(p["tr_so_far_pctile"].to_dict())
            per_bucket.loc[idx, "hist_n"] = per_bucket.loc[idx, "session_date"].map(p["hist_n"].to_dict()).fillna(0).astype(int)

        per_bucket["tr_lift_gate_75"] = (
            (per_bucket["hist_n"] >= MIN_HIST) &
            (per_bucket["tr_so_far_pctile"] >= PCTL_THRESH)
        ).astype(int)

        # --- write out ---
        out = per_bucket[[
            "ts", "session_date", "symbol",
            "minutes_from_open", "bucket_min",
            "session_open", "session_high_so_far", "session_low_so_far",
            "tr_so_far_pct", "tr_so_far_pctile", "hist_n", "tr_lift_gate_75"
        ]].copy()

        out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = out.to_records(index=False).tolist()
        con.executemany(
            f"""
            INSERT OR REPLACE INTO {OUT_TABLE} (
              ts, session_date, symbol,
              minutes_from_open, bucket_min,
              session_open, session_high_so_far, session_low_so_far,
              tr_so_far_pct, tr_so_far_pctile, hist_n, tr_lift_gate_75
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        print(f"OK: wrote {len(rows)} rows into {OUT_TABLE} (lookback={LOOKBACK_SESS}, thresh={PCTL_THRESH})")

    finally:
        con.close()

if __name__ == "__main__":
    main()
