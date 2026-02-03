import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd


DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# Strategy knobs
ATR_WIN = int(os.getenv("ATR_WIN", "20"))
STOP_ATR_MULT = float(os.getenv("STOP_ATR_MULT", "1.0"))

# Aggression mapping (from strength_live)
AGG_MAP = {
    0: 0.0,
    1: 0.0,   # off / noise
    2: 0.5,   # probe
    3: 1.0,   # normal
    4: 1.5,   # hot
    5: 2.0,   # extreme
}

# Overlay weights (tune later)
OVERLAY_WEIGHTS = {
    "fomc": 1.0,
    "tariff": 0.8,
    "darkpool": 1.2,
    "vix": 1.0,
    "vix9d": 1.0,
    "vix_term": 1.2,
    "rates10y": 0.6,
}

def connect():
    return sqlite3.connect(DB_PATH)

def table_exists(con, name: str) -> bool:
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (name,)).fetchone() is not None

def read_bars(con):
    q = """
    SELECT date, symbol, open, high, low, close, volume
    FROM bars_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    return df

def read_features(con):
    if not table_exists(con, "features_daily"):
        return None
    q = """
    SELECT
      date, symbol,
      compression_flag,
      trigger_cluster,
      trigger_gap_15,
      permission_strength,
      trade_permission
    FROM features_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    df["date"] = pd.to_datetime(df["date"])
    return df

def read_overlays(con):
    if not table_exists(con, "overlays_daily"):
        return None
    q = """
    SELECT date, symbol, overlay_type, overlay_strength
    FROM overlays_daily
    WHERE symbol = ?
    """
    o = pd.read_sql_query(q, con, params=(SYMBOL,))
    o["date"] = pd.to_datetime(o["date"])
    o["overlay_type"] = o["overlay_type"].astype(str).str.lower()
    o["overlay_strength"] = pd.to_numeric(o["overlay_strength"], errors="coerce").fillna(0.0)
    return o

def compute_atr(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = pd.Series(tr).rolling(ATR_WIN, min_periods=max(5, ATR_WIN//2)).mean()
    return atr

def overlay_boost_for_day(overlays_day: pd.DataFrame) -> float:
    # overlays_day contains rows for one date: overlay_type, overlay_strength
    if overlays_day is None or overlays_day.empty:
        return 0.0
    boost = 0.0
    for _, r in overlays_day.iterrows():
        w = OVERLAY_WEIGHTS.get(r["overlay_type"], 0.0)
        boost += w * float(r["overlay_strength"])
    return float(boost)

def strength_live(base_strength: int, overlay_boost: float) -> int:
    # Boost strength in 2 steps; tune thresholds later
    add = 0
    if overlay_boost >= 0.8:
        add += 1
    if overlay_boost >= 1.5:
        add += 1
    s = int(base_strength) + add
    return max(0, min(5, s))

def main():
    con = connect()
    try:
        bars = read_bars(con)
        feats = read_features(con)
        overlays = read_overlays(con)

        # Merge features if present, else create minimal columns
        df = bars.copy()
        if feats is not None and not feats.empty:
            df = df.merge(feats, on=["date","symbol"], how="left")
        else:
            df["permission_strength"] = 0
            df["trade_permission"] = 0
            df["trigger_cluster"] = np.nan
            df["compression_flag"] = np.nan
            df["trigger_gap_15"] = np.nan

        # Build overlay boost series
        if overlays is not None and not overlays.empty:
            # group overlays by date
            ob = overlays.groupby("date", as_index=False).apply(lambda g: overlay_boost_for_day(g)).reset_index()
            ob.columns = ["_idx","date","overlay_boost"]
            df = df.merge(ob[["date","overlay_boost"]], on="date", how="left")
        else:
            df["overlay_boost"] = 0.0

        df["overlay_boost"] = df["overlay_boost"].fillna(0.0)

        # Compute ATR for stop sizing
        df["atr"] = compute_atr(df)

        # Prior day levels
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)

        # Base strength: if permission_strength exists use it; else derive a crude one from trigger_cluster/compression
        base_strength = df["permission_strength"].fillna(0).astype(int)

        # Strength live
        df["permission_strength_live"] = [
            strength_live(bs, ob) for bs, ob in zip(base_strength.tolist(), df["overlay_boost"].tolist())
        ]
        df["aggression_mult"] = df["permission_strength_live"].map(AGG_MAP).fillna(0.0)

        # Same-day breakout triggers
        df["long_trigger"] = (df["high"] > df["prev_high"]) & df["prev_high"].notna()
        df["short_trigger"] = (df["low"] < df["prev_low"]) & df["prev_low"].notna()

        # Build trades
        trades = []

        for i, r in df.iterrows():
            date = r["date"]
            atr = r["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            # If aggression is 0, skip (permission window off)
            aggr = float(r["aggression_mult"])
            if aggr <= 0:
                continue

            # Long breakout
            if bool(r["long_trigger"]):
                entry = float(r["prev_high"])
                stop = entry - STOP_ATR_MULT * float(atr)

                # Worst-case rule: if low <= stop, assume stopped before close
                if float(r["low"]) <= stop:
                    exit_px = stop
                    outcome = "stop"
                else:
                    exit_px = float(r["close"])
                    outcome = "close"

                ret = (exit_px - entry) / entry
                trades.append({
                    "date": date,
                    "side": "long",
                    "entry": entry,
                    "exit": exit_px,
                    "outcome": outcome,
                    "ret": ret,
                    "aggression_mult": aggr,
                    "permission_strength_live": int(r["permission_strength_live"]),
                    "overlay_boost": float(r["overlay_boost"]),
                })

            # Short breakout
            if bool(r["short_trigger"]):
                entry = float(r["prev_low"])
                stop = entry + STOP_ATR_MULT * float(atr)

                # Worst-case rule: if high >= stop, assume stopped
                if float(r["high"]) >= stop:
                    exit_px = stop
                    outcome = "stop"
                else:
                    exit_px = float(r["close"])
                    outcome = "close"

                ret = (entry - exit_px) / entry  # short return
                trades.append({
                    "date": date,
                    "side": "short",
                    "entry": entry,
                    "exit": exit_px,
                    "outcome": outcome,
                    "ret": ret,
                    "aggression_mult": aggr,
                    "permission_strength_live": int(r["permission_strength_live"]),
                    "overlay_boost": float(r["overlay_boost"]),
                })

        trades_df = pd.DataFrame(trades)
        os.makedirs("outputs", exist_ok=True)

        if trades_df.empty:
            trades_df.to_csv("outputs/breakout_backtest_trades.csv", index=False)
            print("No trades generated (check aggression mapping, data, or ATR window).")
            return

        # Weighted return (aggression applied)
        trades_df["ret_weighted"] = trades_df["ret"] * trades_df["aggression_mult"]

        # Summary by strength
        def summarize(g):
            n = len(g)
            avg = g["ret"].mean()
            avg_w = g["ret_weighted"].mean()
            win = (g["ret"] > 0).mean()
            stop_rate = (g["outcome"] == "stop").mean()
            return pd.Series({
                "n": n,
                "avg_ret": avg,
                "avg_ret_weighted": avg_w,
                "winrate": win,
                "stop_rate": stop_rate,
                "avg_aggr": g["aggression_mult"].mean(),
            })

        summary = trades_df.groupby(["permission_strength_live", "side"]).apply(summarize).reset_index()

        trades_path = "outputs/breakout_backtest_trades.csv"
        summary_path = "outputs/breakout_backtest_summary.csv"
        trades_df.to_csv(trades_path, index=False)
        summary.to_csv(summary_path, index=False)

        print(f"Wrote {trades_path} ({len(trades_df)} trades)")
        print(f"Wrote {summary_path}")
        print(summary.to_string(index=False))

    finally:
        con.close()

if __name__ == "__main__":
    main()
