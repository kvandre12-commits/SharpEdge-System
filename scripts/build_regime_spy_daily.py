# scripts/build_regime_spy_daily.py
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# --- knobs (keep simple) ---
VOL_RANK_WIN = int(os.getenv("REGIME_VOL_RANK_WIN", "252"))     # ~1y trading days
VOL_TREND_LB = int(os.getenv("REGIME_VOL_TREND_LB", "10"))      # slope lookback
VOL_TREND_EPS = float(os.getenv("REGIME_VOL_TREND_EPS", "0.0")) # deadzone

def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def ensure_regime_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS regime_daily (
      date TEXT NOT NULL,
      symbol TEXT NOT NULL,

      vol20 REAL,
      vol_rank_252 REAL,
      vol_state TEXT,

      vol_trend_10 REAL,
      vol_trend_state TEXT,

      dp_strength REAL,
      dp_state TEXT,

      tr_pct_rank REAL,
      compression_flag INTEGER,

      regime_label TEXT,
      regime_id TEXT,

      regime_ts TEXT,
      PRIMARY KEY (symbol, date)
    )
    """)
    con.commit()

def pct_rank_last(x: pd.Series) -> float:
    if x.isna().all():
        return np.nan
    last = x.iloc[-1]
    return float((x <= last).mean())

def bucket_vol_state(vol_rank: float) -> str:
    if pd.isna(vol_rank):
        return "unknown"
    if vol_rank < 0.20:
        return "low"
    if vol_rank > 0.80:
        return "high"
    return "mid"

def bucket_trend(delta: float, eps: float) -> str:
    if pd.isna(delta):
        return "unknown"
    if delta > eps:
        return "rising"
    if delta < -eps:
        return "falling"
    return "flat"

def bucket_dp(dp_strength: float) -> str:
    # dp_strength is already [0,1] from your overlay mapping
    if pd.isna(dp_strength):
        return "unknown"
    if dp_strength >= 0.70:
        return "high"
    if dp_strength <= 0.30:
        return "low"
    return "normal"

def main():
    con = connect(DB_PATH)
    try:
        ensure_regime_table(con)

        # Pull from features_daily (vol20, compression flags)
        feats = pd.read_sql_query("""
            SELECT
              date, symbol,
              vol20,
              tr_pct_rank,
              compression_flag
            FROM features_daily
            WHERE symbol = ?
            ORDER BY date ASC
        """, con, params=(SYMBOL,))

        if feats.empty:
            raise RuntimeError("features_daily is empty. Run build_features_spy_daily.py first.")

        # Pull darkpool overlay strength (daily) from overlays_daily
        dp = pd.read_sql_query("""
            SELECT date, symbol, overlay_strength AS dp_strength
            FROM overlays_daily
            WHERE symbol = ? AND overlay_type = 'darkpool'
            ORDER BY date ASC
        """, con, params=(SYMBOL,))

        # Merge
        df = feats.merge(dp, on=["date", "symbol"], how="left")

        # Vol rank over trailing window (causal)
        df["vol_rank_252"] = (
            df["vol20"]
            .rolling(VOL_RANK_WIN, min_periods=max(30, VOL_RANK_WIN // 4))
            .apply(pct_rank_last, raw=False)
        )

        df["vol_state"] = df["vol_rank_252"].apply(bucket_vol_state)

        # Vol trend: delta over lookback (causal)
        df["vol_trend_10"] = df["vol20"] - df["vol20"].shift(VOL_TREND_LB)
        df["vol_trend_state"] = df["vol_trend_10"].apply(lambda x: bucket_trend(x, VOL_TREND_EPS))

        # Darkpool pressure state
        df["dp_state"] = df["dp_strength"].apply(bucket_dp)

        # Composite labels
        df["regime_label"] = (
            df["vol_state"].astype(str) + "_vol"
            + "|" + df["vol_trend_state"].astype(str) + "_voltrend"
            + "|" + df["dp_state"].astype(str) + "_dp"
            + "|" + df["compression_flag"].fillna(0).astype(int).astype(str) + "_comp"
        )

        # Compact stable key
        df["regime_id"] = (
            df["vol_state"].astype(str).str[:1]  # l/m/h/u
            + df["vol_trend_state"].astype(str).str[:1]  # r/f/f/u (flat -> f)
            + df["dp_state"].astype(str).str[:1]  # h/n/l/u
            + df["compression_flag"].fillna(0).astype(int).astype(str)
        )
        df.loc[df["vol_trend_state"] == "flat", "regime_id"] = (
            df.loc[df["vol_trend_state"] == "flat", "regime_id"].str.slice(0, 1) + "t"
            + df.loc[df["vol_trend_state"] == "flat", "regime_id"].str.slice(2)
        )

        df["regime_ts"] = datetime.now(timezone.utc).isoformat()

        # Write back
        cols = [
            "date","symbol",
            "vol20","vol_rank_252","vol_state",
            "vol_trend_10","vol_trend_state",
            "dp_strength","dp_state",
            "tr_pct_rank","compression_flag",
            "regime_label","regime_id",
            "regime_ts"
        ]

        upsert = f"""
        INSERT INTO regime_daily ({",".join(cols)})
        VALUES ({",".join(["?"]*len(cols))})
        ON CONFLICT(symbol, date) DO UPDATE SET
          vol20=excluded.vol20,
          vol_rank_252=excluded.vol_rank_252,
          vol_state=excluded.vol_state,
          vol_trend_10=excluded.vol_trend_10,
          vol_trend_state=excluded.vol_trend_state,
          dp_strength=excluded.dp_strength,
          dp_state=excluded.dp_state,
          tr_pct_rank=excluded.tr_pct_rank,
          compression_flag=excluded.compression_flag,
          regime_label=excluded.regime_label,
          regime_id=excluded.regime_id,
          regime_ts=excluded.regime_ts
        """
        con.executemany(upsert, df[cols].to_records(index=False).tolist())
        con.commit()

        # Optional CSV output
        os.makedirs("outputs", exist_ok=True)
        df[cols].to_csv("outputs/spy_regime_daily.csv", index=False)
        print(f"OK: outputs/spy_regime_daily.csv | rows={len(df)}")

    finally:
        con.close()

if __name__ == "__main__":
    main()
