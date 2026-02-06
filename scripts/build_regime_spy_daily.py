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

def existing_cols(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}

def ensure_regime_table(con: sqlite3.Connection):
    # Base table
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

      compression_flag INTEGER,

      regime_label TEXT,
      regime_id TEXT,

      -- transitions
      vol_state_prev TEXT,
      vol_trend_state_prev TEXT,
      dp_state_prev TEXT,
      transition_label TEXT,
      transition_score INTEGER,
      transition_flag INTEGER,

      regime_ts TEXT,
      PRIMARY KEY (symbol, date)
    )
    """)
    con.commit()

    # Forward adds for older DBs
    cols = existing_cols(con, "regime_daily")
    adds = {
        "vol20": "REAL",
        "vol_rank_252": "REAL",
        "vol_state": "TEXT",
        "vol_trend_10": "REAL",
        "vol_trend_state": "TEXT",
        "dp_strength": "REAL",
        "dp_state": "TEXT",
        "compression_flag": "INTEGER",
        "regime_label": "TEXT",
        "regime_id": "TEXT",
        "vol_state_prev": "TEXT",
        "vol_trend_state_prev": "TEXT",
        "dp_state_prev": "TEXT",
        "transition_label": "TEXT",
        "transition_score": "INTEGER",
        "transition_flag": "INTEGER",
        "regime_ts": "TEXT",
    }
    for c, typ in adds.items():
        if c not in cols:
            con.execute(f"ALTER TABLE regime_daily ADD COLUMN {c} {typ}")
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
    # dp_strength is [0,1] from overlays_daily mapping
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

        # --- read features_daily safely (no tr_pct_rank dependency) ---
        fcols = existing_cols(con, "features_daily")
        needed = {"date", "symbol", "vol20", "compression_flag"}
        if not needed.issubset(fcols):
            raise RuntimeError(
                f"features_daily missing required columns: {sorted(list(needed - fcols))}. "
                "Run scripts/build_features_spy_daily.py and ensure features are written."
            )

        feats = pd.read_sql_query("""
            SELECT
              date, symbol,
              vol20,
              compression_flag
            FROM features_daily
            WHERE symbol = ?
            ORDER BY date ASC
        """, con, params=(SYMBOL,))

        if feats.empty:
            raise RuntimeError("features_daily is empty. Run build_features_spy_daily.py first.")

        # --- read darkpool overlay (optional) ---
        ocols = existing_cols(con, "overlays_daily")
        has_overlays = {"date", "symbol", "overlay_type", "overlay_strength"}.issubset(ocols)

        if has_overlays:
            dp = pd.read_sql_query("""
                SELECT date, symbol, overlay_strength AS dp_strength
                FROM overlays_daily
                WHERE symbol = ? AND overlay_type = 'darkpool'
                ORDER BY date ASC
            """, con, params=(SYMBOL,))
        else:
            dp = pd.DataFrame(columns=["date", "symbol", "dp_strength"])
# --- read macro overlays (optional, from overlays_daily) ---
        if has_overlays:
            macro = pd.read_sql_query("""
                SELECT date, symbol, overlay_type, overlay_strength
                FROM overlays_daily
                WHERE symbol = ?
                  AND overlay_type IN ('vix','vix3m','vix_term','rates10y')
                ORDER BY date ASC
            """, con, params=(SYMBOL,))
        else:
            macro = pd.DataFrame(columns=["date","symbol","overlay_type","overlay_strength"])

        if not macro.empty:
            macro_wide = (
                macro.pivot_table(
                    index=["date","symbol"],
                    columns="overlay_type",
                    values="overlay_strength",
                    aggfunc="last",
                )
                .reset_index()
            )
        else:
            macro_wide = pd.DataFrame(columns=["date","symbol","vix","vix3m","vix_term","rates10y"])

        # Merge macro into df (same pattern as dp)
        df = df.merge(macro_wide, on=["date","symbol"], how="left")

        for c in ["vix","vix3m","vix_term","rates10y"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = df[c].fillna(0.0)

        # Macro stress: conservative max-of (so any stress dimension can elevate state)
        df["macro_stress"] = df[["vix","vix_term","rates10y"]].max(axis=1)

        def bucket_macro(x: float) -> str:
            if pd.isna(x):
                return "unknown"
            if x >= 0.70:
                return "high"
            if x <= 0.30:
                return "low"
            return "normal"

        df["macro_state"] = df["macro_stress"].apply(bucket_macro)
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

        # Composite label (slow state)
        df["regime_label"] = (
            df["vol_state"].astype(str) + "_vol"
            + "|" + df["vol_trend_state"].astype(str) + "_voltrend"
            + "|" + df["dp_state"].astype(str) + "_dp"
            + "|" + df["compression_flag"].fillna(0).astype(int).astype(str) + "_comp"
        )

        # Compact stable key
        df["regime_id"] = (
            df["vol_state"].astype(str).str[:1]          # l/m/h/u
            + df["vol_trend_state"].astype(str).str[:1]  # r/f/f/u (we remap flat below)
            + df["dp_state"].astype(str).str[:1]         # h/n/l/u
            + df["compression_flag"].fillna(0).astype(int).astype(str)
        )
        # make flat explicitly "t" (for "flat")
        df.loc[df["vol_trend_state"] == "flat", "regime_id"] = (
            df.loc[df["vol_trend_state"] == "flat", "regime_id"].str.slice(0, 1) + "t"
            + df.loc[df["vol_trend_state"] == "flat", "regime_id"].str.slice(2)
        )

        # -----------------------------
        # Transitions (yesterday -> today)
        # -----------------------------
        df["vol_state_prev"] = df["vol_state"].shift(1)
        df["vol_trend_state_prev"] = df["vol_trend_state"].shift(1)
        df["dp_state_prev"] = df["dp_state"].shift(1)

        def changed(a, b) -> int:
            if pd.isna(a) or pd.isna(b):
                return 0
            return int(a != b)

        df["transition_score"] = (
            df.apply(lambda r: changed(r["vol_state_prev"], r["vol_state"]), axis=1)
            + df.apply(lambda r: changed(r["vol_trend_state_prev"], r["vol_trend_state"]), axis=1)
            + df.apply(lambda r: changed(r["dp_state_prev"], r["dp_state"]), axis=1)
        )

        df["transition_label"] = (
            df["vol_state_prev"].astype(str) + "->" + df["vol_state"].astype(str)
            + " | " + df["vol_trend_state_prev"].astype(str) + "->" + df["vol_trend_state"].astype(str)
            + " | " + df["dp_state_prev"].astype(str) + "->" + df["dp_state"].astype(str)
        )

        trend_flip_up = (
            df["vol_trend_state_prev"].isin(["falling", "flat"])
            & (df["vol_trend_state"] == "rising")
            & df["vol_state"].isin(["low", "mid"])
        )
        df["transition_flag"] = ((df["transition_score"] >= 2) | trend_flip_up).astype(int)

        df["regime_ts"] = datetime.now(timezone.utc).isoformat()

        # Write back
        cols = [
            "date","symbol",
            "vol20","vol_rank_252","vol_state",
            "vol_trend_10","vol_trend_state",
            "dp_strength","dp_state",
            "compression_flag",
            "regime_label","regime_id",
            "vol_state_prev","vol_trend_state_prev","dp_state_prev",
            "transition_label","transition_score","transition_flag",
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
          compression_flag=excluded.compression_flag,
          regime_label=excluded.regime_label,
          regime_id=excluded.regime_id,
          vol_state_prev=excluded.vol_state_prev,
          vol_trend_state_prev=excluded.vol_trend_state_prev,
          dp_state_prev=excluded.dp_state_prev,
          transition_label=excluded.transition_label,
          transition_score=excluded.transition_score,
          transition_flag=excluded.transition_flag,
          regime_ts=excluded.regime_ts
        """
        con.executemany(upsert, df[cols].to_records(index=False).tolist())
        con.commit()

        # CSV
        os.makedirs("outputs", exist_ok=True)
        df[cols].to_csv("outputs/spy_regime_daily.csv", index=False)
        print(f"OK: outputs/spy_regime_daily.csv | rows={len(df)}")

    finally:
        con.close()

if __name__ == "__main__":
    main()
