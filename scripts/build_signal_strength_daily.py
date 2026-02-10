# scripts/build_signal_strength_daily.py
# Purpose:
#   Create a louder, earlier "readiness/ignition" signal using ONLY existing pipeline tables:
#     - features_daily (required)
#     - regime_daily (optional but recommended)
#     - overlays_daily (optional; darkpool overlay if present)
#
# Outputs:
#   - SQLite table: signals_daily
#   - outputs/spy_signal_strength_daily.csv
#   - outputs/latest_signal_strength.csv

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# --- knobs (keep small + sane) ---
TR_BASE_WIN = int(os.getenv("SIG_TR_BASE_WIN", "10"))          # baseline for TR lift
CLUSTER_SLOPE_WIN = int(os.getenv("SIG_CLUSTER_SLOPE_WIN", "5"))  # smooth cluster slope
MIN_PERIODS = int(os.getenv("SIG_MIN_PERIODS", "20"))

# --- FAST knobs (leading / protective) ---
TR_FAST_WIN = int(os.getenv("SIG_TR_FAST_WIN", "5"))
CLUSTER_FAST_WIN = int(os.getenv("SIG_CLUSTER_FAST_WIN", "1"))

PRESSURE_TR_ON = float(os.getenv("SIG_PRESSURE_TR_ON", "1.15"))
RELEASE_TR_ON  = float(os.getenv("SIG_RELEASE_TR_ON", "1.25"))

# Thresholds for buckets
LOUD_THRESH = float(os.getenv("SIG_LOUD_THRESH", "70"))
WATCH_THRESH = float(os.getenv("SIG_WATCH_THRESH", "50"))

# Weighting (0-100 total)
W_COMP = float(os.getenv("SIG_W_COMP", "25"))       # compression_flag
W_TRANS = float(os.getenv("SIG_W_TRANS", "20"))     # transition_flag
W_DP_MAX = float(os.getenv("SIG_W_DP_MAX", "15"))   # darkpool strength scaled 0..1
W_TR_LIFT_MAX = float(os.getenv("SIG_W_TR_LIFT_MAX", "20"))  # TR lift score
W_CLUSTER_MAX = float(os.getenv("SIG_W_CLUSTER_MAX", "20"))  # cluster building score

def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def existing_cols(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.OperationalError:
        return set()

def table_exists(con: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None

def ensure_signals_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS signals_daily (
      date TEXT NOT NULL,
      symbol TEXT NOT NULL,

      readiness_score REAL,
      ignition_score REAL,
      early_score REAL,
      early_bucket TEXT,

      dp_strength REAL,
      transition_flag INTEGER,
      compression_flag INTEGER,
      trigger_cluster REAL,
      trade_permission INTEGER,

      tr_lift REAL,
      cluster_slope REAL,

      signal_ts TEXT,
      signal_version TEXT,

      tr_lift_fast REAL,
      cluster_slope_fast REAL,

      pressure_state TEXT,
      trade_gate INTEGER,
      pressure_reason TEXT,

      PRIMARY KEY (symbol, date)
    )
    """)
    con.commit()

    # Forward adds for older DBs
    cols = existing_cols(con, "signals_daily")
    adds = {
        "readiness_score": "REAL",
        "ignition_score": "REAL",
        "early_score": "REAL",
        "early_bucket": "TEXT",
        "dp_strength": "REAL",
        "transition_flag": "INTEGER",
        "compression_flag": "INTEGER",
        "trigger_cluster": "REAL",
        "trade_permission": "INTEGER",
        "tr_lift": "REAL",
        "cluster_slope": "REAL",
        "signal_ts": "TEXT",
        "signal_version": "TEXT","tr_lift_fast": "REAL",
        "cluster_slope_fast": "REAL",
        "pressure_state": "TEXT",
        "trade_gate": "INTEGER",
        "pressure_reason": "TEXT",}
   
    for c, typ in adds.items():
        if c not in cols:
            con.execute(f"ALTER TABLE signals_daily ADD COLUMN {c} {typ}")
    con.commit()

def clamp01(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return np.nan

def score_tr_lift(tr_lift: float) -> float:
    """
    TR lift scoring:
      - 1.00 => 0 points
      - 1.10 => ~1/3 of max
      - 1.25 => ~2/3 of max
      - 1.50+ => max
    """
    if pd.isna(tr_lift) or tr_lift <= 1.0:
        return 0.0
    # Map [1.0, 1.5] -> [0,1], clamp
    s = (tr_lift - 1.0) / 0.5
    return float(max(0.0, min(1.0, s))) * W_TR_LIFT_MAX

def score_cluster_build(cluster_slope: float) -> float:
    """
    Cluster building scoring:
      - slope <= 0 => 0
      - slope ~0.25 => mild
      - slope ~0.5+ => strong
    """
    if pd.isna(cluster_slope) or cluster_slope <= 0:
        return 0.0
    s = cluster_slope / 0.5  # 0.5 slope ~= full strength
    return float(max(0.0, min(1.0, s))) * W_CLUSTER_MAX

def bucket(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= LOUD_THRESH:
        return "loud"
    if score >= WATCH_THRESH:
        return "watch"
    return "quiet"

def main():
    con = connect(DB_PATH)
    try:
        ensure_signals_table(con)

        # ---------------------------
        # features_daily (required)
        # ---------------------------
        if not table_exists(con, "features_daily"):
            raise RuntimeError("Missing features_daily table. Run scripts/build_features_spy_daily.py first.")

        fcols = existing_cols(con, "features_daily")
        required = {"date","symbol","true_range_pct","trigger_cluster","compression_flag","trade_permission"}
        missing = required - fcols
        if missing:
            raise RuntimeError(
                f"features_daily missing required columns: {sorted(list(missing))}. "
                "Rebuild features and ensure schema writes these fields."
            )

        feats = pd.read_sql_query("""
            SELECT
              date, symbol,
              true_range_pct,
              trigger_cluster,
              compression_flag,
              trade_permission
            FROM features_daily
            WHERE symbol = ?
            ORDER BY date ASC
        """, con, params=(SYMBOL,))
        if feats.empty:
            raise RuntimeError("features_daily has 0 rows for symbol. Nothing to score.")

        feats["date"] = pd.to_datetime(feats["date"])
        feats = feats.sort_values("date").reset_index(drop=True)

        # ---------------------------
        # overlays_daily darkpool (optional)
        # ---------------------------
        dp_strength = None
        if table_exists(con, "overlays_daily"):
            ocols = existing_cols(con, "overlays_daily")
            if {"date","symbol","overlay_type","overlay_strength"}.issubset(ocols):
                dp = pd.read_sql_query("""
                    SELECT date, symbol, overlay_strength AS dp_strength
                    FROM overlays_daily
                    WHERE symbol = ? AND overlay_type='darkpool'
                    ORDER BY date ASC
                """, con, params=(SYMBOL,))
                if not dp.empty:
                    dp["date"] = pd.to_datetime(dp["date"])
                    dp_strength = dp
        # else: no overlays table in this DB

        # ---------------------------
        # regime_daily transitions (optional)
        # ---------------------------
        trans = None
        if table_exists(con, "regime_daily"):
            rcols = existing_cols(con, "regime_daily")
            if {"date","symbol","transition_flag"}.issubset(rcols):
                r = pd.read_sql_query("""
                    SELECT date, symbol, transition_flag
                    FROM regime_daily
                    WHERE symbol = ?
                    ORDER BY date ASC
                """, con, params=(SYMBOL,))
                if not r.empty:
                    r["date"] = pd.to_datetime(r["date"])
                    trans = r

        # ---------------------------
        # Merge base frame
        # ---------------------------
        df = feats.copy()
        if dp_strength is not None:
            df = df.merge(dp_strength, on=["date","symbol"], how="left")
        else:
            df["dp_strength"] = np.nan

        if trans is not None:
            df = df.merge(trans, on=["date","symbol"], how="left")
        else:
            df["transition_flag"] = 0

        # ---------------------------
        # gamma overlay from options_positioning_metrics
        # ---------------------------
        gamma = None
        if table_exists(con, "options_positioning_metrics"):
            gcols = existing_cols(con, "options_positioning_metrics")
            needed = {
                "session_date","underlying","snapshot_ts",
                "spot","gamma_proxy","dealer_state_hint",
                "gamma_wall_strike","gamma_flip_strike"
            }
            if needed.issubset(gcols):
                g = pd.read_sql_query("""
                    WITH latest AS (
                      SELECT session_date, underlying, MAX(snapshot_ts) AS snapshot_ts
                      FROM options_positioning_metrics
                      WHERE underlying = ?
                      GROUP BY session_date, underlying
                    )
                    SELECT
                      m.session_date AS date,
                      m.underlying AS symbol,
                      m.spot,
                      m.gamma_proxy,
                      m.dealer_state_hint,
                      m.gamma_wall_strike,
                      m.gamma_flip_strike
                    FROM options_positioning_metrics m
                    JOIN latest l
                      ON m.session_date = l.session_date
                     AND m.underlying  = l.underlying
                     AND m.snapshot_ts = l.snapshot_ts
                    WHERE m.underlying = ?
                    ORDER BY m.session_date ASC
                """, con, params=(SYMBOL, SYMBOL))

                if not g.empty:
                    g["date"] = pd.to_datetime(g["date"])
                    gamma = g

        if gamma is not None:
            df = df.merge(gamma, on=["date","symbol"], how="left")
        else:
            df["spot"] = np.nan
            df["gamma_proxy"] = np.nan
            df["dealer_state_hint"] = None
            df["gamma_wall_strike"] = np.nan
            df["gamma_flip_strike"] = np.nan
        
        df["compression_flag"] = df["compression_flag"].fillna(0).astype(int)
        df["transition_flag"] = df["transition_flag"].fillna(0).astype(int)
        df["trade_permission"] = df["trade_permission"].fillna(0).astype(int)

        # ---------------------------
        # Ignition metrics (causal)
        # ---------------------------
        # TR baseline: rolling median (robust)
        df["tr_base_med"] = df["true_range_pct"].rolling(
            TR_BASE_WIN, min_periods=max(5, TR_BASE_WIN // 2)
        ).median()

        df["tr_lift"] = df["true_range_pct"] / df["tr_base_med"].replace(0, np.nan)

        # Cluster slope: (today - yesterday), smoothed
        df["cluster_delta"] = df["trigger_cluster"] - df["trigger_cluster"].shift(1)
        df["cluster_slope"] = df["cluster_delta"].rolling(
            CLUSTER_SLOPE_WIN, min_periods=1
        ).mean()

        # ---------------------------
        # Gamma regime interpretation
        # ---------------------------
        df["gamma_wall_dist_pct"] = (df["spot"] - df["gamma_wall_strike"]).abs() / df["spot"]
        df["gamma_flip_dist_pct"] = (df["spot"] - df["gamma_flip_strike"]).abs() / df["spot"]

        df["gamma_pin_flag"] = (
            (df["dealer_state_hint"] == "pin") |
            (df["gamma_wall_dist_pct"] <= 0.0025)
        ).astype(int)

        df["gamma_chase_flag"] = (df["gamma_proxy"] < 0).astype(int)
        
        # ---------------------------
        # FAST ignition metrics (leading / protective)
        # ---------------------------
        df["tr_base_med_fast"] = df["true_range_pct"].rolling(
        TR_FAST_WIN, min_periods=max(3, TR_FAST_WIN // 2)).median()

        df["tr_lift_fast"] = df["true_range_pct"] / df["tr_base_med_fast"].replace(0, np.nan)

        df["cluster_slope_fast"] = df["cluster_delta"].rolling(
        CLUSTER_FAST_WIN, min_periods=1).mean()
        
        # ---------------------------
        # Readiness score (0..60)
        # ---------------------------
        # Compression: +W_COMP
        comp_score = df["compression_flag"].astype(float) * W_COMP

        # Transition: +W_TRANS
        trans_score = df["transition_flag"].astype(float) * W_TRANS

        # Darkpool: dp_strength in [0,1] -> [0..W_DP_MAX]
        df["dp_strength"] = df["dp_strength"].apply(clamp01)
        dp_score = df["dp_strength"].fillna(0.0) * W_DP_MAX

        df["readiness_score"] = comp_score + trans_score + dp_score
        df["readiness_score"] = df["readiness_score"].clip(lower=0, upper=(W_COMP + W_TRANS + W_DP_MAX))

        # ---------------------------
        # Ignition score (0..40)
        # ---------------------------
        df["ign_tr_score"] = df["tr_lift"].apply(score_tr_lift)
        df["ign_cluster_score"] = df["cluster_slope"].apply(score_cluster_build)

        df["ignition_score"] = (df["ign_tr_score"] + df["ign_cluster_score"]).clip(
            lower=0, upper=(W_TR_LIFT_MAX + W_CLUSTER_MAX)
        )

        # ---------------------------
        # Early score (0..100)
        # ---------------------------
        df["early_score"] = (df["readiness_score"] + df["ignition_score"]).clip(lower=0, upper=100)
        df["early_bucket"] = df["early_score"].apply(bucket)
       
        # ---------------------------
        # Pressure state (fast protective layer)
        # ---------------------------
        def pressure_state_row(r) -> tuple[str, str]:
            # 1) Unresolved pressure: you FEEL it, but structure hasn't confirmed
            if (pd.notna(r["tr_lift_fast"]) and r["tr_lift_fast"] >= PRESSURE_TR_ON) and (
                r["early_bucket"] in ["quiet", "watch", "unknown"]
            ):
                reason = (
                    f"tr_lift_fast={r['tr_lift_fast']:.2f} >= {PRESSURE_TR_ON} "
                    f"but early_bucket={r['early_bucket']}"
                )
                return "UNRESOLVED_PRESSURE", reason

            # 2) Release: expansion + positive follow-through building
            if (pd.notna(r["tr_lift_fast"]) and r["tr_lift_fast"] >= RELEASE_TR_ON) and (
                pd.notna(r["cluster_slope_fast"]) and r["cluster_slope_fast"] > 0
            ):
                reason = (
                    f"tr_lift_fast={r['tr_lift_fast']:.2f} >= {RELEASE_TR_ON} "
                    f"and cluster_slope_fast={r['cluster_slope_fast']:.2f} > 0"
                )
                return "RELEASE", reason

            # 3) Coiled: compression flag present
            if int(r.get("compression_flag", 0) or 0) == 1:
                return "COILED", "compression_flag=1"

            return "NORMAL", "default"

        tmp = df.apply(pressure_state_row, axis=1, result_type="expand")
        df["pressure_state"] = tmp[0]
        df["pressure_reason"] = tmp[1]
        # ---------------------------
        # Intraday TR gate (hard lock)
        # ---------------------------
        intraday_gate = None
        if table_exists(con, "intraday_tr_gate"):
            ig = pd.read_sql_query("""
                WITH latest AS (
                  SELECT session_date, symbol, MAX(ts) AS ts
                  FROM intraday_tr_gate
                  WHERE symbol = ?
                  GROUP BY session_date, symbol
                )
                SELECT
                  g.session_date AS date,
                  g.symbol,
                  g.tr_lift_gate_75
                FROM intraday_tr_gate g
                JOIN latest l
                  ON g.session_date = l.session_date
                 AND g.symbol = l.symbol
                 AND g.ts = l.ts
                WHERE g.symbol = ?
                ORDER BY g.session_date ASC
            """, con, params=(SYMBOL, SYMBOL))

            if not ig.empty:
                ig["date"] = pd.to_datetime(ig["date"])
                intraday_gate = ig
            if.       intraday_gate is not None:
            df = df.merge(intraday_gate, on=["date","symbol"], how="left")
            df["tr_lift_gate_75"] = df["tr_lift_gate_75"].fillna(0).astype(int)ls.       e:
            # safest default: closed (forces you to actually build the intraday table)
            df["tr_lift_gate_75"] = 0
        # ---------------------------
        # Gamma permission layer
        # ---------------------------
        df["gamma_permission"] = (
            (df["gamma_chase_flag"] == 1) |
            ((df["gamma_pin_flag"] == 1) & (df["trade_permission"] == 1)) |
            ((df["gamma_flip_dist_pct"] <= 0.003) & (df["trade_permission"] == 1))
        ).astype(int)

        df["trade_gate"] = (
            (df["pressure_state"] == "RELEASE") &
            (df["gamma_permission"] == 1)
        ).astype(int)
        # Minimum history guard (avoid noisy early window)
        # If vol/tr baselines aren’t mature yet, mark bucket unknown but keep score computed.
        df["rows_seen"] = np.arange(1, len(df) + 1)
        df.loc[df["rows_seen"] < MIN_PERIODS, "early_bucket"] = "unknown"

        # ---------------------------
        # Write to DB + CSV
        # ---------------------------
        df["signal_ts"] = datetime.now(timezone.utc).isoformat()
        df["signal_version"] = "v1_early_readiness_ignition"

        out_cols = [
            "date","symbol",
            "readiness_score","ignition_score","early_score","early_bucket",
            "dp_strength","transition_flag","compression_flag",
            "trigger_cluster","trade_permission",
            "tr_lift","cluster_slope",
            "tr_lift_fast","cluster_slope_fast",
            "pressure_state","trade_gate","pressure_reason",
            "signal_ts","signal_version",
        ]

        # SQLite wants strings for date
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        upsert = f"""
        INSERT INTO signals_daily ({",".join(out_cols)})
        VALUES ({",".join(["?"] * len(out_cols))})
        ON CONFLICT(symbol, date) DO UPDATE SET
          readiness_score=excluded.readiness_score,
          ignition_score=excluded.ignition_score,
          early_score=excluded.early_score,
          early_bucket=excluded.early_bucket,
          dp_strength=excluded.dp_strength,
          transition_flag=excluded.transition_flag,
          compression_flag=excluded.compression_flag,
          trigger_cluster=excluded.trigger_cluster,
          trade_permission=excluded.trade_permission,
          tr_lift=excluded.tr_lift,
          cluster_slope=excluded.cluster_slope,
          tr_lift_fast=excluded.tr_lift_fast,
          cluster_slope_fast=excluded.cluster_slope_fast,
          pressure_state=excluded.pressure_state,
          trade_gate=excluded.trade_gate,
          pressure_reason=excluded.pressure_reason,
          signal_ts=excluded.signal_ts,
          signal_version=excluded.signal_version
        """
        con.executemany(upsert, df[out_cols].to_records(index=False).tolist())
        con.commit()

        os.makedirs("outputs", exist_ok=True)
        daily_path = "outputs/spy_signal_strength_daily.csv"
        df[out_cols].to_csv(daily_path, index=False)
        print(f"OK: {daily_path} | rows={len(df)}")

        # Latest row
        last = df.sort_values("date").iloc[-1]
        latest = pd.DataFrame([{
            "date": last["date"],
            "symbol": last["symbol"],
            "early_score": float(last["early_score"]),
            "early_bucket": str(last["early_bucket"]),
            "readiness_score": float(last["readiness_score"]),
            "ignition_score": float(last["ignition_score"]),
            "dp_strength": (None if pd.isna(last["dp_strength"]) else float(last["dp_strength"])),
            "transition_flag": int(last["transition_flag"]),
            "compression_flag": int(last["compression_flag"]),
            "trigger_cluster": (None if pd.isna(last["trigger_cluster"]) else float(last["trigger_cluster"])),
            "trade_permission": int(last["trade_permission"]),
            "tr_lift": (None if pd.isna(last["tr_lift"]) else float(last["tr_lift"])),
            "cluster_slope": (None if pd.isna(last["cluster_slope"]) else float(last["cluster_slope"])),
            "signal_version": last["signal_version"],"tr_lift_fast": (None if pd.isna(last["tr_lift_fast"]) else float(last["tr_lift_fast"])),
            "cluster_slope_fast": (None if pd.isna(last["cluster_slope_fast"]) else float(last["cluster_slope_fast"])),
            "pressure_state": str(last.get("pressure_state", "")),
            "trade_gate": int(last.get("trade_gate", 0)),
            "pressure_reason": str(last.get("pressure_reason", "")),
        }])
        latest_path = "outputs/latest_signal_strength.csv"
        latest.to_csv(latest_path, index=False)
        print(f"OK: {latest_path} (1 row)")

    finally:
        con.close()

if __name__ == "__main__":
    main()
