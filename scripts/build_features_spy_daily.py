--- a/scripts/build_features_spy_daily.py
+++ b/scripts/build_features_spy_daily.py
@@
 import os
 import sqlite3
 from datetime import datetime, timezone
 
 import numpy as np
 import pandas as pd
@@
 EDGE_SHARE_FLAG = float(os.getenv("EDGE_SHARE_FLAG", "0.60"))
 
 def connect(db_path: str) -> sqlite3.Connection:
     os.makedirs(os.path.dirname(db_path), exist_ok=True)
     return sqlite3.connect(db_path)
 
+def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
+    """Divide with 0/NaN protection."""
+    b2 = b.replace(0, np.nan)
+    return (a / b2).replace([np.inf, -np.inf], np.nan)
 
 def read_truth(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
@@
 def build_features(df: pd.DataFrame, con: sqlite3.Connection | None = None) -> pd.DataFrame:
     out = df.copy()
 
     # Prev close + returns
     out["prev_close"] = out["close"].shift(1)
-    out["ret_1d"] = (out["close"] / out["prev_close"]) - 1.0
+    out["ret_1d"] = safe_div(out["close"], out["prev_close"]) - 1.0
 
     # Gap
-    out["gap_open_pct"] = (out["open"] / out["prev_close"]) - 1.0
+    # Interday gap is correctly normalized by prev_close
+    out["gap_open_pct"] = safe_div(out["open"], out["prev_close"]) - 1.0
     out["gap_abs_pct"] = out["gap_open_pct"].abs()
 
     # Range
     out["intraday_range"] = out["high"] - out["low"]
-    out["intraday_range_pct"] = out["intraday_range"] / out["prev_close"]
+    # Intraday % should be "% of price today" => normalize by close
+    out["intraday_range_pct"] = safe_div(out["intraday_range"], out["close"])
 
     # True range
     hl = out["high"] - out["low"]
     hc = (out["high"] - out["prev_close"]).abs()
     lc = (out["low"] - out["prev_close"]).abs()
     out["true_range"] = np.maximum(hl, np.maximum(hc, lc))
-    out["true_range_pct"] = out["true_range"] / out["prev_close"]
+    out["true_range_pct"] = safe_div(out["true_range"], out["close"])
 
     # Vol
     out["vol20"] = out["ret_1d"].rolling(VOL_WIN, min_periods=max(5, VOL_WIN // 2)).std()
@@
     out["trigger_gap_15"] = (out["gap_abs_pct"] >= TRIGGER_PCT).astype(int)
     out["trigger_range_15"] = (out["intraday_range_pct"].abs() >= TRIGGER_PCT).astype(int)
-    out["trigger_tr_15"] = (out["true_range_pct"].abs() >= TRIGGER_PCT).astype(int)
+    # Keep TR trigger as severity, but avoid double counting (TR can subsume gap/range)
+    out["trigger_tr_15"] = (out["true_range_pct"].abs() >= TRIGGER_PCT).astype(int)
 
     out["trigger_any_15"] = (
         (out["trigger_gap_15"] == 1)
         | (out["trigger_range_15"] == 1)
-        | (out["trigger_tr_15"] == 1)
     ).astype(int)
 
     out["trigger_cluster"] = out["trigger_any_15"].rolling(CLUSTER_WIN, min_periods=1).sum()
@@
     out["tr_pct_rank"] = out["true_range_pct"].rolling(
         LOOKBACK_COMPRESSION, min_periods=max(5, LOOKBACK_COMPRESSION // 2)
     ).apply(pct_rank_last, raw=False)
     out["compression_flag"] = (out["tr_pct_rank"] <= COMP_PCTL).astype(int)
@@
     out["next_tr_pct_rank"] = out["next_tr_pct"].rolling(
         LOOKBACK_EXPANSION, min_periods=max(5, LOOKBACK_EXPANSION // 2)
     ).apply(pct_rank_last, raw=False)
-    out["next_day_expansion"] = (out["next_tr_pct_rank"] >= EXP_PCTL).astype(int)
+    # LABEL (uses future data) — keep clearly separated from causal features
+    out["label_next_day_expansion"] = (out["next_tr_pct_rank"] >= EXP_PCTL).astype(int)
@@
     cols = [
         "date", "symbol",
         "open", "high", "low", "close", "volume",
         "prev_close",
         "ret_1d",
         "gap_open_pct", "gap_abs_pct",
         "intraday_range_pct",
         "true_range_pct",
         "vol20",
         "cluster_score",
 
         "trigger_gap_15",
         "trigger_range_15",
         "trigger_tr_15",
         "trigger_any_15",
         "trigger_cluster",
 
         "compression_flag",
-        "next_day_expansion",
+        "label_next_day_expansion",
 
         "permission_strength",
         "permission_reason",
         "trade_permission",
+
+        # ORB / edge (persisted)
+        "open_orb_range",
+        "close_orb_range",
+        "open_orb_share",
+        "close_orb_share",
+        "edge_orb_share",
+        "edge_orb_bias",
+        "edge_orb_flag",
     ]
     feats = out[cols].copy()
 
     feats["feature_ts"] = datetime.now(timezone.utc).isoformat()
     feats["feature_version"] = "v2_triggers_permission_strength"
 
     feats = feats.dropna(subset=["prev_close"]).reset_index(drop=True)
@@
             feats["ats_pressure_flag"] = (
                 feats["ats_ratio"]
                 >= feats["ats_ratio"]
                 .rolling(20, min_periods=5)
                 .quantile(0.80)
             ).astype(int)
@@
     else:
         feats["ats_ratio"] = np.nan
         feats["ats_pressure_flag"] = 0
+
+    # Persist FINRA ATS fields (institutional contract)
+    # (Keep them as part of the features table so signal/regime builders can join if desired.)
 
     # -----------------------------
     # Trend day vs non-trend day
@@
     out["day_type"] = out.apply(classify_day, axis=1)
-    return feats
+    # Bring day_type into feats (computed on 'out' but must be published)
+    feats = feats.merge(out[["date", "day_type"]], on="date", how="left")
+    return feats
 
 
 def ensure_features_table(con: sqlite3.Connection):
     con.execute("""
     CREATE TABLE IF NOT EXISTS features_daily (
         date TEXT NOT NULL,
         symbol TEXT NOT NULL,
         open REAL,
         high REAL,
         low REAL,
         close REAL,
         volume REAL,
         prev_close REAL,
         ret_1d REAL,
         gap_open_pct REAL,
         gap_abs_pct REAL,
         intraday_range_pct REAL,
         true_range_pct REAL,
         vol20 REAL,
         cluster_score REAL,
 
         trigger_gap_15 INTEGER,
         trigger_range_15 INTEGER,
         trigger_tr_15 INTEGER,
         trigger_any_15 INTEGER,
         trigger_cluster REAL,
 
         compression_flag INTEGER,
-        next_day_expansion INTEGER,
+        label_next_day_expansion INTEGER,
 
         permission_strength INTEGER,
         permission_reason TEXT,
         trade_permission INTEGER,
 
+        open_orb_range REAL,
+        close_orb_range REAL,
+        open_orb_share REAL,
+        close_orb_share REAL,
+        edge_orb_share REAL,
+        edge_orb_bias REAL,
+        edge_orb_flag INTEGER,
+
+        ats_ratio REAL,
+        ats_pressure_flag INTEGER,
+
+        day_type TEXT,
+
         feature_ts TEXT,
         feature_version TEXT,
         PRIMARY KEY (symbol, date)
     )
     """)
     con.commit()
@@
     adds = {
         "trigger_gap_15": "INTEGER",
         "trigger_range_15": "INTEGER",
         "trigger_tr_15": "INTEGER",
         "trigger_any_15": "INTEGER",
         "trigger_cluster": "REAL",
         "permission_strength": "INTEGER",
         "permission_reason": "TEXT",
         "trade_permission": "INTEGER",
+
+        "label_next_day_expansion": "INTEGER",
+
+        "open_orb_range": "REAL",
+        "close_orb_range": "REAL",
+        "open_orb_share": "REAL",
+        "close_orb_share": "REAL",
+        "edge_orb_share": "REAL",
+        "edge_orb_bias": "REAL",
+        "edge_orb_flag": "INTEGER",
+
+        "ats_ratio": "REAL",
+        "ats_pressure_flag": "INTEGER",
+        "day_type": "TEXT",
     }
     for col, typ in adds.items():
         if col not in existing:
             con.execute(f"ALTER TABLE features_daily ADD COLUMN {col} {typ}")
+    con.commit()
+
+def upsert_features(con: sqlite3.Connection, feats: pd.DataFrame) -> None:
+    if feats.empty:
+        return
+
+    # Ensure date is TEXT
+    feats2 = feats.copy()
+    feats2["date"] = pd.to_datetime(feats2["date"]).dt.strftime("%Y-%m-%d")
+
+    out_cols = list(feats2.columns)
+    placeholders = ",".join(["?"] * len(out_cols))
+    col_list = ",".join(out_cols)
+
+    # Update everything except PK
+    updates = ",".join([f"{c}=excluded.{c}" for c in out_cols if c not in ["symbol", "date"]])
+
+    sql = f"""
+    INSERT INTO features_daily ({col_list})
+    VALUES ({placeholders})
+    ON CONFLICT(symbol, date) DO UPDATE SET
+      {updates}
+    """
+    con.executemany(sql, feats2[out_cols].to_records(index=False).tolist())
+    con.commit()
@@
 def main():
     con = connect(DB_PATH)
     try:
+        ensure_features_table(con)
         truth = read_truth(con, SYMBOL)
-        feats = build_features(truth)
+        # PASS con so ORB runs (institutional: features must be reproducible from DB)
+        feats = build_features(truth, con=con)
         write_csv(feats)
         write_latest_signal(feats)
+        upsert_features(con, feats)
         print("OK: features + latest signal built from truth.")
     finally:
         con.close()
