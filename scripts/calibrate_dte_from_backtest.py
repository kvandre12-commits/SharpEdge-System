#!/usr/bin/env python3
import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("data/spy_truth.db")
BACKTEST_CSV = Path("outputs/failed_breakdown_backtest.csv")

# Map horizons → DTE buckets
HORIZONS = [
    ("0-1", "ret_fwd_1"),
    ("2-3", "ret_fwd_2"),
    ("5-7", "ret_fwd_5"),
]

def hit_rate(s: pd.Series) -> float:
    s = s.dropna()
    return float((s > 0).mean()) if len(s) else float("nan")

def robust_score(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    return float(hit_rate(s) * s.median())

def main():
    df = pd.read_csv(BACKTEST_CSV)

    required = {"ret_fwd_1", "ret_fwd_2", "ret_fwd_5", "is_favorable_slow"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    rows = []

    # ALL signals
    for bucket, col in HORIZONS:
        rows.append({
            "subset": "ALL",
            "dte_bucket": bucket,
            "score": robust_score(df[col])
        })

    # Slow subtype
    slow = df[df["is_favorable_slow"] == 1]
    for bucket, col in HORIZONS:
        rows.append({
            "subset": "SLOW",
            "dte_bucket": bucket,
            "score": robust_score(slow[col])
        })

    scores = pd.DataFrame(rows)

    def best_bucket(subset):
        block = scores[scores["subset"] == subset]
        return block.sort_values("score", ascending=False).iloc[0]["dte_bucket"]

    best_all = best_bucket("ALL")
    best_slow = best_bucket("SLOW")

    print("Calibration result:")
    print(scores)
    print("\nRecommended:")
    print("ALL  →", best_all)
    print("SLOW →", best_slow)

    # Persist to DB
    con = sqlite3.connect(DB_PATH)
    con.execute("""
      CREATE TABLE IF NOT EXISTS dte_calibration (
        rule_id TEXT PRIMARY KEY,
        default_bucket TEXT,
        slow_bucket TEXT,
        notes TEXT
      )
    """)
    con.execute("""
      INSERT OR REPLACE INTO dte_calibration
      (rule_id, default_bucket, slow_bucket, notes)
      VALUES (?, ?, ?, ?)
    """, (
        "FB_LONG_OPT_V1",
        best_all,
        best_slow,
        "Calibrated from failed_breakdown_backtest.csv"
    ))
    con.commit()
    con.close()

if __name__ == "__main__":
    main() 
