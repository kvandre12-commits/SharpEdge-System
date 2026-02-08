#!/usr/bin/env python3
import os, sqlite3
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
OUTDIR = os.getenv("OUTDIR", "outputs")
OUT_PATH = os.path.join(OUTDIR, "liquidity_regime_events.csv")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM liquidity_regime_events ORDER BY session_date",
            con
        )
        df.to_csv(OUT_PATH, index=False)
        print(f"OK: wrote {len(df)} rows -> {OUT_PATH}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
