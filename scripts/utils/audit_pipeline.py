#!/usr/bin/env python3
import os, sqlite3
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")

CONTRACT = {
    "features_daily": ["date", "symbol", "ret_1d"],
    "spy_regime_daily": ["date", "symbol", "vol_state", "regime_label"],
    "signals_daily": ["date", "symbol", "trade_gate", "pressure_state"],
    "execution_state_daily": ["date", "symbol", "trade_gate"],
    "liquidity_regime_events": ["session_date", "underlying", "regime_type"],
}

def table_exists(cur, t):
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

def cols(cur, t):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]

def main():
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        print(f"\nDB: {DB_PATH}\n")

        for t, required in CONTRACT.items():
            print("="*72)
            print(f"TABLE: {t}")

            if not table_exists(cur, t):
                print("  MISSING TABLE")
                continue

            c = cols(cur, t)
            missing_cols = [x for x in required if x not in c]
            if missing_cols:
                print(f"  MISSING COLUMNS: {missing_cols}")

            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  rows: {n}")

            # best-effort latest date
            date_col = "date" if "date" in c else ("session_date" if "session_date" in c else None)
            if date_col:
                latest = cur.execute(f"SELECT MAX({date_col}) FROM {t}").fetchone()[0]
                print(f"  latest_{date_col}: {latest}")

            # null-rate for required columns that exist
            for col in required:
                if col not in c:
                    continue
                nulls = cur.execute(f"SELECT SUM(CASE WHEN {col} IS NULL OR {col}='' THEN 1 ELSE 0 END) FROM {t}").fetchone()[0]
                pct = (nulls / n * 100.0) if n else 0.0
                print(f"  {col}: nulls={nulls} ({pct:.1f}%)")

        print("\nDONE\n")
    finally:
        con.close()

if __name__ == "__main__":
    main()
