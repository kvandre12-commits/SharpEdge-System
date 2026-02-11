#!/usr/bin/env python3
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/spy_truth.db"
CSV_PATH = "outputs/attribution_daily.csv"

def main():
    con = sqlite3.connect(DB_PATH)

    # If table already exists and has rows, do nothing
    try:
        existing = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM edge_regime_pressure_dte_lut",
            con
        )
        if existing["n"].iloc[0] > 0:
            print("edge_regime_pressure_dte_lut already populated; skipping reload")
            con.close()
            return
    except Exception:
        pass  # table doesn't exist yet

    # Fallback: load from CSV
    df = pd.read_csv(CSV_PATH)
    # normalize
    df.columns = [c.strip().lower() for c in df.columns]
    df["regime"] = df["regime"].str.strip()
    df["pressure"] = df["pressure"].str.strip()
    df["dte"] = df["dte"].astype(str).str.strip()
    df["updated_ts"] = datetime.utcnow().isoformat()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS edge_regime_pressure_dte_lut (
      regime TEXT NOT NULL,
      pressure TEXT NOT NULL,
      dte TEXT NOT NULL,
      n INTEGER NOT NULL,
      win REAL,
      exp REAL,
      sharpe REAL,
      t REAL,
      maxdd REAL,
      updated_ts TEXT,
      PRIMARY KEY (regime, pressure, dte)
    )
    """)

    df.to_sql(
        "edge_regime_pressure_dte_lut",
        con,
        if_exists="replace",
        index=False
    )

    con.commit()
    con.close()
    print(f"Loaded {len(df)} rows into edge_regime_pressure_dte_lut")

if __name__ == "__main__":
    main()
