#!/usr/bin/env python3
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/spy_truth.db"
CSV_PATH = "outputs/attribution_daily.csv"  # change to outputs/edge_regime_pressure_dte.csv when you have it


def main():
    con = sqlite3.connect(DB_PATH)

    # If table already exists and has rows, do nothing
    try:
        existing = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM edge_regime_pressure_dte_lut",
            con
        )
        if int(existing["n"].iloc[0]) > 0:
            print("edge_regime_pressure_dte_lut already populated; skipping reload")
            con.close()
            return
    except Exception:
        pass  # table doesn't exist yet

    # Load from CSV
    df = pd.read_csv(CSV_PATH)

    # --- normalize column names ---
    df.columns = [c.strip().lower() for c in df.columns]

    # --- map possible upstream names to canonical schema ---
    rename_map = {
        "regime_label": "regime",
        "pressure_state": "pressure",
        "max_dd": "maxdd",
    }
    df = df.rename(columns=rename_map)

    # If we don't have a true LUT CSV, we can build a stub LUT from attribution_daily.csv
    if "regime" in df.columns and "pressure" in df.columns and "dte" not in df.columns:
        # attribution_daily-style input: build a minimal LUT using ret_1d_net
        if "ret_1d_net" not in df.columns:
            raise RuntimeError(
                "CSV looks like attribution_daily but missing ret_1d_net. "
                f"Have columns: {df.columns.tolist()}"
            )

        df["dte"] = "ALL"
        g = df.groupby(["regime", "pressure", "dte"], dropna=False)

        lut = g["ret_1d_net"].agg(n="count", exp="mean").reset_index()
        lut["win"] = g.apply(lambda x: (x["ret_1d_net"] > 0).mean()).values
        lut["sharpe"] = g.apply(
            lambda x: (x["ret_1d_net"].mean() / (x["ret_1d_net"].std(ddof=0) + 1e-12)) * (252 ** 0.5)
        ).values
        lut["t"] = None
        lut["maxdd"] = None
        df = lut

    # --- verify required columns exist (true LUT path) ---
    required = {"regime", "pressure", "dte", "n", "win", "exp"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"LUT CSV missing required columns: {missing}. "
            f"Have: {df.columns.tolist()}"
        )

    # --- clean strings + types ---
    df["regime"] = df["regime"].astype(str).str.strip()
    df["pressure"] = df["pressure"].astype(str).str.strip()
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
