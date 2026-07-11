#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
OUTDIR = os.getenv("OUTDIR", "outputs")
OUT_PATH = os.path.join(OUTDIR, "liquidity_regime_events.csv")


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute("SELECT * FROM liquidity_regime_events ORDER BY session_date")
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        with open(OUT_PATH, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"OK: wrote {len(rows)} rows -> {OUT_PATH}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
