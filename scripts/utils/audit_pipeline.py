#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")

CONTRACT = {
    "features_daily": ["date", "symbol", "ret_1d"],
    "regime_daily": ["date", "symbol", "vol_state", "regime_label"],
    "signals_daily": ["session_date", "symbol", "pressure_state"],
    "execution_state_daily": [
        "session_date",
        "symbol",
        "execution_score",
        "final_bias",
        "dealer_state_hint",
        "wall_strike",
    ],
    "options_positioning_metrics": [
        "session_date",
        "underlying",
        "gamma_proxy",
        "dealer_state_hint",
        "max_total_oi_strike",
    ],
    "liquidity_regime_events": ["session_date", "underlying", "regime_type"],
}


def table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def cols(cur: sqlite3.Cursor, table_name: str) -> list[str]:
    return [
        row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    ]


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        print(f"\nDB: {DB_PATH}\n")

        for table_name, required in CONTRACT.items():
            print("=" * 72)
            print(f"TABLE: {table_name}")

            if not table_exists(cur, table_name):
                print("  MISSING TABLE")
                continue

            column_names = cols(cur, table_name)
            missing_cols = [name for name in required if name not in column_names]
            if missing_cols:
                print(f"  MISSING COLUMNS: {missing_cols}")

            row_count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  rows: {row_count}")

            date_col = next(
                (
                    candidate
                    for candidate in ["session_date", "date"]
                    if candidate in column_names
                ),
                None,
            )
            if date_col:
                latest = cur.execute(
                    f"SELECT MAX({date_col}) FROM {table_name}"
                ).fetchone()[0]
                print(f"  latest_{date_col}: {latest}")

            for column in required:
                if column not in column_names:
                    continue
                nulls = cur.execute(
                    f"SELECT SUM(CASE WHEN {column} IS NULL OR {column}='' THEN 1 ELSE 0 END) FROM {table_name}"
                ).fetchone()[0]
                pct = (nulls / row_count * 100.0) if row_count else 0.0
                print(f"  {column}: nulls={nulls} ({pct:.1f}%)")

            if (
                table_name == "options_positioning_metrics"
                and "dealer_state_hint" in column_names
            ):
                latest_non_null = cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM options_positioning_metrics
                    WHERE dealer_state_hint IS NOT NULL
                      AND dealer_state_hint != ''
                    """
                ).fetchone()[0]
                if latest_non_null == 0:
                    print("  WARNING: dealer_state_hint entirely NULL")

        print("\nDONE\n")
    finally:
        con.close()


if __name__ == "__main__":
    main()
