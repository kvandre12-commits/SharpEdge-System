import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("data/spy_truth.db")

TRUTH_TABLE_CANDIDATES = [
    "truth_daily",
    "ohlc_daily",
    "bars_daily",
    "spy_daily",
]

REQUIRED_TABLES = [
    "features_daily",
]


def fail(msg: str):
    print(f"VALIDATION FAILED: {msg}")
    sys.exit(1)


def find_truth_table(cur):
    for t in TRUTH_TABLE_CANDIDATES:
        row = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (t,),
        ).fetchone()
        if row:
            return t
    return None


def check_row_count(cur, table):
    count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if count == 0:
        fail(f"{table} is empty")
    print(f"{table}: {count} rows OK")


def main():
    if not DB_PATH.exists():
        fail("Database missing")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # --- truth table discovery ---
    truth = find_truth_table(cur)
    if not truth:
        fail("No daily truth table found")

    print(f"Truth table detected: {truth}")
    check_row_count(cur, truth)

    # --- required tables ---
    for table in REQUIRED_TABLES:
        row = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row:
            fail(f"Missing table: {table}")

        check_row_count(cur, table)

        # latest date freshness check
        latest = cur.execute(
            f"SELECT MAX(date) FROM {table}"
        ).fetchone()[0]

        print(f"{table} latest date: {latest}")

    print("DB VALIDATION PASSED")


if __name__ == "__main__":
    main()
