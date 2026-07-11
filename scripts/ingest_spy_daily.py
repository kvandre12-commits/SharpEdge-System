#!/usr/bin/env python3
"""Incrementally ingest daily SPY bars from Yahoo chart data."""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cockpit.market_data_sources import fetch_yahoo_daily_bars  # noqa: E402

try:
    from scripts.utils.pipeline_state import is_fresh, write_state
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import is_fresh, write_state

SYMBOL = os.getenv("SYMBOL", "SPY")
DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SOURCE = "yahoo_chart"
FULL_PERIOD = os.getenv("DAILY_FULL_PERIOD", "2y")
INCREMENTAL_PERIOD = os.getenv("DAILY_INCREMENTAL_PERIOD", "10d")
CACHE_TTL_HOURS = float(os.getenv("DAILY_CACHE_TTL_HOURS", "12"))
FORCE_REFRESH = os.getenv("DAILY_FORCE_REFRESH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
TIMEOUT = int(os.getenv("YAHOO_DAILY_TIMEOUT", "20"))
CSV_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "ingest_ts",
]

BarRow = dict[str, Any]


def connect(db_path: str) -> sqlite3.Connection:
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return sqlite3.connect(db_path)


def ensure_truth_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bars_daily (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            ingest_ts TEXT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    con.commit()


def latest_state(con: sqlite3.Connection) -> dict[str, Any]:
    ensure_truth_table(con)
    row = con.execute(
        """
        SELECT COUNT(*) AS rows,
               MIN(date) AS earliest_date,
               MAX(date) AS latest_date,
               MAX(ingest_ts) AS latest_ingest_ts
        FROM bars_daily
        WHERE symbol = ?
        """,
        (SYMBOL,),
    ).fetchone()
    return {
        "rows": row[0] or 0,
        "earliest_date": row[1],
        "latest_date": row[2],
        "latest_ingest_ts": row[3],
    }


def should_skip_network(state: dict[str, Any]) -> bool:
    if FORCE_REFRESH or not state.get("rows"):
        return False
    return is_fresh(state.get("latest_ingest_ts"), CACHE_TTL_HOURS)


def fetch_period_for_state(state: dict[str, Any]) -> str:
    if FORCE_REFRESH or not state.get("rows"):
        return FULL_PERIOD
    return INCREMENTAL_PERIOD


def fetch_daily(period: str) -> tuple[list[BarRow], dict[str, Any]]:
    raw_rows, source = fetch_yahoo_daily_bars(
        SYMBOL,
        interval="1d",
        range_=period,
        timeout=TIMEOUT,
    )
    ingest_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "date": row["date"],
                "symbol": SYMBOL,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "source": SOURCE,
                "ingest_ts": ingest_ts,
            }
        )
    if not rows:
        raise RuntimeError(f"No rows returned for symbol={SYMBOL}, period={period}")
    return rows, source


def upsert_truth(con: sqlite3.Connection, truth: list[BarRow]) -> int:
    ensure_truth_table(con)
    rows = [tuple(row[column] for column in CSV_COLUMNS) for row in truth]
    con.executemany(
        """
        INSERT INTO bars_daily (date, symbol, open, high, low, close, volume, source, ingest_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source,
            ingest_ts=excluded.ingest_ts
        """,
        rows,
    )
    con.commit()
    return len(rows)


def write_truth_csv(con: sqlite3.Connection) -> int:
    os.makedirs("outputs", exist_ok=True)
    rows = con.execute(
        """
        SELECT date, symbol, open, high, low, close, volume, source, ingest_ts
        FROM bars_daily
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        (SYMBOL,),
    ).fetchall()
    path = Path(f"outputs/{SYMBOL.lower()}_truth_daily.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    con = connect(DB_PATH)
    source_details: dict[str, Any] = {}
    try:
        before = latest_state(con)
        network_refresh = not should_skip_network(before)
        period = fetch_period_for_state(before)
        upserted = 0

        if network_refresh:
            truth, source_details = fetch_daily(period=period)
            upserted = upsert_truth(con, truth)

        csv_rows = write_truth_csv(con)
        after = latest_state(con)
    finally:
        con.close()

    state = {
        "symbol": SYMBOL,
        "source": SOURCE,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "force_refresh": FORCE_REFRESH,
        "network_refresh": network_refresh,
        "period": period,
        "timeout": TIMEOUT,
        "upserted_rows": upserted,
        "csv_rows": csv_rows,
        "source_details": source_details,
        "before": before,
        "after": after,
    }
    write_state("daily_bars", state)
    mode = "network_refresh" if network_refresh else "cache_export_only"
    print(
        f"OK: {SYMBOL} daily bars | mode={mode} | source={SOURCE} | period={period} | "
        f"upserted={upserted} | csv_rows={csv_rows} | latest={after.get('latest_date')}"
    )


if __name__ == "__main__":
    main()
