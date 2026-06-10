#!/usr/bin/env python3
"""Incrementally ingest daily SPY bars from yfinance."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

try:
    from scripts.utils.pipeline_state import is_fresh, write_state
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import is_fresh, write_state

SYMBOL = os.getenv("SYMBOL", "SPY")
DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SOURCE = "yfinance"
FULL_PERIOD = os.getenv("DAILY_FULL_PERIOD", "2y")
INCREMENTAL_PERIOD = os.getenv("DAILY_INCREMENTAL_PERIOD", "10d")
CACHE_TTL_HOURS = float(os.getenv("DAILY_CACHE_TTL_HOURS", "12"))
FORCE_REFRESH = os.getenv("DAILY_FORCE_REFRESH", "0").strip().lower() in {"1", "true", "yes", "y"}


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


def fetch_daily(period: str) -> pd.DataFrame:
    frame = yf.download(
        SYMBOL,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]

    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    if "datetime" in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    if "index" in frame.columns:
        frame = frame.rename(columns={"index": "date"})
    if "date" not in frame.columns:
        raise RuntimeError(f"Could not find date column after yfinance download. Columns={list(frame.columns)}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    out = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    out["symbol"] = SYMBOL
    out["source"] = SOURCE
    out["ingest_ts"] = datetime.now(timezone.utc).isoformat()
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if out.empty:
        raise RuntimeError(f"No rows returned for symbol={SYMBOL}, period={period}")

    return out


def upsert_truth(con: sqlite3.Connection, truth: pd.DataFrame) -> int:
    ensure_truth_table(con)
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "source", "ingest_ts"]
    rows = truth[cols].to_records(index=False).tolist()
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
    frame = pd.read_sql_query(
        """
        SELECT date, symbol, open, high, low, close, volume, source, ingest_ts
        FROM bars_daily
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        con,
        params=(SYMBOL,),
    )
    path = Path(f"outputs/{SYMBOL.lower()}_truth_daily.csv")
    frame.to_csv(path, index=False)
    return len(frame)


def main() -> None:
    con = connect(DB_PATH)
    try:
        before = latest_state(con)
        network_refresh = not should_skip_network(before)
        period = fetch_period_for_state(before)
        upserted = 0

        if network_refresh:
            truth = fetch_daily(period=period)
            upserted = upsert_truth(con, truth)

        csv_rows = write_truth_csv(con)
        after = latest_state(con)
    finally:
        con.close()

    state = {
        "symbol": SYMBOL,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "force_refresh": FORCE_REFRESH,
        "network_refresh": network_refresh,
        "period": period,
        "upserted_rows": upserted,
        "csv_rows": csv_rows,
        "before": before,
        "after": after,
    }
    write_state("daily_bars", state)
    mode = "network_refresh" if network_refresh else "cache_export_only"
    print(
        f"OK: {SYMBOL} daily bars | mode={mode} | period={period} | "
        f"upserted={upserted} | csv_rows={csv_rows} | latest={after.get('latest_date')}"
    )


if __name__ == "__main__":
    main()
