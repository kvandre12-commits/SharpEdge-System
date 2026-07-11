#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dateutil import tz

try:
    from scripts.utils.pipeline_state import write_state
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import write_state

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
TIMEFRAME = os.getenv("INTRADAY_TIMEFRAME", "15Min")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets")
ALPACA_FEED = os.getenv("ALPACA_BAR_FEED", "iex")
ALPACA_ADJUSTMENT = os.getenv("ALPACA_BAR_ADJUSTMENT", "raw")
REQUEST_TIMEOUT = int(os.getenv("ALPACA_BAR_TIMEOUT", "30"))
PAGE_LIMIT = int(os.getenv("ALPACA_BAR_LIMIT", "10000"))

BAR_COLUMNS = [
    "ts",
    "session_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]

try:
    NY = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:  # Termux tzdata can be feral.
    NY = tz.gettz("America/New_York")
    if NY is None:
        raise


BarRow = dict[str, Any]


def parse_utc_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table_name})")}


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BARS_TABLE} (
          ts TEXT NOT NULL,
          session_date TEXT NOT NULL,
          symbol TEXT NOT NULL,
          open REAL NOT NULL,
          high REAL NOT NULL,
          low REAL NOT NULL,
          close REAL NOT NULL,
          volume REAL,
          trade_count INTEGER,
          vwap REAL,
          PRIMARY KEY (symbol, ts)
        )
        """
    )
    existing = table_columns(con, BARS_TABLE)
    if "trade_count" not in existing:
        con.execute(f"ALTER TABLE {BARS_TABLE} ADD COLUMN trade_count INTEGER")
    if "vwap" not in existing:
        con.execute(f"ALTER TABLE {BARS_TABLE} ADD COLUMN vwap REAL")
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{BARS_TABLE}_session ON {BARS_TABLE}(symbol, session_date)"
    )
    con.commit()


def last_ts(con: sqlite3.Connection) -> str | None:
    row = con.execute(
        f"SELECT MAX(ts) FROM {BARS_TABLE} WHERE symbol = ?",
        (SYMBOL,),
    ).fetchone()
    return row[0] if row and row[0] else None


def normalize_bar(raw_bar: dict[str, Any]) -> BarRow:
    raw_ts = raw_bar.get("t")
    if not raw_ts:
        raise RuntimeError(f"Alpaca bar missing timestamp: {raw_bar}")

    ts = parse_utc_timestamp(str(raw_ts))
    return {
        "ts": format_utc_z(ts),
        "session_date": ts.astimezone(NY).date().isoformat(),
        "symbol": SYMBOL,
        "open": raw_bar.get("o"),
        "high": raw_bar.get("h"),
        "low": raw_bar.get("l"),
        "close": raw_bar.get("c"),
        "volume": raw_bar.get("v"),
        "trade_count": raw_bar.get("n"),
        "vwap": raw_bar.get("vw"),
    }


def fetch_bars(
    start_iso: str | None, end_iso: str | None = None
) -> tuple[list[BarRow], dict[str, Any]]:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars.")

    url = f"{ALPACA_DATA_BASE}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params: dict[str, Any] = {
        "symbols": SYMBOL,
        "timeframe": TIMEFRAME,
        "limit": PAGE_LIMIT,
        "adjustment": ALPACA_ADJUSTMENT,
        "feed": ALPACA_FEED,
        "sort": "asc",
    }
    if start_iso:
        params["start"] = start_iso
    if end_iso:
        params["end"] = end_iso

    rows: list[BarRow] = []
    page_count = 0
    page_token = None

    while True:
        request_params = dict(params)
        if page_token:
            request_params["page_token"] = page_token
        response = requests.get(
            url, headers=headers, params=request_params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
        page_count += 1

        page_bars = ((payload.get("bars") or {}).get(SYMBOL)) or []
        rows.extend(normalize_bar(bar) for bar in page_bars)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    metadata = {
        "endpoint": url,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "feed": ALPACA_FEED,
        "adjustment": ALPACA_ADJUSTMENT,
        "requested_start": start_iso,
        "requested_end": end_iso,
        "page_limit": PAGE_LIMIT,
        "pages_fetched": page_count,
        "raw_bar_count": len(rows),
        "earliest_fetched_ts": rows[0]["ts"] if rows else None,
        "latest_fetched_ts": rows[-1]["ts"] if rows else None,
    }
    return rows, metadata


def upsert(con: sqlite3.Connection, rows: list[BarRow]) -> int:
    if not rows:
        return 0

    values = [tuple(row[column] for column in BAR_COLUMNS) for row in rows]
    con.executemany(
        f"""
        INSERT INTO {BARS_TABLE}
        (ts, session_date, symbol, open, high, low, close, volume, trade_count, vwap)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          session_date=excluded.session_date,
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume,
          trade_count=excluded.trade_count,
          vwap=excluded.vwap
        """,
        values,
    )
    con.commit()
    return len(values)


def intraday_state(con: sqlite3.Connection) -> dict[str, Any]:
    row = con.execute(
        f"""
        SELECT COUNT(*),
               MIN(ts),
               MAX(ts),
               COUNT(DISTINCT session_date),
               MIN(session_date),
               MAX(session_date)
        FROM {BARS_TABLE}
        WHERE symbol = ?
        """,
        (SYMBOL,),
    ).fetchone()
    return {
        "rows": row[0] or 0,
        "earliest_ts": row[1],
        "latest_ts": row[2],
        "session_count": row[3] or 0,
        "earliest_session_date": row[4],
        "latest_session_date": row[5],
        "latest_date": row[5],
    }


def incremental_start_iso(existing_last_ts: str | None) -> str | None:
    if not existing_last_ts:
        return None
    return format_utc_z(parse_utc_timestamp(existing_last_ts) + timedelta(seconds=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default=os.getenv("MODE", "incremental"),
        choices=["incremental", "backfill"],
    )
    ap.add_argument(
        "--start",
        default=os.getenv("START_ISO"),
        help="ISO start, e.g. 2025-01-01T00:00:00Z",
    )
    ap.add_argument(
        "--end",
        default=os.getenv("END_ISO"),
        help="ISO end, e.g. 2025-02-01T00:00:00Z",
    )
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        before = intraday_state(con)

        if args.mode == "incremental":
            start_iso = incremental_start_iso(last_ts(con))
            rows, source_details = fetch_bars(start_iso=start_iso)
        else:
            if not args.start:
                raise RuntimeError(
                    "Backfill mode requires --start (or START_ISO env var)."
                )
            start_iso = args.start
            rows, source_details = fetch_bars(start_iso=args.start, end_iso=args.end)

        upserted = upsert(con, rows)
        after = intraday_state(con)
    finally:
        con.close()

    latest_ingest_ts = datetime.now(timezone.utc).isoformat()
    write_state(
        "intraday_bars",
        {
            "symbol": SYMBOL,
            "table": BARS_TABLE,
            "source": "alpaca_bars",
            "timeframe": TIMEFRAME,
            "feed": ALPACA_FEED,
            "adjustment": ALPACA_ADJUSTMENT,
            "mode": args.mode,
            "start": start_iso,
            "end": args.end,
            "requested_start": source_details.get("requested_start"),
            "requested_end": source_details.get("requested_end"),
            "fetched_rows": len(rows),
            "upserted_rows": upserted,
            "latest_ingest_ts": latest_ingest_ts,
            "latest_date": after.get("latest_session_date"),
            "source_details": source_details,
            "before": before,
            "after": after,
        },
    )
    print(
        f"OK: wrote {upserted} bars into {BARS_TABLE} "
        f"(timeframe={TIMEFRAME}, feed={ALPACA_FEED}) latest={after.get('latest_ts')}"
    )


if __name__ == "__main__":
    main()
