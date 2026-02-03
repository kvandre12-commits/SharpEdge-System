#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
TIMEFRAME = os.getenv("INTRADAY_TIMEFRAME", "15Min")  # "5Min" later
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

# Alpaca data endpoint (v2). For most accounts this works:
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets")

NY = ZoneInfo("America/New_York")


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {BARS_TABLE} (
      ts TEXT NOT NULL,
      session_date TEXT NOT NULL,
      symbol TEXT NOT NULL,
      open REAL NOT NULL,
      high REAL NOT NULL,
      low REAL NOT NULL,
      close REAL NOT NULL,
      volume REAL,
      PRIMARY KEY (symbol, ts)
    )
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{BARS_TABLE}_session ON {BARS_TABLE}(symbol, session_date)")
    con.commit()


def last_ts(con: sqlite3.Connection) -> str | None:
    row = con.execute(
        f"SELECT MAX(ts) FROM {BARS_TABLE} WHERE symbol = ?",
        (SYMBOL,),
    ).fetchone()
    return row[0] if row and row[0] else None


def fetch_bars(start_iso: str | None, end_iso: str | None = None) -> pd.DataFrame:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars.")

    url = f"{ALPACA_DATA_BASE}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    params = {
        "symbols": SYMBOL,
        "timeframe": TIMEFRAME,
        "limit": 10000,     # Alpaca paginates; we loop with next_page_token
        "adjustment": "raw",
        "feed": "sip",      # if your plan allows; otherwise you may need "iex"
        "sort": "asc",
    }
    if start_iso:
        params["start"] = start_iso
    if end_iso:
        params["end"] = end_iso

    out = []
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()

        bars = (j.get("bars") or {}).get(SYMBOL, [])
        out.extend(bars)

        page_token = j.get("next_page_token")
        if not page_token:
            break

    if not out:
        return pd.DataFrame(columns=["ts","o","h","l","c","v"])

    df = pd.DataFrame(out)
    # Alpaca returns "t" for time, o/h/l/c/v columns
    df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def compute_session_date_yyyy_mm_dd(ts_utc: pd.Series) -> pd.Series:
    # Convert UTC -> NY date (session date)
    return ts_utc.dt.tz_convert(NY).dt.date.astype(str)


def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    df["session_date"] = compute_session_date_yyyy_mm_dd(df["ts"])
    df["symbol"] = SYMBOL
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = df[["ts","session_date","symbol","open","high","low","close","volume"]].to_records(index=False).tolist()

    con.executemany(
        f"""
        INSERT OR REPLACE INTO {BARS_TABLE}
        (ts, session_date, symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    return len(rows)


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)

        # Continue from last timestamp
        last = last_ts(con)
        start = None
        if last:
            # add 1 second to avoid re-pulling last bar
            dt = datetime.fromisoformat(last.replace("Z","+00:00")).astimezone(timezone.utc)
            start = (dt + pd.Timedelta(seconds=1)).isoformat().replace("+00:00","Z")

        df = fetch_bars(start_iso=start)
        n = upsert(con, df)
        print(f"OK: wrote {n} bars into {BARS_TABLE} (timeframe={TIMEFRAME})")
    finally:
        con.close()


if __name__ == "__main__":
    main()
