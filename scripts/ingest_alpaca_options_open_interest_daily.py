#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone
import time
import requests
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("UNDERLYING", "SPY")

ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets")
ALPACA_KEY = os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

ALPACA_RETRIES = int(os.getenv("ALPACA_RETRIES", "4"))
ALPACA_TIMEOUT = int(os.getenv("ALPACA_TIMEOUT", "30"))
ALPACA_FAIL_OPEN = os.getenv("ALPACA_FAIL_OPEN", "1") == "1"

def iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

def ny_session_date(now_utc: datetime) -> str:
    # Keep your existing implementation if you already have one elsewhere.
    # Minimal: use UTC date (OK for now; replace with NY logic later if you want).
    return now_utc.date().isoformat()

def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def ensure_table(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS options_open_interest_daily (
      snapshot_ts TEXT NOT NULL,
      session_date TEXT NOT NULL,
      underlying TEXT NOT NULL,
      expiration_date TEXT NOT NULL,
      strike REAL NOT NULL,
      call_oi INTEGER NOT NULL DEFAULT 0,
      put_oi INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (session_date, underlying, expiration_date, strike)
    );
    """)
    con.commit()

def upsert_row(con, row):
    con.execute("""
    INSERT INTO options_open_interest_daily (
      snapshot_ts, session_date, underlying, expiration_date, strike, call_oi, put_oi
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_date, underlying, expiration_date, strike) DO UPDATE SET
      snapshot_ts=excluded.snapshot_ts,
      call_oi=excluded.call_oi,
      put_oi=excluded.put_oi;
    """, row)

def alpaca_get_contracts_page(underlying: str, page_token: str | None):
    url = f"{ALPACA_TRADING_BASE}/v2/options/contracts"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY or "",
        "APCA-API-SECRET-KEY": ALPACA_SECRET or "",
    }
    params = {
        "underlying_symbols": underlying,
        "limit": 1000,
        # You can add more filters later if desired:
        # "status": "active",
    }
    if page_token:
        params["page_token"] = page_token

    r = requests.get(url, headers=headers, params=params, timeout=ALPACA_TIMEOUT)
    r.raise_for_status()
    return r.json()

def fetch_all_contracts(underlying: str):
    page_token = None
    out = []
    while True:
        last_err = None
        for attempt in range(ALPACA_RETRIES):
            try:
                payload = alpaca_get_contracts_page(underlying, page_token)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Alpaca contracts fetch failed: {last_err}")

        contracts = payload.get("option_contracts") or payload.get("contracts") or []
        out.extend(contracts)

        page_token = payload.get("next_page_token") or payload.get("page_token")
        if not page_token:
            return out

def main():
    snap_ts = iso_utc_now()
    session = ny_session_date(datetime.now(timezone.utc))

    con = connect()
    try:
        ensure_table(con)

        try:
            contracts = fetch_all_contracts(UNDERLYING)
        except Exception as e:
            if ALPACA_FAIL_OPEN:
                print(f"[alpaca] WARNING: contracts OI ingest failed: {e}")
                return
            raise

        if not contracts:
            print("[alpaca] contracts returned 0 rows")
            return

        # Normalize into strike-buckets with call/put OI
        rows = []
        for c in contracts:
            exp = c.get("expiration_date") or c.get("expiration") or c.get("expiry")
            strike = c.get("strike_price") or c.get("strike")
            opt_type = (c.get("type") or c.get("option_type") or "").lower()
            oi = c.get("open_interest")

            if exp is None or strike is None or opt_type not in ("call", "put") or oi is None:
                continue

            rows.append((exp, float(strike), opt_type, int(oi)))

        df = pd.DataFrame(rows, columns=["expiration_date", "strike", "type", "oi"])
        if df.empty:
            print("[alpaca] contracts parsed, but no rows with open_interest")
            return

        pivot = (
            df.pivot_table(index=["expiration_date", "strike"], columns="type", values="oi", aggfunc="sum", fill_value=0)
              .reset_index()
        )
        if "call" not in pivot.columns:
            pivot["call"] = 0
        if "put" not in pivot.columns:
            pivot["put"] = 0

        n = 0
        for _, r in pivot.iterrows():
            upsert_row(con, (
                snap_ts, session, UNDERLYING,
                r["expiration_date"], float(r["strike"]),
                int(r["call"]), int(r["put"])
            ))
            n += 1

        con.commit()
        print(f"[alpaca] wrote options_open_interest_daily rows={n}")

    finally:
        con.close()

if __name__ == "__main__":
    main()
