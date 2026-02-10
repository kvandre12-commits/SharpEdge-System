#!/usr/bin/env python3
import os
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time
import random
import socket
import gzip
import urllib.error

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")
FEED = os.getenv("ALPACA_DATA_FEED", "").strip()  # optional: "sip" etc.

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()
ALPACA_TIMEOUT = int(os.getenv("ALPACA_TIMEOUT", "120"))
ALPACA_RETRIES = int(os.getenv("ALPACA_RETRIES", "6"))
ALPACA_FAIL_OPEN = os.getenv("ALPACA_FAIL_OPEN", "1").strip() == "1"  # 1 = don't fail workflow

NY = ZoneInfo("America/New_York")

def iso_utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def ny_session_date(dt_utc: datetime) -> str:
    return dt_utc.astimezone(NY).date().isoformat()

def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def ensure_table(con):
    # In case migrations haven't run yet (safe)
    con.execute("""
    CREATE TABLE IF NOT EXISTS options_chain_snapshots (
      snapshot_ts   TEXT NOT NULL,
      session_date  TEXT NOT NULL,
      underlying    TEXT NOT NULL,
      expiry_date   TEXT NOT NULL,
      dte           INTEGER NOT NULL,
      strike        REAL NOT NULL,
      call_oi       INTEGER,
      put_oi        INTEGER,
      call_volume   INTEGER,
      put_volume    INTEGER,
      call_gamma    REAL,
      put_gamma     REAL,
      source        TEXT DEFAULT 'alpaca',
      PRIMARY KEY (snapshot_ts, underlying, expiry_date, strike)
    );
    """)
    con.commit()

def alpaca_get_chain_snapshots(underlying: str) -> dict:
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}"
    if FEED:
        url += f"?feed={FEED}"

    req = urllib.request.Request(url)
    req.add_header("APCA-API-KEY-ID", ALPACA_API_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_API_SECRET)
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Encoding", "gzip")  # helps a lot

    last_err = None
    for attempt in range(ALPACA_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=ALPACA_TIMEOUT) as resp:
                raw = resp.read()
                enc = (resp.headers.get("Content-Encoding") or "").lower()
                if "gzip" in enc:
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))

        except (TimeoutError, socket.timeout) as e:
            last_err = e
        except urllib.error.HTTPError as e:
            # Retry only transient statuses
            if e.code in (429, 500, 502, 503, 504):
                last_err = e
            else:
                raise
        except urllib.error.URLError as e:
            last_err = e

        # exponential backoff + jitter
        sleep_s = min(2 ** attempt, 30) + random.random()
        print(f"[alpaca] fetch failed attempt {attempt+1}/{ALPACA_RETRIES}: {last_err} — sleeping {sleep_s:.1f}s")
        time.sleep(sleep_s)
        raise RuntimeError(f"Alpaca snapshot fetch failed after {ALPACA_RETRIES} retries: {last_err}")
        
def parse_contract_symbol(sym: str):
    """
    Alpaca uses OCC-style option symbols (e.g. SPY240119C00450000).
    We'll infer expiry/type/strike from the symbol string.
    """
    # Very light OCC parse:
    # UNDERLYING (1-6) + YYMMDD + C/P + strike*1000 padded
    # This is common across brokers; still treat as best-effort.
    # If symbol doesn't match, skip.
    try:
        # find the date block by scanning digits
        # simplest: last 15 chars = YYMMDD + C/P + 8 digit strike
        tail = sym[-15:]
        yymmdd = tail[:6]
        cp = tail[6]
        strike_int = int(tail[7:])  # strike * 1000
        strike = strike_int / 1000.0

        yy = int(yymmdd[:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        year = 2000 + yy
        expiry = datetime(year, mm, dd).date().isoformat()

        opt_type = "call" if cp.upper() == "C" else "put"
        return expiry, strike, opt_type
    except Exception:
        return None

def dte(session_date: str, expiry_date: str) -> int:
    sd = datetime.fromisoformat(session_date).date()
    ed = datetime.fromisoformat(expiry_date).date()
    return (ed - sd).days

def upsert_row(con, row):
    con.execute("""
      INSERT INTO options_chain_snapshots (
        snapshot_ts, session_date, underlying,
        expiry_date, dte, strike,
        call_oi, put_oi, call_volume, put_volume,
        call_gamma, put_gamma, source
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'alpaca')
      ON CONFLICT(snapshot_ts, underlying, expiry_date, strike) DO UPDATE SET
        session_date = excluded.session_date,
        dte          = excluded.dte,
        call_oi      = COALESCE(excluded.call_oi, options_chain_snapshots.call_oi),
        put_oi       = COALESCE(excluded.put_oi, options_chain_snapshots.put_oi),
        call_volume  = COALESCE(excluded.call_volume, options_chain_snapshots.call_volume),
        put_volume   = COALESCE(excluded.put_volume, options_chain_snapshots.put_volume),
        call_gamma   = COALESCE(excluded.call_gamma, options_chain_snapshots.call_gamma),
        put_gamma    = COALESCE(excluded.put_gamma, options_chain_snapshots.put_gamma),
        source       = 'alpaca'
    """, row)

def main():
    snap_ts = iso_utc_now()
    now_utc = datetime.now(timezone.utc)
    session = ny_session_date(now_utc)

    con = connect()
    ensure_table(con)

    try:
        payload = alpaca_get_chain_snapshots(UNDERLYING)
        print("[DEBUG] payload keys:", list(payload.keys()))
        print("[DEBUG] snapshots type:", type(payload.get("snapshots")))
        print("[DEBUG] snapshots sample:", str(payload.get("snapshots"))[:500])
    except Exception as e:
        if ALPACA_FAIL_OPEN:
            print(f"[alpaca] WARNING: options snapshot ingest failed: {e}")
            print("[alpaca] Continuing workflow without fresh options snapshots.")
            return
        raise
    # -------- parse contracts and insert --------
    count = 0

    # Alpaca snapshot payload shape:
    # { "snapshots": { "OCC_SYMBOL": { ... } } }
    snaps = payload.get("snapshots", {})

    for sym, snap in snaps.items():
        parsed = parse_contract_symbol(sym)
        if not parsed:
            continue

        expiry, strike, opt_type = parsed
        dte_val = dte(session, expiry)

        # pull metrics safely
        oi = (snap.get("open_interest") or 0)
        vol = (snap.get("latest_trade", {}) or {}).get("size") or 0
        gamma = (snap.get("greeks", {}) or {}).get("gamma")

        # build row with call/put separation
        call_oi = oi if opt_type == "call" else None
        put_oi = oi if opt_type == "put" else None
        call_vol = vol if opt_type == "call" else None
        put_vol = vol if opt_type == "put" else None
        call_gamma = gamma if opt_type == "call" else None
        put_gamma = gamma if opt_type == "put" else None

        row = (
            snap_ts,
            session,
            UNDERLYING,
            expiry,
            dte_val,
            strike,
            call_oi,
            put_oi,
            call_vol,
            put_vol,
            call_gamma,
            put_gamma,
        )

        upsert_row(con, row)
        count += 1

    con.commit()
    con.close()

    print(f"[alpaca] inserted {count} option snapshot rows")
if __name__ == "__main__":
    main()
