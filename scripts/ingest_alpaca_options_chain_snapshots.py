#!/usr/bin/env python3
import os
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")
FEED = os.getenv("ALPACA_DATA_FEED", "").strip()  # optional: "sip" etc.

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

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

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

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
    try:
        ensure_table(con)

        payload = alpaca_get_chain_snapshots(UNDERLYING)

        # Docs describe "snapshots" response; fields can vary a bit.
        # We'll look for a dict of contract_symbol -> snapshot blob.
        snaps = payload.get("snapshots") or payload.get("option_snapshots") or payload
        if not isinstance(snaps, dict):
            raise RuntimeError("Unexpected Alpaca response shape for option chain snapshots")

        # We'll aggregate to strike-level rows with call/put columns on same (expiry,strike).
        # Start with empty containers.
        agg = {}  # (expiry, strike) -> dict

        for contract_sym, blob in snaps.items():
            parsed = parse_contract_symbol(contract_sym)
            if not parsed:
                continue
            expiry, strike, opt_type = parsed
            key = (expiry, strike)

            # Greeks (if present)
            greeks = (blob.get("greeks") or {})
            gamma = greeks.get("gamma")

            # Volume/OI: may or may not exist in snapshot payload; keep nullable.
            # If your payload includes daily volume, you can map it here.
            vol = None
            oi = None

            if key not in agg:
                agg[key] = {
                    "call_gamma": None, "put_gamma": None,
                    "call_volume": None, "put_volume": None,
                    "call_oi": None, "put_oi": None,
                }

            if opt_type == "call":
                agg[key]["call_gamma"] = gamma
                agg[key]["call_volume"] = vol
                agg[key]["call_oi"] = oi
            else:
                agg[key]["put_gamma"] = gamma
                agg[key]["put_volume"] = vol
                agg[key]["put_oi"] = oi

        n = 0
        for (expiry, strike), vals in agg.items():
            row = (
                snap_ts, session, UNDERLYING,
                expiry, dte(session, expiry), float(strike),
                vals["call_oi"], vals["put_oi"], vals["call_volume"], vals["put_volume"],
                vals["call_gamma"], vals["put_gamma"],
            )
            upsert_row(con, row)
            n += 1

        con.commit()
        print(f"OK: ingested option chain snapshots underlying={UNDERLYING} rows={n} snapshot_ts={snap_ts}")

    finally:
        con.close()

if __name__ == "__main__":
    main()
