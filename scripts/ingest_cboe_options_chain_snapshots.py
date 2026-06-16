#!/usr/bin/env python3
"""Persist SPY options chain snapshots from CBOE delayed quotes.

This REPLACES the Alpaca ingester as the open-interest source. Alpaca's options
snapshot endpoint does not return open interest at all, which is why
``options_chain_snapshots`` had all-zero OI/volume historically and the
gamma-regime / wall gates were un-backtestable.

CBOE is the SAME feed the live cockpit (``make_cockpit.fetch_options``) already
trusts, so persisting it gives history that matches what live signals were
computed from. One row = one (expiry, strike) with call + put legs merged.

  python3 scripts/ingest_cboe_options_chain_snapshots.py
  python3 scripts/ingest_cboe_options_chain_snapshots.py --verify   # print coverage
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import re
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from options_snapshot_store import (  # noqa: E402
    connect,
    dte,
    ensure_table,
    iso_utc_now,
    ny_session_date,
    upsert_row,
)

UNDERLYING = os.getenv("SYMBOL", "SPY")
CBOE_URL = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{UNDERLYING}.json"
UA = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = int(os.getenv("CBOE_TIMEOUT", "30"))
FAIL_OPEN = os.getenv("CBOE_FAIL_OPEN", "1").strip() == "1"
# OCC: UNDERLYING + YYMMDD + C/P + strike*1000 (8 digits)
SYM_RE = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")


def fetch_chain() -> list[dict]:
    r = requests.get(CBOE_URL, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]["options"]


def _num(v):
    """Coerce to a value or None (so COALESCE preserves real data)."""
    return v if v not in (None, "") else None


def build_rows(options: list[dict], snap_ts: str, session: str):
    """Merge call+put per (expiry, strike) into one persistable row."""
    merged: dict[tuple[str, float], dict] = collections.defaultdict(dict)
    for o in options:
        m = SYM_RE.match(o.get("option", ""))
        if not m:
            continue
        yymmdd, cp, strike8 = m.groups()
        expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date().isoformat()
        strike = int(strike8) / 1000.0
        leg = merged[(expiry, strike)]
        prefix = "call" if cp == "C" else "put"
        leg[f"{prefix}_oi"] = _num(o.get("open_interest"))
        leg[f"{prefix}_volume"] = _num(o.get("volume"))
        leg[f"{prefix}_gamma"] = _num(o.get("gamma"))

    for (expiry, strike), leg in merged.items():
        yield (
            snap_ts,
            session,
            UNDERLYING,
            expiry,
            dte(session, expiry),
            strike,
            leg.get("call_oi"),
            leg.get("put_oi"),
            leg.get("call_volume"),
            leg.get("put_volume"),
            leg.get("call_gamma"),
            leg.get("put_gamma"),
        )


def verify(con) -> None:
    cur = con.cursor()
    tot = cur.execute("SELECT COUNT(*) FROM options_chain_snapshots").fetchone()[0]
    oi = cur.execute(
        "SELECT COUNT(*) FROM options_chain_snapshots WHERE call_oi>0 OR put_oi>0"
    ).fetchone()[0]
    days = cur.execute(
        "SELECT COUNT(DISTINCT session_date) FROM options_chain_snapshots"
    ).fetchone()[0]
    cboe = cur.execute(
        "SELECT COUNT(*) FROM options_chain_snapshots WHERE source='cboe' AND (call_oi>0 OR put_oi>0)"
    ).fetchone()[0]
    print(f"rows={tot} days={days} rows_with_OI>0={oi} cboe_rows_with_OI>0={cboe}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="print coverage and exit")
    args = ap.parse_args()

    con = connect()
    ensure_table(con)
    if args.verify:
        verify(con)
        con.close()
        return 0

    snap_ts = iso_utc_now()
    session = ny_session_date()
    try:
        options = fetch_chain()
    except Exception as e:  # noqa: BLE001 - network is best-effort
        if FAIL_OPEN:
            print(f"[cboe] WARNING: chain fetch failed: {e} - continuing without snapshot.")
            return 0
        raise

    count = 0
    for row in build_rows(options, snap_ts, session):
        upsert_row(con, row, source="cboe")
        count += 1
    con.commit()
    print(f"[cboe] upserted {count} merged option rows for {session} ({UNDERLYING})")
    verify(con)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
