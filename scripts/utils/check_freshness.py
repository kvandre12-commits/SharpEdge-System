#!/usr/bin/env python3
import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

NY = ZoneInfo("America/New_York")
MAX_OPTIONS_AGE_MIN = int(os.getenv("MAX_OPTIONS_AGE_MIN", "180"))
MAX_INTRADAY_AGE_MIN = int(os.getenv("MAX_INTRADAY_AGE_MIN", "90"))
FAIL_CLOSED = os.getenv("FRESHNESS_FAIL_CLOSED", "0") == "1"


def fail(msg: str):
    print(f"FRESHNESS FAILED: {msg}")
    if FAIL_CLOSED:
        sys.exit(1)


def warn(msg: str):
    print(f"FRESHNESS WARNING: {msg}")


def ok(msg: str):
    print(f"OK: {msg}")


def table_exists(cur, table: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def parse_ts(ts: str):
    if not ts:
        return None

    ts = str(ts).replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def minutes_old(ts: datetime) -> float:
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 60.0


def main():
    if not os.path.exists(DB_PATH):
        fail(f"DB missing: {DB_PATH}")
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print("=" * 72)
    print("SHARP EDGE FRESHNESS AUDIT")
    print("=" * 72)

    # --------------------------------------------------
    # OPTIONS SNAPSHOTS
    # --------------------------------------------------
    if table_exists(cur, "options_chain_snapshots"):
        row = cur.execute(
            """
            SELECT MAX(snapshot_ts)
            FROM options_chain_snapshots
            WHERE underlying = ?
            """,
            (SYMBOL,),
        ).fetchone()

        latest = row[0] if row else None

        if latest:
            dt = parse_ts(latest)
            age = minutes_old(dt)

            print(f"latest_options_snapshot_ts: {latest}")
            print(f"options_snapshot_age_min : {age:.1f}")

            if age > MAX_OPTIONS_AGE_MIN:
                fail(f"options snapshots stale ({age:.1f} min old)")
            else:
                ok("options snapshots fresh")
        else:
            fail("options_chain_snapshots empty")
    else:
        fail("missing table: options_chain_snapshots")

    # --------------------------------------------------
    # POSITIONING METRICS
    # --------------------------------------------------
    if table_exists(cur, "options_positioning_metrics"):
        row = cur.execute(
            """
            SELECT MAX(snapshot_ts), MAX(session_date)
            FROM options_positioning_metrics
            WHERE underlying = ?
            """,
            (SYMBOL,),
        ).fetchone()

        snap_ts, session_date = row if row else (None, None)

        if snap_ts:
            dt = parse_ts(snap_ts)
            age = minutes_old(dt)
            ny_today = datetime.now(NY).date().isoformat()

            print(f"latest_positioning_snapshot_ts: {snap_ts}")
            print(f"latest_positioning_session   : {session_date}")
            print(f"positioning_age_min         : {age:.1f}")

            if session_date != ny_today:
                warn(f"positioning session_date ({session_date}) != NY today ({ny_today})")

            if age > MAX_OPTIONS_AGE_MIN:
                fail(f"options positioning stale ({age:.1f} min old)")
            else:
                ok("options positioning fresh")
        else:
            fail("options_positioning_metrics empty")
    else:
        fail("missing table: options_positioning_metrics")

    # --------------------------------------------------
    # INTRADAY BARS
    # --------------------------------------------------
    if table_exists(cur, "spy_bars_15m"):
        row = cur.execute(
            """
            SELECT MAX(ts)
            FROM spy_bars_15m
            WHERE symbol = ?
            """,
            (SYMBOL,),
        ).fetchone()

        latest = row[0] if row else None

        if latest:
            dt = parse_ts(latest)
            age = minutes_old(dt)

            print(f"latest_intraday_bar_ts: {latest}")
            print(f"intraday_bar_age_min : {age:.1f}")

            if age > MAX_INTRADAY_AGE_MIN:
                warn(f"intraday bars look stale ({age:.1f} min old)")
            else:
                ok("intraday bars fresh")
        else:
            fail("spy_bars_15m empty")
    else:
        fail("missing table: spy_bars_15m")

    # --------------------------------------------------
    # EXECUTION STATE
    # --------------------------------------------------
    if table_exists(cur, "execution_state_daily"):
        row = cur.execute(
            """
            SELECT
              session_date,
              final_bias,
              wall_strike,
              dist_to_wall_pct
            FROM execution_state_daily
            WHERE symbol = ?
            ORDER BY session_date DESC
            LIMIT 1
            """,
            (SYMBOL,),
        ).fetchone()

        if row:
            session_date, bias, wall, dist = row

            print(f"latest_execution_session: {session_date}")
            print(f"latest_execution_bias   : {bias}")

            if wall is None:
                warn("execution_state_daily missing wall_strike")

            if dist is None:
                warn("execution_state_daily missing dist_to_wall_pct")

            ok("execution_state_daily present")
        else:
            fail("execution_state_daily empty")
    else:
        fail("missing table: execution_state_daily")

    print("=" * 72)
    print("FRESHNESS AUDIT COMPLETE")
    print("=" * 72)

    con.close()


if __name__ == "__main__":
    main()
