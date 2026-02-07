#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
MIGRATIONS_DIR = Path("sql/migrations")
NY = ZoneInfo("America/New_York")

def get_latest_trade_gate(con: sqlite3.Connection, symbol: str = "SPY"):
    """
    Returns (trade_gate:int, pressure_state:str, pressure_reason:str)
    from the most recent signals_daily row.
    """
    row = con.execute("""
        SELECT trade_gate, pressure_state, pressure_reason
        FROM signals_daily
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 1
    """, (symbol,)).fetchone()

    if not row:
        # Fail closed (institutional default)
        return 0, "UNKNOWN", "no signals_daily row found"

    trade_gate, pressure_state, pressure_reason = row
    return int(trade_gate or 0), pressure_state or "", pressure_reason or ""

def ensure_migrations_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id TEXT PRIMARY KEY,
      applied_ts TEXT DEFAULT (datetime('now'))
    )
    """)
    con.commit()


def apply_migrations(con: sqlite3.Connection):
    ensure_migrations_table(con)
    for p in sorted(MIGRATIONS_DIR.glob("*.sql")):
        mid = p.name
        already = con.execute("SELECT 1 FROM schema_migrations WHERE id=?", (mid,)).fetchone()
        if already:
            continue
        sql = p.read_text(encoding="utf-8")
        con.executescript(sql)
        con.execute("INSERT INTO schema_migrations(id) VALUES (?)", (mid,))
        con.commit()
        print(f"applied migration: {mid}")

def is_slow_from_spy_regime(vol_state, vol_trend_state, regime_label) -> bool:
    for v in (vol_state, vol_trend_state, regime_label):
        if isinstance(v, str) and "slow" in v.lower():
            return True
    return False

def infer_is_slow(reg_row: dict) -> bool:
    """
    Robust 'slow' detector across different regime schemas.
    We don't assume a single column name.
    """
    if not reg_row:
        return False

    # common boolean flags
    for k in ("is_slow", "slow_flag", "slow_regime_flag"):
        v = reg_row.get(k)
        if v in (1, True, "1", "true", "TRUE", "True"):
            return True

    # common string labels
    for k in ("vol_state", "speed_state", "regime", "regime_label", "state"):
        v = reg_row.get(k)
        if isinstance(v, str) and "slow" in v.lower():
            return True

    return False


def generate_signals(con: sqlite3.Connection):
    # Insert new signals from the view into trade_signals (dedupe by entry_ts+rule_id)
    con.execute("""
    INSERT INTO trade_signals (rule_id, symbol, sweep_ts, entry_ts, entry_price, prior_support, sweep_low)
    SELECT v.rule_id, v.symbol, v.sweep_ts, v.entry_ts, v.entry_price, v.prior_support, v.sweep_low
    FROM v_failed_breakdown_long_opt v
    WHERE NOT EXISTS (
      SELECT 1 FROM trade_signals t
      WHERE t.rule_id = v.rule_id AND t.entry_ts = v.entry_ts
    )
    """)
    con.commit()


def parse_ts_utc(ts: str) -> datetime:
    # stored like 2026-02-03T19:45:00Z from Alpaca ingest
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def get_calibrated_bucket(con: sqlite3.Connection, rule_id: str, is_slow: bool):
    """
    Returns (bucket, reason). Always returns a safe fallback if no calibration.
    """
    cur = con.cursor()
    row = cur.execute("""
        SELECT default_bucket, slow_bucket
        FROM dte_calibration
        WHERE rule_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
    """, (rule_id,)).fetchone()

    if row:
        default_bucket, slow_bucket = row
        chosen = slow_bucket if (is_slow and slow_bucket) else default_bucket
        reason = f"calibrated:{'slow' if is_slow else 'default'}:{chosen}"
        return chosen, reason

    return "2-3", "fallback:2-3 (no calibration)"


def attach_dte_and_plan(con: sqlite3.Connection):
    """
    Attaches DTE bucket + reason to trade_signals using spy_regime_daily context.
    Closes the loop: (spy_regime_daily -> is_slow) + (dte_calibration -> bucket).
    """

    cur = con.cursor()

    # Pull signals missing DTE; derive a date if session_date isn't populated.
    rows = cur.execute("""
        SELECT
          s.signal_id,
          COALESCE(s.session_date, date(s.entry_ts)) AS session_date,
          s.rule_id,
          r.vol_state,
          r.vol_trend_state,
          r.regime_label
        FROM trade_signals s
        LEFT JOIN spy_regime_daily r
          ON r.date = COALESCE(s.session_date, date(s.entry_ts))
        WHERE COALESCE(s.dte_bucket, '') = ''
    """).fetchall()

    for signal_id, session_date, rule_id, vol_state, vol_trend_state, regime_label in rows:
        slow = is_slow_from_spy_regime(vol_state, vol_trend_state, regime_label)
        bucket, reason = get_calibrated_bucket(con, rule_id, slow)

        cur.execute("""
            UPDATE trade_signals
            SET dte_bucket = ?, dte_reason = ?
            WHERE signal_id = ?
        """, (bucket, reason, signal_id))

    con.commit()
def emit_trade_cards(con: sqlite3.Connection, limit: int = 5):
    cards = con.execute("""
      SELECT
        signal_id, rule_id, symbol,
        sweep_ts, entry_ts, entry_price, prior_support, sweep_low,
        dte_bucket, dte_reason,
        entry_timing, strike_bias, stop_logic, profit_plan
      FROM v_trade_cards
      ORDER BY signal_id DESC
      LIMIT ?
    """, (limit,)).fetchall()

    for c in cards:
        risk = float(c[5]) - float(c[7])
        print("\n--- TRADE CARD ---")
        print(f"signal_id: {c[0]} | rule: {c[1]} | sym: {c[2]}")
        print(f"sweep_ts:  {c[3]}")
        print(f"entry_ts:  {c[4]} | entry: {float(c[5]):.2f}")
        print(f"support:   {float(c[6]):.2f} | sweep_low: {float(c[7]):.2f} | risk: {risk:.2f}")
        print(f"DTE AUTO:  {c[8]}  ({c[9]})")
        print(f"Entry:     {c[10]}")
        print(f"Strikes:   {c[11]}")
        print(f"Stop:      {c[12]}")
        print(f"Profits:   {c[13]}")


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        apply_migrations(con)

        # -----------------------------
        # INSTITUTIONAL HARD GATE
        # -----------------------------
        trade_gate, pressure_state, pressure_reason = get_latest_trade_gate(con)

        if trade_gate != 1:
            print("\n=== NO TRADE STATE ===")
            print(f"trade_gate:     {trade_gate}")
            print(f"pressure_state:{pressure_state}")
            print(f"reason:         {pressure_reason}")
            print("Action:         Stand down. No signals generated.")
            return

        # -----------------------------
        # Only proceed if gate is open
        # -----------------------------
        generate_signals(con)
        attach_dte_and_plan(con)
        emit_trade_cards(con, limit=5)

    finally:
        con.close()


if __name__ == "__main__":
    main()
