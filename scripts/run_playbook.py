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


def choose_dte_bucket(entry_ts_utc: str, risk_per_share: float) -> tuple[str, str]:
    dt_ny = parse_ts_utc(entry_ts_utc).astimezone(NY)
    minutes = dt_ny.hour * 60 + dt_ny.minute

    late_session = minutes >= (13 * 60 + 30)  # after 1:30pm NY
    tight = risk_per_share <= 0.70
    wide = risk_per_share >= 1.40

    if wide:
        return ("5-7", f"risk wide ({risk_per_share:.2f}) → buy time (less shakeouts)")
    if late_session and tight:
        return ("0-1", f"late session {dt_ny:%H:%M} NY + tight risk {risk_per_share:.2f} → gamma scalp")
    return ("2-3", f"default balanced | time={dt_ny:%H:%M} NY risk={risk_per_share:.2f} (options-friendly)")


def attach_dte_and_plan(con: sqlite3.Connection):
    rows = con.execute("""
      SELECT signal_id, rule_id, entry_ts, entry_price, sweep_low
      FROM trade_signals
      WHERE dte_bucket is_slow = False  # default

# OPTIONAL: infer slow regime if you want later
# e.g. is_slow = (regime == 'slow')

calibrated = get_calibrated_bucket(con, rule_id, is_slow)
if calibrated:
    bucket = calibrated
    reason = f"calibrated ({'slow' if is_slow else 'default'})"
    """).fetchall()

    for signal_id, rule_id, entry_ts, entry_price, sweep_low in rows:
        risk = float(entry_price) - float(sweep_low)
        bucket, reason = choose_dte_bucket(entry_ts, risk)

        plan = con.execute("""
          SELECT entry_timing, strike_bias, stop_logic, profit_plan,
                 expected_behavior, sizing_guidance, notes
          FROM options_playbook
          WHERE rule_id = ? AND dte_bucket = ?
          LIMIT 1
        """, (rule_id, bucket)).fetchone()

        plan_obj = None
        if plan:
            plan_obj = {
                "dte_bucket": bucket,
                "entry_timing": plan[0],
                "strike_bias": plan[1],
                "stop_logic": plan[2],
                "profit_plan": plan[3],
                "expected_behavior": plan[4],
                "sizing_guidance": plan[5],
                "notes": plan[6],
            }

        con.execute("""
          UPDATE trade_signals
          SET dte_bucket = ?, dte_reason = ?, plan_json = ?
          WHERE signal_id = ?
        """, (bucket, reason, json.dumps(plan_obj) if plan_obj else None, signal_id))

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
        generate_signals(con)
        attach_dte_and_plan(con)
        emit_trade_cards(con, limit=5)
    finally:
        con.close()


if __name__ == "__main__":
    main()
