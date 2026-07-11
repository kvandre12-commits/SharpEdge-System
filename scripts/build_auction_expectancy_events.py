#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
TABLE = "auction_expectancy_events"
OUT = Path("outputs/auction_expectancy_events.csv")
REQ = [
    "bars_daily",
    "regime_daily",
    "open_resolution_regime",
    "liquidity_regime_events",
    "overlays_daily",
    "options_positioning_metrics",
]


def exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def export_csv(con: sqlite3.Connection) -> int:
    cur = con.execute(f"SELECT * FROM {TABLE} ORDER BY session_date")
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        missing = [
            table_name for table_name in REQ + [BARS] if not exists(con, table_name)
        ]
        if missing:
            raise RuntimeError(f"Missing tables: {missing}")

        build_ts = datetime.now(timezone.utc).isoformat()
        query = f"""
        WITH daily AS (
          SELECT date AS session_date,
                 symbol,
                 open AS session_open,
                 high AS session_high,
                 low AS session_low,
                 close AS session_close,
                 LAG(close) OVER(PARTITION BY symbol ORDER BY date) AS prior_close
          FROM bars_daily
          WHERE symbol='{SYMBOL}'
        ), fills AS (
          SELECT d.session_date, MIN(i.ts) AS fill_ts
          FROM daily d
          LEFT JOIN {BARS} i ON i.session_date=d.session_date AND i.symbol=d.symbol
            AND i.low<=d.prior_close AND i.high>=d.prior_close
          GROUP BY d.session_date
        )
        SELECT d.symbol,
               d.session_date,
               COALESCE(l.regime_type,'DAILY_AUCTION') AS event_type,
               ((d.session_open-d.prior_close)/NULLIF(d.prior_close,0)) AS gap_pct,
               CASE
                 WHEN d.session_open>d.prior_close THEN 'UP'
                 WHEN d.session_open<d.prior_close THEN 'DOWN'
                 ELSE 'FLAT'
               END AS gap_direction,
               d.prior_close AS gap_fill_level,
               CASE WHEN f.fill_ts IS NULL THEN 0 ELSE 1 END AS fill_completed,
               f.fill_ts,
               CASE
                 WHEN f.fill_ts IS NULL THEN NULL
                 ELSE (julianday(f.fill_ts)-julianday((SELECT MIN(ts) FROM {BARS} b WHERE b.session_date=d.session_date)))*1440
               END AS time_to_fill_minutes,
               d.prior_close,
               d.session_open,
               d.session_high,
               d.session_low,
               d.session_close,
               r.vol_state,
               r.vol_trend_state,
               r.dp_state,
               r.macro_state,
               r.regime_label,
               o.open_regime_label,
               COALESCE(o.failed_breakdown_open,0) AS failed_breakdown_open,
               COALESCE(o.accepted_breakdown_open,0) AS accepted_breakdown_open,
               o.regime_confidence,
               o.setup_dir,
               o.key_source,
               l.regime_type AS liquidity_regime_type,
               op.spot,
               op.gamma_wall_strike,
               op.pcr_oi,
               '{build_ts}' AS build_ts
        FROM daily d
        LEFT JOIN fills f ON f.session_date=d.session_date
        LEFT JOIN regime_daily r ON r.date=d.session_date AND r.symbol=d.symbol
        LEFT JOIN open_resolution_regime o ON o.session_date=d.session_date AND o.underlying=d.symbol
        LEFT JOIN liquidity_regime_events l ON l.session_date=d.session_date AND l.underlying=d.symbol
        LEFT JOIN options_positioning_metrics op ON op.session_date=d.session_date AND op.underlying=d.symbol
        """

        con.execute(f"DROP TABLE IF EXISTS {TABLE}")
        con.execute(f"CREATE TABLE {TABLE} AS {query}")
        con.commit()
        row_count = export_csv(con)
        print(f"OK: wrote {TABLE} rows={row_count}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
