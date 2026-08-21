#!/usr/bin/env python3
"""Build the CANONICAL auction-expectancy event table (stage 1 of 4).

One row per completed SPY session: the day's gap, whether/when it filled, and
the regime / liquidity / options context that existed that day. This is the
historical memory the conditional-expectancy edge math is trained on.

!!! PIPELINE ORDER MATTERS !!!
This script does DROP TABLE + CREATE TABLE AS, which WIPES the enrichment
columns that stages 2-3 add in place:
    1. build_auction_expectancy_events.py   (this file)   -> base rows
    2. measure_gap_excursions.py            -> MAE/MFE/reward-risk columns
    3. classify_fill_paths.py               -> fill_path_type + path labels
    4. build_conditional_expectancy_matrix.py -> the edge matrix
Running THIS script alone silently destroys stages 2-3. Prefer:
    python scripts/build_auction_expectancy_pipeline.py
"""

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

# Identifiers cannot be SQL-parametrized; validate them so f-string
# interpolation of table names can never carry an injection payload.
_SAFE_IDENT = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _safe_identifier(name: str, label: str) -> str:
    if not name or any(ch not in _SAFE_IDENT for ch in name):
        raise ValueError(f"unsafe {label} identifier: {name!r}")
    return name


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
    bars_table = _safe_identifier(BARS, "intraday bars table")
    con = sqlite3.connect(DB_PATH)
    try:
        missing = [
            table_name for table_name in REQ + [bars_table] if not exists(con, table_name)
        ]
        if missing:
            raise RuntimeError(f"Missing tables: {missing}")

        build_ts = datetime.now(timezone.utc).isoformat()

        # intraday_coverage/session_open_ts: symbol-correct intraday coverage
        # and first bar per session. Coverage is explicit so downstream code can
        # distinguish "gap failed to fill" from "we don't have 15m bars." The
        # open timestamp is used to (a) exclude the OPENING bar from the fill
        # scan -- otherwise a small gap whose open bar already straddles
        # prior_close registers a bogus t~0 "instant fill" -- and (b) anchor
        # time_to_fill correctly.
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
          WHERE symbol = ?
        ), intraday_coverage AS (
          SELECT session_date, COUNT(*) AS intraday_bar_count
          FROM {bars_table}
          WHERE symbol = ?
          GROUP BY session_date
        ), session_open_ts AS (
          SELECT session_date, MIN(ts) AS open_ts
          FROM {bars_table}
          WHERE symbol = ?
          GROUP BY session_date
        ), fills AS (
          SELECT d.session_date, MIN(i.ts) AS fill_ts
          FROM daily d
          LEFT JOIN session_open_ts s ON s.session_date = d.session_date
          LEFT JOIN {bars_table} i
            ON i.session_date = d.session_date
           AND i.symbol = d.symbol
           AND i.ts > s.open_ts
           AND i.low <= d.prior_close
           AND i.high >= d.prior_close
          WHERE d.prior_close IS NOT NULL
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
                 WHEN f.fill_ts IS NULL OR s.open_ts IS NULL THEN NULL
                 ELSE (julianday(f.fill_ts)-julianday(s.open_ts))*1440
               END AS time_to_fill_minutes,
               CASE WHEN COALESCE(ic.intraday_bar_count, 0) > 0 THEN 1 ELSE 0 END AS has_intraday_bars,
               COALESCE(ic.intraday_bar_count, 0) AS intraday_bar_count,
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
               ? AS build_ts
        FROM daily d
        LEFT JOIN intraday_coverage ic ON ic.session_date = d.session_date
        LEFT JOIN session_open_ts s ON s.session_date = d.session_date
        LEFT JOIN fills f ON f.session_date = d.session_date
        LEFT JOIN regime_daily r ON r.date=d.session_date AND r.symbol=d.symbol
        LEFT JOIN open_resolution_regime o ON o.session_date=d.session_date AND o.underlying=d.symbol
        LEFT JOIN liquidity_regime_events l ON l.session_date=d.session_date AND l.underlying=d.symbol
        LEFT JOIN (
          -- options_positioning_metrics has MANY rows per session
          -- (PK snapshot_ts, underlying, dte_min, dte_max): multiple intraday
          -- snapshots x DTE buckets. Collapse to ONE deterministic row per
          -- session -- latest snapshot, nearest-DTE bucket -- so the join
          -- cannot fan out the canonical event set.
          SELECT session_date, underlying, spot, gamma_wall_strike, pcr_oi
          FROM (
            SELECT session_date, underlying, spot, gamma_wall_strike, pcr_oi,
                   ROW_NUMBER() OVER (
                     PARTITION BY session_date, underlying
                     ORDER BY snapshot_ts DESC, dte_min ASC, dte_max DESC
                   ) AS _rn
            FROM options_positioning_metrics
          )
          WHERE _rn = 1
        ) op ON op.session_date=d.session_date AND op.underlying=d.symbol
        WHERE d.prior_close IS NOT NULL
        """

        con.execute(f"DROP TABLE IF EXISTS {TABLE}")
        con.execute(
            f"CREATE TABLE {TABLE} AS {query}",
            (SYMBOL, SYMBOL, SYMBOL, build_ts),
        )
        con.commit()

        # Fan-out guard: the context LEFT JOINs assume <=1 row per session. If
        # open_resolution_regime / liquidity_regime_events / options_positioning
        # ever holds >1 row per (session_date, symbol), rows multiply and the
        # downstream expectancy counts silently inflate. Fail loudly instead.
        n_rows = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        n_sessions = con.execute(
            f"SELECT COUNT(DISTINCT session_date) FROM {TABLE}"
        ).fetchone()[0]
        if n_rows != n_sessions:
            raise RuntimeError(
                f"Fan-out detected: {n_rows} rows across {n_sessions} sessions. "
                "A context table (open_resolution_regime / liquidity_regime_events / "
                "options_positioning_metrics) has >1 row per session_date. Dedup it "
                "before trusting expectancy stats."
            )

        row_count = export_csv(con)
        print(f"OK: wrote {TABLE} rows={row_count} (sessions={n_sessions})")
        print(
            "NOTE: stage 1 of 4 complete. Now run measure_gap_excursions.py -> "
            "classify_fill_paths.py -> build_conditional_expectancy_matrix.py "
            "(or use build_auction_expectancy_pipeline.py)."
        )
    finally:
        con.close()


if __name__ == "__main__":
    main()
