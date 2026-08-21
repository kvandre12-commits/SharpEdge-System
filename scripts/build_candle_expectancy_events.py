#!/usr/bin/env python3
"""Build raw candle-conditioned expectancy events.

One row per non-ordinary intraday candle event. Event/context columns are causal
(known at the event bar). Forward bars are used only for outcome labels.

Usage:
    cd ~/SharpEdge-System
    python scripts/build_candle_expectancy_events.py

Environment:
    SPY_DB_PATH=data/spy_truth.db
    SYMBOL=SPY
    INTRADAY_BARS_TABLE=spy_bars_15m
    CANDLE_HORIZON_BARS=4
    CANDLE_TARGET_PCT=0.0010
    CANDLE_STOP_PCT=0.0008
    CANDLE_INCLUDE_ORDINARY=0
"""

from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from candle_expectancy_core import build_event_rows_for_session

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
TABLE = os.getenv("CANDLE_EXPECTANCY_TABLE", "candle_expectancy_events")
OUT = Path(os.getenv("CANDLE_EXPECTANCY_OUT", "outputs/candle_expectancy_events.csv"))
HORIZON_BARS = int(os.getenv("CANDLE_HORIZON_BARS", "4"))
TARGET_PCT = float(os.getenv("CANDLE_TARGET_PCT", "0.0010"))
STOP_PCT = float(os.getenv("CANDLE_STOP_PCT", "0.0008"))
INCLUDE_ORDINARY = os.getenv("CANDLE_INCLUDE_ORDINARY", "0") == "1"

_SAFE_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

COLUMNS = [
    ("symbol", "TEXT"),
    ("session_date", "TEXT"),
    ("ts", "TEXT"),
    ("bar_index", "INTEGER"),
    ("minutes_since_open", "INTEGER"),
    ("event_name", "TEXT"),
    ("event_direction", "TEXT"),
    ("open", "REAL"),
    ("high", "REAL"),
    ("low", "REAL"),
    ("close", "REAL"),
    ("volume", "REAL"),
    ("trade_count", "INTEGER"),
    ("vwap", "REAL"),
    ("range_pct", "REAL"),
    ("body_pct", "REAL"),
    ("upper_wick_pct", "REAL"),
    ("lower_wick_pct", "REAL"),
    ("nearest_reference_name", "TEXT"),
    ("nearest_reference_price", "REAL"),
    ("nearest_reference_distance_pct", "REAL"),
    ("nearest_reference_relation", "TEXT"),
    ("acceptance_state", "TEXT"),
    ("volume_confirmation", "TEXT"),
    ("relative_volume", "REAL"),
    ("vol_state", "TEXT"),
    ("macro_state", "TEXT"),
    ("dp_state", "TEXT"),
    ("regime_label", "TEXT"),
    ("open_regime_label", "TEXT"),
    ("setup_dir", "TEXT"),
    ("gamma_wall_strike", "REAL"),
    ("pcr_oi", "REAL"),
    ("horizon_bars", "INTEGER"),
    ("target_pct", "REAL"),
    ("stop_pct", "REAL"),
    ("target_before_stop_label", "TEXT"),
    ("bars_to_resolution", "INTEGER"),
    ("realized_R", "REAL"),
    ("two_sided_first_touch", "TEXT"),
    ("bars_to_two_sided_touch", "INTEGER"),
    ("favorable_excursion_pct", "REAL"),
    ("adverse_excursion_pct", "REAL"),
    ("forward_bar_count", "INTEGER"),
    ("build_ts", "TEXT"),
]


def safe_identifier(name: str, label: str) -> str:
    if not name or any(ch not in _SAFE_IDENT for ch in name):
        raise ValueError(f"unsafe {label} identifier: {name!r}")
    return name


def exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def rows_for_query(
    con: sqlite3.Connection, query: str, params: tuple[Any, ...]
) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    return [dict(row) for row in con.execute(query, params).fetchall()]


def one_by_session(
    con: sqlite3.Connection,
    table: str,
    date_col: str,
    symbol_col: str,
    symbol: str,
) -> dict[str, dict[str, Any]]:
    if not exists(con, table):
        return {}
    rows = rows_for_query(
        con,
        f"SELECT * FROM {table} WHERE {symbol_col} = ? ORDER BY {date_col} ASC",
        (symbol,),
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row.get(date_col))] = row
    return out


def daily_context(con: sqlite3.Connection, symbol: str) -> dict[str, dict[str, Any]]:
    if not exists(con, "bars_daily"):
        return {}
    rows = rows_for_query(
        con,
        """
        SELECT date, symbol, open, high, low, close,
               LAG(high) OVER(PARTITION BY symbol ORDER BY date) AS prior_high,
               LAG(low) OVER(PARTITION BY symbol ORDER BY date) AS prior_low,
               LAG(close) OVER(PARTITION BY symbol ORDER BY date) AS prior_close
        FROM bars_daily
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        (symbol,),
    )
    return {str(row["date"]): row for row in rows}


def options_context(con: sqlite3.Connection, symbol: str) -> dict[str, dict[str, Any]]:
    if not exists(con, "options_positioning_metrics"):
        return {}
    rows = rows_for_query(
        con,
        """
        SELECT * FROM (
          SELECT session_date, underlying, spot, gamma_wall_strike, pcr_oi,
                 ROW_NUMBER() OVER (
                   PARTITION BY session_date, underlying
                   ORDER BY snapshot_ts DESC, dte_min ASC, dte_max DESC
                 ) AS rn
          FROM options_positioning_metrics
          WHERE underlying = ?
        )
        WHERE rn = 1
        """,
        (symbol,),
    )
    return {str(row["session_date"]): row for row in rows}


def load_session_bars(
    con: sqlite3.Connection, bars_table: str, symbol: str
) -> dict[str, list[dict[str, Any]]]:
    rows = rows_for_query(
        con,
        f"""
        SELECT ts, session_date, symbol, open, high, low, close, volume, trade_count, vwap
        FROM {bars_table}
        WHERE symbol = ?
        ORDER BY session_date ASC, ts ASC
        """,
        (symbol,),
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_date"]), []).append(row)
    return grouped


def create_table(con: sqlite3.Connection, table: str) -> None:
    ddl_cols = ",\n      ".join(f"{name} {typ}" for name, typ in COLUMNS)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} (\n      {ddl_cols}\n    )")
    con.commit()


def insert_rows(
    con: sqlite3.Connection, table: str, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    names = [name for name, _ in COLUMNS]
    placeholders = ",".join("?" for _ in names)
    con.executemany(
        f"INSERT INTO {table} ({','.join(names)}) VALUES ({placeholders})",
        [[row.get(name) for name in names] for row in rows],
    )
    con.commit()


def export_csv(rows: list[dict[str, Any]]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in COLUMNS]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in names} for row in rows)


def main() -> int:
    bars_table = safe_identifier(BARS_TABLE, "intraday bars table")
    out_table = safe_identifier(TABLE, "output table")
    con = sqlite3.connect(DB_PATH)
    try:
        if not exists(con, bars_table):
            raise RuntimeError(f"Missing intraday table: {bars_table}")
        sessions = load_session_bars(con, bars_table, SYMBOL)
        if not sessions:
            raise RuntimeError(f"{bars_table} returned 0 rows for {SYMBOL}")

        daily = daily_context(con, SYMBOL)
        regimes = one_by_session(con, "regime_daily", "date", "symbol", SYMBOL)
        opens = one_by_session(
            con, "open_resolution_regime", "session_date", "underlying", SYMBOL
        )
        options = options_context(con, SYMBOL)
        build_ts = datetime.now(timezone.utc).isoformat()

        event_rows: list[dict[str, Any]] = []
        for session_date, bars in sessions.items():
            if len(bars) < 2:
                continue
            for row in build_event_rows_for_session(
                symbol=SYMBOL,
                session_date=session_date,
                bars=bars,
                daily=daily.get(session_date),
                regime=regimes.get(session_date),
                open_regime=opens.get(session_date),
                options=options.get(session_date),
                horizon_bars=HORIZON_BARS,
                target_pct=TARGET_PCT,
                stop_pct=STOP_PCT,
                include_ordinary=INCLUDE_ORDINARY,
            ):
                row["build_ts"] = build_ts
                event_rows.append(row)

        create_table(con, out_table)
        insert_rows(con, out_table, event_rows)
        export_csv(event_rows)
        print(
            f"OK: wrote {out_table} rows={len(event_rows)} sessions={len(sessions)} "
            f"horizon_bars={HORIZON_BARS} target={TARGET_PCT:.4f} stop={STOP_PCT:.4f}"
        )
        print(f"OK: wrote {OUT}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
