#!/usr/bin/env python3
"""Unify every overlay into one per-trading-day context layer."""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))

MACRO_TYPES = {
    "vix": "ovl_vix",
    "vix3m": "ovl_vix3m",
    "vix_term": "ovl_vix_term",
    "rates10y": "ovl_rates10y",
}
EVENT_TYPES = {"darkpool": "ovl_darkpool", "tariff": "ovl_tariff"}
WEEKLY_TYPES = {
    "shares_z_26w": "dp_shares_z_26w",
    "trades_vs_13w_avg": "dp_trades_vs_13w",
    "shares_vs_13w_avg": "dp_shares_vs_13w",
    "avg_trade_size": "dp_avg_trade_size",
}


def _trading_days(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT date FROM bars_daily WHERE symbol = ? ORDER BY date",
        (SYMBOL,),
    ).fetchall()
    return [str(row[0])[:10] for row in rows if row[0]]


def _pivot_overlays(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        "SELECT date, overlay_type, overlay_strength FROM overlays_daily "
        "WHERE symbol = ? ORDER BY date, rowid",
        (SYMBOL,),
    ).fetchall()
    rename = {**MACRO_TYPES, **EVENT_TYPES}
    by_date: dict[str, dict[str, float]] = {}
    for date_value, overlay_type, overlay_strength in rows:
        column = rename.get(str(overlay_type))
        if column is None:
            continue
        date_text = str(date_value)[:10]
        by_date.setdefault(date_text, {})[column] = float(overlay_strength)
    return by_date


def _weekly_darkpool(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT week_start, shares_z_26w, trades_vs_13w_avg, shares_vs_13w_avg, "
        "avg_trade_size FROM ats_weekly WHERE symbol = ? ORDER BY week_start",
        (SYMBOL,),
    ).fetchall()
    weekly = []
    for row in rows:
        week_start = str(row[0])[:10]
        weekly.append(
            {
                "week_start": week_start,
                "dp_shares_z_26w": _maybe_float(row[1]),
                "dp_trades_vs_13w": _maybe_float(row[2]),
                "dp_shares_vs_13w": _maybe_float(row[3]),
                "dp_avg_trade_size": _maybe_float(row[4]),
            }
        )
    return weekly


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_context_row(date_text: str) -> dict[str, Any]:
    row = {"date": date_text, "symbol": SYMBOL}
    for column in MACRO_TYPES.values():
        row[column] = None
    for column in EVENT_TYPES.values():
        row[column] = None
    row["ovl_vix_contango"] = None
    for column in WEEKLY_TYPES.values():
        row[column] = None
    return row


def build(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    trading_days = _trading_days(conn)
    overlay_by_date = _pivot_overlays(conn)
    weekly = _weekly_darkpool(conn)

    latest_macro = {column: None for column in MACRO_TYPES.values()}
    weekly_index = -1
    current_weekly: dict[str, Any] | None = None
    context_rows: list[dict[str, Any]] = []

    for date_text in trading_days:
        row = _empty_context_row(date_text)
        daily_overlay = overlay_by_date.get(date_text, {})

        for column in EVENT_TYPES.values():
            if column in daily_overlay:
                row[column] = daily_overlay[column]

        for column in MACRO_TYPES.values():
            if column in daily_overlay:
                latest_macro[column] = daily_overlay[column]
            row[column] = latest_macro[column]

        if row["ovl_vix3m"] is not None and row["ovl_vix"] is not None:
            row["ovl_vix_contango"] = row["ovl_vix3m"] - row["ovl_vix"]

        while (
            weekly_index + 1 < len(weekly)
            and weekly[weekly_index + 1]["week_start"] <= date_text
        ):
            weekly_index += 1
            current_weekly = weekly[weekly_index]

        if current_weekly is not None:
            for column in WEEKLY_TYPES.values():
                row[column] = current_weekly.get(column)

        context_rows.append(row)

    return context_rows


def persist(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    conn.execute("DROP TABLE IF EXISTS overlay_context_daily")
    conn.execute(
        """
        CREATE TABLE overlay_context_daily (
            date TEXT,
            symbol TEXT,
            ovl_vix REAL,
            ovl_vix3m REAL,
            ovl_vix_term REAL,
            ovl_rates10y REAL,
            ovl_darkpool REAL,
            ovl_tariff REAL,
            ovl_vix_contango REAL,
            dp_shares_z_26w REAL,
            dp_trades_vs_13w REAL,
            dp_shares_vs_13w REAL,
            dp_avg_trade_size REAL
        )
        """
    )
    fieldnames = csv_columns(rows)
    insert_sql = (
        "INSERT INTO overlay_context_daily ("
        + ", ".join(fieldnames)
        + ") VALUES ("
        + ", ".join(["?"] * len(fieldnames))
        + ")"
    )
    payload = [tuple(row.get(name) for name in fieldnames) for row in rows]
    conn.executemany(insert_sql, payload)
    conn.commit()


def csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            "date",
            "symbol",
            *list(MACRO_TYPES.values()),
            *list(EVENT_TYPES.values()),
            "ovl_vix_contango",
            *list(WEEKLY_TYPES.values()),
        ]
    return list(rows[0].keys())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = csv_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    feature_columns = [
        column
        for column in rows[0]
        if column.startswith("ovl_") or column.startswith("dp_")
    ]
    total = len(rows)
    return {
        column: sum(1 for row in rows if row.get(column) is not None) / total
        for column in feature_columns
    }


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = build(conn)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(OUT_DIR / "overlay_context_daily.csv", rows)
        write_csv(
            OUT_DIR / "latest_overlay_context_daily.csv", rows[-1:] if rows else []
        )
        persist(conn, rows)
    finally:
        conn.close()

    cov = coverage(rows)
    feature_columns = list(cov.keys())
    if rows:
        print(
            f"overlay_context_daily: {len(rows)} trading days "
            f"[{rows[0]['date']} .. {rows[-1]['date']}]"
        )
    else:
        print("overlay_context_daily: 0 trading days")
    print(f"context columns: {feature_columns}")
    print("coverage (non-null fraction):")
    for column, value in cov.items():
        print(f"  {column:22s} {value:.3f}")
    if rows:
        print(f"\nlatest row:\n{rows[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
