#!/usr/bin/env python3
"""Aggregate candle expectancy events into a causal conditional matrix.

Inputs:
- candle_expectancy_events

Outputs:
- SQLite table: candle_conditioned_expectancy_matrix
- outputs/candle_conditioned_expectancy_matrix.csv

The matrix is tiered. Specific rows are nice, but SPY intraday samples get thin
fast. Tiers let a live reader start specific and relax without using outcome
fields or pretending a three-sample miracle is an edge.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
EVENTS_TABLE = os.getenv("CANDLE_EXPECTANCY_TABLE", "candle_expectancy_events")
OUT_TABLE = os.getenv(
    "CANDLE_EXPECTANCY_MATRIX_TABLE", "candle_conditioned_expectancy_matrix"
)
OUT = Path(
    os.getenv(
        "CANDLE_EXPECTANCY_MATRIX_OUT",
        "outputs/candle_conditioned_expectancy_matrix.csv",
    )
)
MIN_N = int(os.getenv("CANDLE_EXPECTANCY_MIN_N", "8"))

_SAFE_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

DIMENSIONS = [
    "event_name",
    "event_direction",
    "nearest_reference_name",
    "nearest_reference_relation",
    "reference_distance_bucket",
    "acceptance_state",
    "volume_confirmation",
    "vol_state",
    "macro_state",
    "dp_state",
    "regime_label",
    "open_regime_label",
    "time_bucket",
]

TIERS = [
    (
        "tier_1_full",
        [
            "event_name",
            "event_direction",
            "nearest_reference_name",
            "nearest_reference_relation",
            "reference_distance_bucket",
            "acceptance_state",
            "volume_confirmation",
            "vol_state",
            "macro_state",
            "dp_state",
            "regime_label",
            "open_regime_label",
            "time_bucket",
        ],
    ),
    (
        "tier_2_execution",
        [
            "event_name",
            "event_direction",
            "nearest_reference_name",
            "reference_distance_bucket",
            "acceptance_state",
            "volume_confirmation",
            "time_bucket",
        ],
    ),
    (
        "tier_3_core",
        ["event_name", "event_direction", "acceptance_state", "volume_confirmation"],
    ),
    ("tier_4_event_only", ["event_name", "event_direction"]),
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


def export_csv(con: sqlite3.Connection, table: str) -> int:
    cur = con.execute(
        f"""
        SELECT *
        FROM {table}
        ORDER BY
          CASE sample_quality
            WHEN 'usable' THEN 3
            WHEN 'thin' THEN 2
            ELSE 1
          END DESC,
          match_tier ASC,
          avg_realized_R DESC,
          n DESC
        """
    )
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def create_output_table(con: sqlite3.Connection, out_table: str) -> None:
    con.execute(f"DROP TABLE IF EXISTS {out_table}")
    con.execute(
        f"""
        CREATE TABLE {out_table} (
          match_tier TEXT,
          event_name TEXT,
          event_direction TEXT,
          nearest_reference_name TEXT,
          nearest_reference_relation TEXT,
          reference_distance_bucket TEXT,
          acceptance_state TEXT,
          volume_confirmation TEXT,
          vol_state TEXT,
          macro_state TEXT,
          dp_state TEXT,
          regime_label TEXT,
          open_regime_label TEXT,
          time_bucket TEXT,
          n INTEGER,
          target_before_stop_rate REAL,
          stop_before_target_rate REAL,
          same_bar_rate REAL,
          no_resolution_rate REAL,
          up_target_first_rate REAL,
          down_target_first_rate REAL,
          avg_realized_R REAL,
          avg_favorable_excursion_pct REAL,
          avg_adverse_excursion_pct REAL,
          sample_quality TEXT,
          build_ts TEXT
        )
        """
    )


def create_base_view(con: sqlite3.Connection, events_table: str) -> None:
    con.execute("DROP VIEW IF EXISTS candle_expectancy_matrix_base")
    con.execute(
        f"""
        CREATE TEMP VIEW candle_expectancy_matrix_base AS
        SELECT
          event_name,
          event_direction,
          COALESCE(nearest_reference_name, 'UNKNOWN') AS nearest_reference_name,
          COALESCE(nearest_reference_relation, 'UNKNOWN') AS nearest_reference_relation,
          CASE
            WHEN nearest_reference_distance_pct IS NULL THEN 'unknown_distance'
            WHEN nearest_reference_distance_pct <= 0.001 THEN 'at_reference'
            WHEN nearest_reference_distance_pct <= 0.003 THEN 'near_reference'
            ELSE 'away_from_reference'
          END AS reference_distance_bucket,
          COALESCE(acceptance_state, 'unknown') AS acceptance_state,
          COALESCE(volume_confirmation, 'unknown') AS volume_confirmation,
          COALESCE(vol_state, 'UNKNOWN') AS vol_state,
          COALESCE(macro_state, 'UNKNOWN') AS macro_state,
          COALESCE(dp_state, 'UNKNOWN') AS dp_state,
          COALESCE(regime_label, 'UNKNOWN') AS regime_label,
          COALESCE(open_regime_label, 'UNKNOWN') AS open_regime_label,
          CASE
            WHEN minutes_since_open IS NULL THEN 'unknown_time'
            WHEN minutes_since_open < 60 THEN 'opening_60m'
            WHEN minutes_since_open >= 330 THEN 'power_hour'
            ELSE 'midday'
          END AS time_bucket,
          target_before_stop_label,
          two_sided_first_touch,
          realized_R,
          favorable_excursion_pct,
          adverse_excursion_pct,
          forward_bar_count,
          horizon_bars
        FROM {events_table}
        WHERE forward_bar_count >= horizon_bars
          AND event_name NOT IN ('ordinary_range', 'insufficient_bar')
        """
    )


def tier_select(match_tier: str, group_cols: list[str]) -> str:
    dim_select = [col if col in group_cols else f"'ANY' AS {col}" for col in DIMENSIONS]
    group_clause = ", ".join(group_cols)
    return f"""
    SELECT
      '{match_tier}' AS match_tier,
      {", ".join(dim_select)},
      COUNT(*) AS n,
      AVG(CASE WHEN target_before_stop_label = 'target_before_stop' THEN 1.0 ELSE 0.0 END) AS target_before_stop_rate,
      AVG(CASE WHEN target_before_stop_label = 'stop_before_target' THEN 1.0 ELSE 0.0 END) AS stop_before_target_rate,
      AVG(CASE WHEN target_before_stop_label = 'same_bar_target_stop' THEN 1.0 ELSE 0.0 END) AS same_bar_rate,
      AVG(CASE WHEN target_before_stop_label = 'no_resolution' THEN 1.0 ELSE 0.0 END) AS no_resolution_rate,
      AVG(CASE WHEN two_sided_first_touch = 'up_target_first' THEN 1.0 ELSE 0.0 END) AS up_target_first_rate,
      AVG(CASE WHEN two_sided_first_touch = 'down_target_first' THEN 1.0 ELSE 0.0 END) AS down_target_first_rate,
      AVG(realized_R) AS avg_realized_R,
      AVG(favorable_excursion_pct) AS avg_favorable_excursion_pct,
      AVG(adverse_excursion_pct) AS avg_adverse_excursion_pct,
      CASE
        WHEN COUNT(*) >= ? THEN 'usable'
        WHEN COUNT(*) >= MAX(3, ? / 2) THEN 'thin'
        ELSE 'too_small'
      END AS sample_quality,
      ? AS build_ts
    FROM candle_expectancy_matrix_base
    GROUP BY {group_clause}
    """


def build_matrix(con: sqlite3.Connection, events_table: str, out_table: str) -> None:
    build_ts = datetime.now(timezone.utc).isoformat()
    create_output_table(con, out_table)
    create_base_view(con, events_table)

    columns = ["match_tier", *DIMENSIONS]
    metrics = [
        "n",
        "target_before_stop_rate",
        "stop_before_target_rate",
        "same_bar_rate",
        "no_resolution_rate",
        "up_target_first_rate",
        "down_target_first_rate",
        "avg_realized_R",
        "avg_favorable_excursion_pct",
        "avg_adverse_excursion_pct",
        "sample_quality",
        "build_ts",
    ]
    for match_tier, group_cols in TIERS:
        sql = f"""
        INSERT INTO {out_table} ({", ".join([*columns, *metrics])})
        {tier_select(match_tier, group_cols)}
        """
        con.execute(sql, (MIN_N, MIN_N, build_ts))
    con.commit()


def main() -> int:
    events_table = safe_identifier(EVENTS_TABLE, "events table")
    out_table = safe_identifier(OUT_TABLE, "matrix table")
    con = sqlite3.connect(DB_PATH)
    try:
        if not exists(con, events_table):
            raise RuntimeError(
                f"Missing {events_table}; run scripts/build_candle_expectancy_events.py first"
            )
        build_matrix(con, events_table, out_table)
        row_count = export_csv(con, out_table)
        event_count = con.execute(f"SELECT COUNT(*) FROM {events_table}").fetchone()[0]
        print(
            f"OK: wrote {out_table} rows={row_count} from {event_count} raw candle events"
        )
        print(f"OK: wrote {OUT}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
