#!/usr/bin/env python3
"""
Measure path-aware gap excursion / tradability metrics for auction_expectancy_events.

Inputs:
- auction_expectancy_events
- intraday bars table from INTRADAY_BARS_TABLE, default spy_bars_15m

Writes:
- Updates auction_expectancy_events metric columns in-place
- outputs/gap_excursion_metrics.csv

Environment knobs:
- SPY_DB_PATH=data/spy_truth.db
- SYMBOL=SPY
- INTRADAY_BARS_TABLE=spy_bars_15m
- PREMARKET_INCLUDE=0        # 1 includes all session bars; 0 uses 09:30-16:00 NY only
- MIN_STOP_DISTANCE_PCT=0.003 # 0.30% floor for tradability proxy
- STOP_GAP_FRACTION=0.50     # expected stop = 50% of absolute gap pct, floored above
- MAX_STOP_DISTANCE_PCT=0.020 # cap expected stop proxy at 2.0%
"""

import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
PREMARKET_INCLUDE = int(os.getenv("PREMARKET_INCLUDE", "0"))

MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.003"))
STOP_GAP_FRACTION = float(os.getenv("STOP_GAP_FRACTION", "0.50"))
MAX_STOP_DISTANCE_PCT = float(os.getenv("MAX_STOP_DISTANCE_PCT", "0.020"))

NY = ZoneInfo("America/New_York")

REQUIRED_EVENT_COLS = {
    "session_date",
    "symbol",
    "event_type",
    "gap_pct",
    "gap_direction",
    "prior_close",
    "session_open",
    "fill_completed",
}

METRIC_COLS = {
    "MAE_pct": "REAL",
    "MFE_pct": "REAL",
    "MAE_points": "REAL",
    "MFE_points": "REAL",
    "max_drawdown_intraday": "REAL",
    "max_runup_intraday": "REAL",
    "time_to_MAE_minutes": "REAL",
    "time_to_MFE_minutes": "REAL",
    "time_to_failure_minutes": "REAL",
    "stop_out_probability_proxy": "REAL",
    "expected_stop_distance_pct": "REAL",
    "reward_risk_realized": "REAL",
}


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def existing_cols(con: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def ensure_event_columns(con: sqlite3.Connection) -> None:
    cols = existing_cols(con, "auction_expectancy_events")

    missing_required = REQUIRED_EVENT_COLS - cols
    if missing_required:
        raise RuntimeError(
            f"auction_expectancy_events missing required columns: {sorted(missing_required)}"
        )

    for col, typ in METRIC_COLS.items():
        if col not in cols:
            con.execute(
                f"ALTER TABLE auction_expectancy_events ADD COLUMN {col} {typ}"
            )

    con.commit()


def ensure_output_dir() -> None:
    os.makedirs("outputs", exist_ok=True)


def parse_ts(ts_text: str) -> datetime:
    return datetime.fromisoformat(ts_text.replace("Z", "+00:00"))


def is_rth(ts_text: str) -> bool:
    dt_ny = parse_ts(ts_text).astimezone(NY)
    t = dt_ny.time()
    return (
        t >= datetime(2000, 1, 1, 9, 30).time()
        and t <= datetime(2000, 1, 1, 16, 0).time()
    )


def load_events(con: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT *
    FROM auction_expectancy_events
    WHERE symbol = ?
    ORDER BY session_date ASC
    """

    df = pd.read_sql_query(q, con, params=(SYMBOL,))

    if df.empty:
        raise RuntimeError("auction_expectancy_events returned 0 rows.")

    return df


def load_intraday_for_session(
    con: sqlite3.Connection,
    session_date: str,
) -> pd.DataFrame:
    cols = existing_cols(con, BARS_TABLE)

    required = {
        "ts",
        "session_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - cols
    if missing:
        raise RuntimeError(
            f"{BARS_TABLE} missing required columns: {sorted(missing)}"
        )

    q = f"""
    SELECT
      ts,
      open,
      high,
      low,
      close
    FROM {BARS_TABLE}
    WHERE symbol = ?
      AND session_date = ?
    ORDER BY ts ASC
    """

    bars = pd.read_sql_query(q, con, params=(SYMBOL, session_date))

    if bars.empty:
        return bars

    if not PREMARKET_INCLUDE:
        bars = bars[bars["ts"].apply(is_rth)].copy()

    return bars.reset_index(drop=True)


def minutes_between(ts1: str, ts2: str) -> float:
    return (
        (parse_ts(ts2) - parse_ts(ts1)).total_seconds() / 60.0
    )


def infer_direction(row: pd.Series) -> str:
    gd = str(row.get("gap_direction", "")).upper()

    if gd in {"DOWN", "GAP_DOWN"}:
        return "UP"

    if gd in {"UP", "GAP_UP"}:
        return "DOWN"

    gap_pct = float(row.get("gap_pct", 0.0) or 0.0)

    return "UP" if gap_pct < 0 else "DOWN"


def compute_expected_stop_distance_pct(gap_pct: float) -> float:
    stop_dist = max(
        abs(gap_pct) * STOP_GAP_FRACTION,
        MIN_STOP_DISTANCE_PCT,
    )

    return min(stop_dist, MAX_STOP_DISTANCE_PCT)


def compute_metrics_for_event(
    row: pd.Series,
    bars: pd.DataFrame,
) -> Dict[str, Any]:

    out = {k: None for k in METRIC_COLS.keys()}

    if bars.empty:
        return out

    entry = float(row["session_open"])
    target = float(row["prior_close"])
    direction = infer_direction(row)

    first_ts = str(bars.iloc[0]["ts"])

    favorable_moves = []
    adverse_moves = []

    mfe_idx = None
    mae_idx = None

    fill_idx = None
    stop_idx = None

    expected_stop_pct = compute_expected_stop_distance_pct(
        float(row.get("gap_pct", 0.0) or 0.0)
    )

    if direction == "UP":
        stop_price = entry * (1.0 - expected_stop_pct)
    else:
        stop_price = entry * (1.0 + expected_stop_pct)

    for idx, b in bars.iterrows():
        high = float(b["high"])
        low = float(b["low"])
        ts = str(b["ts"])

        if direction == "UP":
            favorable = high - entry
            adverse = entry - low

            filled = high >= target
            stopped = low <= stop_price

        else:
            favorable = entry - low
            adverse = high - entry

            filled = low <= target
            stopped = high >= stop_price

        favorable_moves.append(favorable)
        adverse_moves.append(adverse)

        if fill_idx is None and filled:
            fill_idx = idx

        if stop_idx is None and stopped:
            stop_idx = idx

    favorable_arr = np.array(favorable_moves, dtype=float)
    adverse_arr = np.array(adverse_moves, dtype=float)

    mfe_points = float(np.nanmax(favorable_arr)) if len(favorable_arr) else np.nan
    mae_points = float(np.nanmax(adverse_arr)) if len(adverse_arr) else np.nan

    mfe_idx = int(np.nanargmax(favorable_arr)) if len(favorable_arr) else None
    mae_idx = int(np.nanargmax(adverse_arr)) if len(adverse_arr) else None

    mfe_pct = mfe_points / entry if entry else np.nan
    mae_pct = mae_points / entry if entry else np.nan

    out["MFE_points"] = mfe_points
    out["MAE_points"] = mae_points
    out["MFE_pct"] = mfe_pct
    out["MAE_pct"] = mae_pct

    out["max_runup_intraday"] = mfe_points
    out["max_drawdown_intraday"] = mae_points

    if mfe_idx is not None:
        out["time_to_MFE_minutes"] = minutes_between(
            first_ts,
            str(bars.iloc[mfe_idx]["ts"]),
        )

    if mae_idx is not None:
        out["time_to_MAE_minutes"] = minutes_between(
            first_ts,
            str(bars.iloc[mae_idx]["ts"]),
        )

    if stop_idx is not None:
        out["time_to_failure_minutes"] = minutes_between(
            first_ts,
            str(bars.iloc[stop_idx]["ts"]),
        )

    stop_prob = float(np.mean(adverse_arr >= (entry * expected_stop_pct)))

    out["stop_out_probability_proxy"] = stop_prob
    out["expected_stop_distance_pct"] = expected_stop_pct

    realized_risk = mae_points if mae_points > 0 else np.nan

    if realized_risk and np.isfinite(realized_risk):
        out["reward_risk_realized"] = mfe_points / realized_risk
    else:
        out["reward_risk_realized"] = np.nan

    return out


def upsert_metrics(
    con: sqlite3.Connection,
    row: pd.Series,
    metrics: Dict[str, Any],
) -> None:

    assignments = ",\n".join([
        f"{k} = :{k}" for k in metrics.keys()
    ])

    payload = dict(metrics)
    payload["session_date"] = row["session_date"]
    payload["symbol"] = row["symbol"]
    payload["event_type"] = row["event_type"]

    q = f"""
    UPDATE auction_expectancy_events
    SET
      {assignments}
    WHERE session_date = :session_date
      AND symbol = :symbol
      AND event_type = :event_type
    """

    con.execute(q, payload)


def main():
    ensure_output_dir()

    con = connect()

    try:
        ensure_event_columns(con)

        events = load_events(con)

        all_rows = []

        for _, row in events.iterrows():
            session_date = str(row["session_date"])

            bars = load_intraday_for_session(con, session_date)

            metrics = compute_metrics_for_event(row, bars)

            upsert_metrics(con, row, metrics)

            merged = dict(row)
            merged.update(metrics)

            all_rows.append(merged)

        con.commit()

        out = pd.DataFrame(all_rows)

        out_path = "outputs/gap_excursion_metrics.csv"
        out.to_csv(out_path, index=False)

        print(
            f"OK: updated auction_expectancy_events + wrote {out_path} rows={len(out)}"
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
