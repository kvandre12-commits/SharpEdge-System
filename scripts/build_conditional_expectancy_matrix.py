#!/usr/bin/env python3
"""
Aggregate auction expectancy events into conditional expectancy matrices.

Outputs:
- SQLite table: conditional_expectancy_matrix
- outputs/conditional_expectancy_matrix.csv
- outputs/top_gap_fill_edges.csv
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTDIR = os.getenv("OUTDIR", "outputs")
EVENTS_TABLE = os.getenv("AUCTION_EVENTS_TABLE", "auction_expectancy_events")
OUT_TABLE = os.getenv("CONDITIONAL_EXPECTANCY_TABLE", "conditional_expectancy_matrix")
MIN_N = int(os.getenv("MIN_N", "20"))
TOP_N = int(os.getenv("TOP_N", "50"))

GROUP_COLS = [
    "event_type",
    "gap_direction",
    "fill_path_type",
    "regime_id",
    "vol_state",
    "vol_trend_state",
    "macro_state",
    "dp_state",
    "open_regime_label",
    "gamma_state",
    "dealer_state_hint",
    "liquidity_regime_type",
]


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def table_exists(con, table):
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None


def safe_num(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index)


def ensure_output_table(con):
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
      event_type TEXT,
      gap_direction TEXT,
      fill_path_type TEXT,
      regime_id TEXT,
      vol_state TEXT,
      vol_trend_state TEXT,
      macro_state TEXT,
      dp_state TEXT,
      open_regime_label TEXT,
      gamma_state TEXT,
      dealer_state_hint TEXT,
      liquidity_regime_type TEXT,
      n INTEGER,
      fill_rate REAL,
      direct_fill_rate REAL,
      failed_fill_rate REAL,
      avg_time_to_fill_minutes REAL,
      median_time_to_fill_minutes REAL,
      avg_MAE_pct REAL,
      median_MAE_pct REAL,
      avg_MFE_pct REAL,
      median_MFE_pct REAL,
      payoff_ratio REAL,
      expectancy REAL,
      sortino_ratio REAL,
      max_drawdown REAL,
      stop_out_rate_proxy REAL,
      tradability_score REAL,
      sample_quality TEXT,
      build_ts TEXT
    )
    """)
    con.commit()


def load_events(con):
    if not table_exists(con, EVENTS_TABLE):
        raise RuntimeError(f"Missing table: {EVENTS_TABLE}")

    df = pd.read_sql_query(f"SELECT * FROM {EVENTS_TABLE}", con)
    if df.empty:
        raise RuntimeError(f"{EVENTS_TABLE} returned 0 rows")

    if "session_date" not in df.columns:
        if "date" in df.columns:
            df["session_date"] = df["date"]

    if "symbol" in df.columns:
        df = df[df["symbol"].fillna(SYMBOL) == SYMBOL].copy()

    required = {"event_type", "gap_direction", "session_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for c in GROUP_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    return df


def enrich_metrics(df):
    out = df.copy()

    out["_fill"] = safe_num(out, ["filled", "gap_filled", "fill_success"], 0).fillna(0)
    out["_direct_fill"] = (
        safe_num(out, ["direct_fill"], 0).fillna(0)
        | out["fill_path_type"].astype(str).str.upper().eq("DIRECT_FILL")
    ).astype(int)

    out["_failed_fill"] = (
        safe_num(out, ["failed_fill"], 0).fillna(0)
        | out["fill_path_type"].astype(str).str.upper().isin([
            "FAILED_FILL_CONTINUATION",
            "PARTIAL_FILL_REJECT",
            "LIQUIDITY_VACUUM_CONTINUATION",
        ])
    ).astype(int)

    out["_time_to_fill"] = safe_num(out, ["time_to_fill_minutes", "minutes_to_fill"])
    out["_mae"] = safe_num(out, ["MAE_pct", "mae_pct"])
    out["_mfe"] = safe_num(out, ["MFE_pct", "mfe_pct"])
    out["_stop"] = safe_num(out, ["stop_out_probability_proxy", "stop_out_rate_proxy"])
    out["_drawdown"] = safe_num(out, ["max_drawdown_intraday", "drawdown_intraday", "MAE_pct"])

    payoff = safe_num(out, ["reward_risk_realized", "realized_r", "ret"])
    missing = payoff.isna()
    payoff.loc[missing] = out.loc[missing, "_mfe"].fillna(0) - out.loc[missing, "_mae"].abs().fillna(0)
    out["_payoff"] = payoff

    return out


def sortino_ratio(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan
    dstd = downside.std(ddof=1)
    if dstd <= 0 or np.isnan(dstd):
        return np.nan
    return float(r.mean() / dstd)


def max_drawdown(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return np.nan
    eq = r.cumsum()
    dd = eq - eq.cummax()
    return float(dd.min())


def tradability_score(row):
    expectancy = np.nan_to_num(row["expectancy"])
    fill_rate = np.nan_to_num(row["fill_rate"])
    payoff = np.nan_to_num(row["payoff_ratio"])
    mae_penalty = abs(np.nan_to_num(row["avg_MAE_pct"]))
    drawdown_penalty = abs(np.nan_to_num(row["max_drawdown"]))
    failed_penalty = np.nan_to_num(row["failed_fill_rate"])

    score = (
        expectancy * 35
        + fill_rate * 20
        + payoff * 15
        - mae_penalty * 15
        - drawdown_penalty * 10
        - failed_penalty * 5
    )

    if row["n"] < MIN_N:
        score *= 0.75

    return float(score)


def build_matrix(df):
    rows = []

    for keys, g in df.groupby(GROUP_COLS, dropna=False):
        payoff = pd.to_numeric(g["_payoff"], errors="coerce")

        wins = payoff[payoff > 0]
        losses = payoff[payoff < 0].abs()

        payoff_ratio = np.nan
        if len(losses) > 0 and losses.mean() > 0:
            payoff_ratio = float(wins.mean() / losses.mean()) if len(wins) else 0.0

        row = dict(zip(GROUP_COLS, keys))
        row.update({
            "n": int(len(g)),
            "fill_rate": float(g["_fill"].mean()),
            "direct_fill_rate": float(g["_direct_fill"].mean()),
            "failed_fill_rate": float(g["_failed_fill"].mean()),
            "avg_time_to_fill_minutes": float(g["_time_to_fill"].mean()),
            "median_time_to_fill_minutes": float(g["_time_to_fill"].median()),
            "avg_MAE_pct": float(g["_mae"].mean()),
            "median_MAE_pct": float(g["_mae"].median()),
            "avg_MFE_pct": float(g["_mfe"].mean()),
            "median_MFE_pct": float(g["_mfe"].median()),
            "payoff_ratio": payoff_ratio,
            "expectancy": float(payoff.mean()),
            "sortino_ratio": sortino_ratio(payoff),
            "max_drawdown": max_drawdown(payoff),
            "stop_out_rate_proxy": float(g["_stop"].mean()),
        })

        row["sample_quality"] = "LOW_SAMPLE" if row["n"] < MIN_N else "OK"
        row["tradability_score"] = tradability_score(row)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No grouped expectancy rows produced")

    out = out.sort_values([
        "tradability_score",
        "expectancy",
        "fill_rate",
        "n",
    ], ascending=[False, False, False, False]).reset_index(drop=True)

    out["build_ts"] = datetime.now(timezone.utc).isoformat()
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    con = connect()
    try:
        ensure_output_table(con)

        events = load_events(con)
        events = enrich_metrics(events)

        matrix = build_matrix(events)

        matrix.to_sql(OUT_TABLE, con, if_exists="replace", index=False)

        matrix.to_csv(os.path.join(OUTDIR, "conditional_expectancy_matrix.csv"), index=False)

        top_edges = matrix[
            (matrix["n"] >= max(5, MIN_N // 2))
            & (matrix["expectancy"] > 0)
        ].head(TOP_N)

        top_edges.to_csv(os.path.join(OUTDIR, "top_gap_fill_edges.csv"), index=False)

        print(f"OK: wrote {OUT_TABLE} rows={len(matrix)}")
        print("OK: wrote outputs/conditional_expectancy_matrix.csv")
        print("OK: wrote outputs/top_gap_fill_edges.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
