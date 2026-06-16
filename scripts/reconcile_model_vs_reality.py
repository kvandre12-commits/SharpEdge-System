#!/usr/bin/env python3
"""Nightly model-vs-reality reconciliation.

The scoreboard. Every night this compares what the gate PREDICTED for each
session (final_bias + prob_trend from execution_state) against what the day
ACTUALLY did (day_type label + realized direction), and logs a hit/miss plus
calibration metrics. It is the honest evidence ledger that turns one-off
anecdotes into an accumulating, auditable track record.

READ-ONLY on sources. Writes two artifacts:
  outputs/reconcile_daily.csv    - one row per reconciled session
  outputs/reconcile_summary.txt  - rolling accuracy / Brier / calibration

  python3 scripts/reconcile_model_vs_reality.py
  python3 scripts/reconcile_model_vs_reality.py --window 60   # rolling window
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))


# --- pure mapping (unit-tested) -------------------------------------------
def predicted_regime(final_bias: str) -> str:
    """Map a gate bias to a regime call: 'trend' | 'range' | 'neutral'."""
    fb = (final_bias or "").upper()
    if fb.startswith("EXPANSION_FOLLOW"):
        return "trend"
    if fb in ("RANGE_FADE", "PIN_FADE"):
        return "range"
    return "neutral"  # BALANCED_SMALL / WHIP_WAIT = no decisive call


def predicted_direction(final_bias: str) -> int:
    """+1 long / -1 short / 0 none, from the directional bias states."""
    fb = (final_bias or "").upper()
    if fb == "EXPANSION_FOLLOW_LONG":
        return 1
    if fb == "EXPANSION_FOLLOW_SHORT":
        return -1
    # A fade at the call wall is short; at the put wall is long. Those are not
    # encoded in final_bias today, so fades are regime-only until the execution
    # layer writes its chosen direction back. Keep honest: 0 here.
    return 0


def reconcile(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    df = pred.merge(actual, on="session_date", how="inner").sort_values("session_date")
    df["pred_regime"] = df["final_bias"].map(predicted_regime)
    df["pred_dir"] = df["final_bias"].map(predicted_direction)
    df["actual_regime"] = df["day_type"].astype(str).str.lower()
    df["actual_dir"] = np.sign(df["ret_1d"].fillna(0.0)).astype(int)

    decisive = df["pred_regime"] != "neutral"
    df["regime_hit"] = np.where(decisive, df["pred_regime"] == df["actual_regime"], np.nan)

    directional = df["pred_dir"] != 0
    df["dir_hit"] = np.where(directional, df["pred_dir"] == df["actual_dir"], np.nan)

    df["y_trend"] = (df["actual_regime"] == "trend").astype(int)
    return df


# --- metrics ---------------------------------------------------------------
def brier(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2))


def calibration_table(df: pd.DataFrame, bins=(0.0, 0.35, 0.45, 0.55, 0.65, 1.01)) -> str:
    p = df["prob_trend_fused"].astype(float)
    y = df["y_trend"].astype(int)
    lines = ["  prob bucket    n   pred   observed_trend_rate"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        lines.append(f"  [{lo:.2f},{hi:.2f})  {m.sum():4d}  {p[m].mean():.3f}  {y[m].mean():.3f}")
    return "\n".join(lines)


def summarize(df: pd.DataFrame, window: int | None) -> dict:
    d = df.tail(window) if window else df
    reg = d["regime_hit"].dropna()
    drc = d["dir_hit"].dropna()
    p = d["prob_trend_fused"].astype(float).values
    y = d["y_trend"].astype(int).values
    base = y.mean() if len(y) else float("nan")
    return {
        "sessions": int(len(d)),
        "decisive_regime_calls": int(len(reg)),
        "regime_accuracy": float(reg.mean()) if len(reg) else None,
        "directional_calls": int(len(drc)),
        "direction_accuracy": float(drc.mean()) if len(drc) else None,
        "brier": brier(p, y) if len(y) else None,
        "brier_baseline_constant": brier(np.full_like(y, base, dtype=float), y) if len(y) else None,
        "trend_base_rate": float(base) if len(y) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0, help="rolling window (0 = all history)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    pred = pd.read_sql_query(
        "SELECT session_date, final_bias, prob_trend_fused, prob_range_fused, execution_score "
        "FROM execution_state_daily WHERE symbol = ?",
        conn, params=(SYMBOL,),
    )
    fcols = [r[1] for r in conn.execute("PRAGMA table_info(features_daily)")]
    date_col = "session_date" if "session_date" in fcols else "date"
    actual = pd.read_sql_query(
        f"SELECT {date_col} AS session_date, day_type, ret_1d "
        "FROM features_daily WHERE symbol = ? AND day_type IS NOT NULL",
        conn, params=(SYMBOL,),
    )
    for d in (pred, actual):
        d["session_date"] = pd.to_datetime(d["session_date"]).dt.date.astype(str)

    df = reconcile(pred.dropna(subset=["prob_trend_fused"]), actual)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["session_date", "final_bias", "prob_trend_fused", "pred_regime", "actual_regime",
            "regime_hit", "pred_dir", "actual_dir", "dir_hit", "execution_score"]
    df[keep].to_csv(OUT_DIR / "reconcile_daily.csv", index=False)

    window = args.window or None
    summary = summarize(df, window)
    summary["calibration"] = "see reconcile_summary.txt"

    lines = [
        "SharpEdge model-vs-reality reconciliation",
        f"window: {'all' if not window else f'last {window}'}  | sessions: {summary['sessions']}",
        "",
        f"regime accuracy:    {summary['regime_accuracy']}  "
        f"({summary['decisive_regime_calls']} decisive calls)",
        f"direction accuracy: {summary['direction_accuracy']}  "
        f"({summary['directional_calls']} directional calls)",
        f"Brier:              {summary['brier']:.4f}  "
        f"(constant-base-rate baseline {summary['brier_baseline_constant']:.4f}; lower is better)"
        if summary["brier"] is not None else "Brier: n/a",
        f"trend base rate:    {summary['trend_base_rate']:.3f}",
        "",
        "calibration (predicted prob_trend vs observed trend frequency):",
        calibration_table(df.tail(window) if window else df),
    ]
    report = "\n".join(lines)
    (OUT_DIR / "reconcile_summary.txt").write_text(report + "\n", encoding="utf-8")
    (OUT_DIR / "reconcile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2) if args.json else report)
    print(f"\nWrote {OUT_DIR}/reconcile_daily.csv + reconcile_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
