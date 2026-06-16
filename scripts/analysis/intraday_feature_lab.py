#!/usr/bin/env python3
"""Intraday after-open feature lab: judge features by OUT-OF-SAMPLE AUC.

The trend/range model's ceiling is feature quality (train AUC ~0.62). This lab
exists so feature ideas are judged honestly - by walk-forward OOS AUC, never by
the optimistic train fit. It is READ-ONLY: it never writes the DB. Once a feature
set genuinely lifts OOS AUC here, it gets promoted into
build_intraday_trendday_prob_1130_fused.compute_intraday_features.

  python3 scripts/analysis/intraday_feature_lab.py            # baseline vs rich
  python3 scripts/analysis/intraday_feature_lab.py --folds 6

We reuse the production loaders/fit so the lab and prod can never drift.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sqlite3
from datetime import time

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
CUTOFF = time(11, 30)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROD = os.path.join(_HERE, "..", "build_intraday_trendday_prob_1130_fused.py")
_spec = importlib.util.spec_from_file_location("prodmod", _PROD)
prod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prod)

BASELINE = [
    "ret_open_to_cutoff", "orbrange_pct", "orb_break_strength",
    "range_pct_to_cutoff", "true_range_pct_to_cutoff",
    "hhll_persistence", "vwap_proxy",
]


def rich_features(g_cut: pd.DataFrame) -> dict:
    """Texture of the open->cutoff tape. These are the candidate additions."""
    o = float(g_cut.iloc[0]["open"])
    c = float(g_cut.iloc[-1]["close"])
    closes = g_cut["close"].astype(float).values
    opens = g_cut["open"].astype(float).values
    highs = g_cut["high"].astype(float).values
    lows = g_cut["low"].astype(float).values
    vols = g_cut["volume"].astype(float).values
    n = len(closes)
    net_dir = np.sign(c - o) or 1.0

    # running VWAP and one-sidedness of the tape
    typ = (highs + lows + closes) / 3.0
    cum_v = np.cumsum(vols)
    cum_pv = np.cumsum(typ * vols)
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), closes)
    vwap_above_frac = float(np.mean(closes > vwap))

    # opening drive: first-bar signed strength
    open_drive = (closes[0] / opens[0] - 1.0) * 100.0 if opens[0] else 0.0

    # bar bodies vs ranges: directional conviction vs wicky chop
    rng = np.maximum(highs - lows, 1e-9)
    body_ratio = float(np.mean(np.abs(closes - opens) / rng))

    # longest consecutive same-direction streak (normalized)
    deltas = np.sign(np.diff(closes))
    max_run = run = 0
    for d in deltas:
        run = run + 1 if d == net_dir else 0
        max_run = max(max_run, run)
    max_run_frac = max_run / max(n - 1, 1)

    # deepest pullback against net direction, as % of the move
    path = (closes - o) * net_dir  # positive = with the trend
    peak = np.maximum.accumulate(np.maximum(path, 0.0))
    drawdown = peak - path
    pullback_depth = float(np.max(drawdown) / max(np.max(np.abs(path)), 1e-9))

    # volume expansion: second half vs first half
    half = max(n // 2, 1)
    v1, v2 = vols[:half].mean(), vols[half:].mean() if n > half else vols.mean()
    vol_slope = float((v2 - v1) / max(v1, 1e-9))

    # effort: net signed volume share
    bar_dir = np.sign(closes - opens)
    signed_vol = float(np.sum(bar_dir * vols) / max(np.sum(vols), 1e-9))

    # close location within the cutoff range (1=at highs, 0=at lows)
    hi, lo = highs.max(), lows.min()
    close_loc = float((c - lo) / max(hi - lo, 1e-9))

    # late acceleration: last-third vs first-third drift
    third = max(n // 3, 1)
    early = (closes[third - 1] / opens[0] - 1.0) * 100.0 if opens[0] else 0.0
    late = (closes[-1] / closes[-1 - third] - 1.0) * 100.0 if n > third else 0.0
    late_accel = late - early

    return {
        "f_vwap_above_frac": vwap_above_frac,
        "f_open_drive": open_drive,
        "f_body_ratio": body_ratio,
        "f_max_run_frac": max_run_frac,
        "f_pullback_depth": pullback_depth,
        "f_vol_slope": vol_slope,
        "f_signed_vol": signed_vol,
        "f_close_loc": close_loc,
        "f_late_accel": late_accel,
    }


def build_matrix(conn) -> pd.DataFrame:
    labels = prod.load_daily_labels(conn, SYMBOL)
    intra = prod.load_intraday(conn, SYMBOL)
    base = prod.compute_intraday_features(intra)

    rich_rows = []
    for session_date, g in intra.groupby("session_date", sort=True):
        g = g.sort_values("ts_ny")
        g = g[(g["time_ny"] >= time(9, 30)) & (g["time_ny"] <= time(16, 0))]
        g_cut = g[g["time_ny"] <= CUTOFF]
        if len(g_cut) < 3:
            continue
        row = {"session_date": session_date, "symbol": SYMBOL}
        row.update(rich_features(g_cut))
        rich_rows.append(row)
    rich = pd.DataFrame(rich_rows)

    df = base.merge(rich, on=["session_date", "symbol"], how="inner")
    df = df.merge(labels, on=["session_date", "symbol"], how="inner")
    return df.sort_values("session_date").reset_index(drop=True)


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC (Mann-Whitney), tie-aware
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def walk_forward_auc(df: pd.DataFrame, feats: list[str], folds: int) -> float:
    X = df[feats].astype(float).values
    y = df["y"].astype(int).values
    n = len(df)
    start = n // (folds + 1)
    preds, truth = [], []
    for k in range(1, folds + 1):
        cut = start * k
        tr, te = slice(0, cut), slice(cut, min(cut + start, n))
        if te.stop <= te.start or y[tr].sum() in (0, (tr.stop - tr.start)):
            continue
        Xs, mu, sig = prod.standardize(X[tr])
        w, b = prod.fit_logreg(Xs, y[tr], l2=0.01, lr=0.2, steps=400)
        Xte = (X[te] - mu) / sig
        preds.append(prod.sigmoid(Xte @ w + b))
        truth.append(y[te])
    if not preds:
        return float("nan")
    return _auc(np.concatenate(preds), np.concatenate(truth))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    df = build_matrix(conn)
    rich_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"samples={len(df)}  trend_rate={df['y'].mean():.3f}  folds={args.folds}")

    base_auc = walk_forward_auc(df, BASELINE, args.folds)
    all_auc = walk_forward_auc(df, BASELINE + rich_cols, args.folds)
    print(f"\nOOS AUC  baseline (7 feats)      = {base_auc:.4f}")
    print(f"OOS AUC  baseline + rich         = {all_auc:.4f}   (delta {all_auc - base_auc:+.4f})")

    print("\nPer-feature OOS AUC (baseline + ONE rich feature):")
    rows = []
    for f in rich_cols:
        a = walk_forward_auc(df, BASELINE + [f], args.folds)
        rows.append((a - base_auc, a, f))
    for d, a, f in sorted(rows, reverse=True):
        print(f"  {a:.4f}  ({d:+.4f})  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
