#!/usr/bin/env python3
"""Unify every overlay into ONE per-trading-day context layer.

Overlays are scattered: macro (vix/vix3m/vix_term/rates10y) + darkpool live
'tall' in overlays_daily (one row per type per date); richer dark-pool weekly
stats live in ats_weekly; tariff is sparse. Nothing downstream can consume that
shape. This builder pivots them onto the trading-day spine (bars_daily) into a
single wide table/CSV: overlay_context_daily.

It is a CONTEXT layer, not a model. It does not decide anything - it just makes
the overlays joinable in one place so the feature lab (rung 4) and the gate can
use them, and the reconciliation (rung 1) can later prove whether they help.

  python3 scripts/build_overlay_context_daily.py
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))

# overlay_type -> output column. Macro types are continuous (forward-fill ok);
# darkpool/tariff are event-ish (left as-is, NaN where absent).
MACRO_TYPES = {"vix": "ovl_vix", "vix3m": "ovl_vix3m",
               "vix_term": "ovl_vix_term", "rates10y": "ovl_rates10y"}
EVENT_TYPES = {"darkpool": "ovl_darkpool", "tariff": "ovl_tariff"}


def _trading_days(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date FROM bars_daily WHERE symbol = ? ORDER BY date",
        conn, params=(SYMBOL,),
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def _pivot_overlays(conn) -> pd.DataFrame:
    raw = pd.read_sql_query(
        "SELECT date, overlay_type, overlay_strength FROM overlays_daily WHERE symbol = ?",
        conn, params=(SYMBOL,),
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.date.astype(str)
    wide = raw.pivot_table(index="date", columns="overlay_type",
                           values="overlay_strength", aggfunc="last")
    rename = {**MACRO_TYPES, **EVENT_TYPES}
    wide = wide.rename(columns=rename)
    keep = [c for c in rename.values() if c in wide.columns]
    return wide[keep].reset_index()


def _weekly_darkpool(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT week_start, shares_z_26w, trades_vs_13w_avg, shares_vs_13w_avg, "
        "avg_trade_size FROM ats_weekly WHERE symbol = ? ORDER BY week_start",
        conn, params=(SYMBOL,),
    )
    if df.empty:
        return df
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.rename(columns={
        "shares_z_26w": "dp_shares_z_26w",
        "trades_vs_13w_avg": "dp_trades_vs_13w",
        "shares_vs_13w_avg": "dp_shares_vs_13w",
        "avg_trade_size": "dp_avg_trade_size",
    })


def build(conn) -> pd.DataFrame:
    spine = _trading_days(conn)
    wide = _pivot_overlays(conn)
    ctx = spine.merge(wide, on="date", how="left").sort_values("date")

    # macro is continuous market state -> forward-fill across non-quote gaps.
    for col in MACRO_TYPES.values():
        if col in ctx.columns:
            ctx[col] = ctx[col].ffill()

    # derived: VIX term structure (contango>0 = calm, backwardation<0 = stress).
    if {"ovl_vix3m", "ovl_vix"}.issubset(ctx.columns):
        ctx["ovl_vix_contango"] = ctx["ovl_vix3m"] - ctx["ovl_vix"]

    # weekly dark-pool: as-of join (most recent week_start <= trading day).
    weekly = _weekly_darkpool(conn)
    if not weekly.empty:
        left = ctx.assign(_d=pd.to_datetime(ctx["date"]))
        ctx = pd.merge_asof(
            left.sort_values("_d"), weekly.sort_values("week_start"),
            left_on="_d", right_on="week_start", direction="backward",
        ).drop(columns=["_d", "week_start"])

    ctx["symbol"] = SYMBOL
    front = ["date", "symbol"]
    cols = front + [c for c in ctx.columns if c not in front]
    return ctx[cols]


def persist(conn, df: pd.DataFrame) -> None:
    df.to_sql("overlay_context_daily", conn, if_exists="replace", index=False)
    conn.commit()


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    ctx = build(conn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx.to_csv(OUT_DIR / "overlay_context_daily.csv", index=False)
    ctx.tail(1).to_csv(OUT_DIR / "latest_overlay_context_daily.csv", index=False)
    persist(conn, ctx)

    feat_cols = [c for c in ctx.columns if c.startswith(("ovl_", "dp_"))]
    cov = {c: float(ctx[c].notna().mean()) for c in feat_cols}
    print(f"overlay_context_daily: {len(ctx)} trading days "
          f"[{ctx['date'].min()} .. {ctx['date'].max()}]")
    print(f"context columns: {feat_cols}")
    print("coverage (non-null fraction):")
    for c, v in cov.items():
        print(f"  {c:22s} {v:.3f}")
    print(f"\nlatest row:\n{ctx.tail(1).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
