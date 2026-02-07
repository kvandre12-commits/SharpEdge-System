#!/usr/bin/env python3
"""
3:00pm Gamma Close Card (Decision Support)

Goal:
  Give a late-day plan: PIN fade vs EXPANSION follow vs WHIP wait
  using the *latest* options_positioning_metrics snapshot for the session.

Reads (SQLite):
  - options_positioning_metrics (required)  3
  - overlays_daily (optional: dealer_pin_score / dealer_expand_score / dealer_whip_score / dealer_late_day_mode)
  - signals_daily (optional)
  - features_daily (optional)

Outputs:
  - Console trade card for the latest session_date (or --date)

Usage:
  python scripts/print_gamma_close_card_300.py
  python scripts/print_gamma_close_card_300.py --date 2026-02-06
"""

import os
import sqlite3
import argparse
import pandas as pd
import numpy as np

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# pin threshold in percent (0.25% default aligns with your aggregator) 4
PIN_DIST_PCT = float(os.getenv("PIN_DIST_PCT", "0.25"))

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return r is not None

def pick_latest_session_date(conn: sqlite3.Connection, symbol: str) -> str:
    # prefer options metrics (most relevant for late-day)
    if table_exists(conn, "options_positioning_metrics"):
        row = conn.execute(
            "SELECT MAX(session_date) FROM options_positioning_metrics WHERE underlying=?",
            (symbol,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])[:10]
    # fallback to features_daily
    if table_exists(conn, "features_daily"):
        cols = pd.read_sql_query("PRAGMA table_info(features_daily);", conn)["name"].tolist()
        date_col = "session_date" if "session_date" in cols else ("date" if "date" in cols else None)
        if date_col:
            row = conn.execute(
                f"SELECT MAX({date_col}) FROM features_daily WHERE symbol=?",
                (symbol,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])[:10]
    raise RuntimeError("Could not determine latest session_date.")

def load_latest_metrics_for_date(conn: sqlite3.Connection, symbol: str, session_date: str) -> dict:
    if not table_exists(conn, "options_positioning_metrics"):
        raise RuntimeError("Missing options_positioning_metrics. Run your aggregation first.")

    df = pd.read_sql_query(
        """
        WITH latest AS (
          SELECT MAX(snapshot_ts) AS snapshot_ts
          FROM options_positioning_metrics
          WHERE underlying=? AND session_date=?
        )
        SELECT
          session_date,
          underlying AS symbol,
          snapshot_ts,
          spot,
          max_total_oi_strike AS wall_strike,
          max_call_oi_strike,
          max_put_oi_strike,
          total_call_oi,
          total_put_oi,
          pcr_oi,
          total_call_vol,
          total_put_vol,
          pcr_vol,
          gamma_proxy,
          dealer_state_hint
        FROM options_positioning_metrics
        WHERE underlying=? AND session_date=?
          AND snapshot_ts = (SELECT snapshot_ts FROM latest)
        LIMIT 1
        """,
        conn, params=(symbol, session_date, symbol, session_date),
    )
    if df.empty:
        raise RuntimeError(f"No options_positioning_metrics rows for {symbol} on {session_date}.")
    m = df.iloc[0].to_dict()

    # distance to wall (%)
    spot = m.get("spot")
    wall = m.get("wall_strike")
    if spot is not None and wall is not None and float(spot) != 0:
        m["dist_to_wall_pct"] = abs(float(spot) - float(wall)) / float(spot) * 100.0
    else:
        m["dist_to_wall_pct"] = None

    return m

def load_optional_overlay(conn: sqlite3.Connection, symbol: str, session_date: str) -> dict:
    if not table_exists(conn, "overlays_daily"):
        return {}
    cols = pd.read_sql_query("PRAGMA table_info(overlays_daily);", conn)["name"].tolist()

    wanted = [
        "dealer_pin_score","dealer_expand_score","dealer_whip_score","dealer_late_day_mode",
        "dealer_wall_strike","dealer_dist_to_wall_pct"
    ]
    present = [c for c in wanted if c in cols]
    if not present:
        return {}

    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(present)}
        FROM overlays_daily
        WHERE symbol=? AND session_date=?
        LIMIT 1
        """,
        conn, params=(symbol, session_date),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

def load_optional_features(conn: sqlite3.Connection, symbol: str, session_date: str) -> dict:
    if not table_exists(conn, "features_daily"):
        return {}
    cols = pd.read_sql_query("PRAGMA table_info(features_daily);", conn)["name"].tolist()
    date_col = "session_date" if "session_date" in cols else ("date" if "date" in cols else None)
    if not date_col:
        return {}

    keep = [c for c in ["cluster_score","compression_flag","intraday_range_pct","true_range_pct","day_type"] if c in cols]
    if not keep:
        return {}

    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(keep)}
        FROM features_daily
        WHERE symbol=? AND {date_col}=?
        LIMIT 1
        """,
        conn, params=(symbol, session_date),
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

def fmt_num(x, d=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return "—"

def fmt_pct(x, d=2):
    v = fmt_num(x, d)
    return v + "%" if v != "—" else "—"

def decide_plan(metrics: dict, overlay: dict) -> tuple[str, float, list[str]]:
    # Prefer overlay mode if present; else derive from metrics + wall distance + dealer_state_hint.
    mode = (overlay.get("dealer_late_day_mode") or "").upper()
    pin_s = overlay.get("dealer_pin_score")
    exp_s = overlay.get("dealer_expand_score")
    whp_s = overlay.get("dealer_whip_score")

    dist = metrics.get("dist_to_wall_pct")
    st = (metrics.get("dealer_state_hint") or "").lower()

    # If overlay exists, use it
    if mode in {"PIN","EXPAND","WHIP"}:
        bias = {"PIN":"PIN_FADE", "EXPAND":"EXPANSION_FOLLOW", "WHIP":"WHIP_WAIT"}[mode]
        # confidence from max score
        scores = [s for s in [pin_s, exp_s, whp_s] if isinstance(s, (int, float)) and s is not None]
        conf = float(np.clip(max(scores) if scores else 50.0, 0.0, 100.0))
    else:
        # derive
        conf = 55.0
        if isinstance(dist, (int, float)) and dist is not None and float(dist) <= PIN_DIST_PCT:
            bias = "PIN_FADE"
            conf = 72.0
        elif st == "chase":
            bias = "EXPANSION_FOLLOW"
            conf = 70.0
        elif st == "unwind":
            bias = "PIN_FADE"
            conf = 62.0
        else:
            bias = "WHIP_WAIT"
            conf = 55.0

    # action bullets
    if bias == "PIN_FADE":
        bullets = [
            "Core idea: dealer hedging tends to pull price back toward the wall/ATM late-day.",
            "Execution: fade edges → target wall; take profits quickly; avoid chasing breakouts.",
            "Trigger: if price holds away from wall with expanding candles/volume, stop fading."
        ]
    elif bias == "EXPANSION_FOLLOW":
        bullets = [
            "Core idea: negative gamma / chase behavior can create late-day acceleration.",
            "Execution: join the move on pullback after break; hold runner into close.",
            "Trigger: if move stalls and snaps back toward wall, reduce and protect."
        ]
    else:
        bullets = [
            "Core idea: late-day fakeouts are common—trade the second move, not the first.",
            "Execution: wait for break + retest / confirmation candle; smaller size.",
            "Trigger: if price pins tightly to wall, switch to PIN_FADE."
        ]

    return bias, conf, bullets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="YYYY-MM-DD (optional)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        session_date = args.date.strip() or pick_latest_session_date(conn, SYMBOL)

        metrics = load_latest_metrics_for_date(conn, SYMBOL, session_date)
        overlay = load_optional_overlay(conn, SYMBOL, session_date)
        feats = load_optional_features(conn, SYMBOL, session_date)

        bias, conf, bullets = decide_plan(metrics, overlay)

        line = "═" * 72
        print(line)
        print(f"GAMMA CLOSE CARD | {SYMBOL} | session_date: {session_date} | snapshot_ts: {metrics.get('snapshot_ts')}")
        print(line)
        print(f"Final Bias        : {bias}")
        print(f"Confidence        : {fmt_num(conf,1)} / 100")
        print("")
        print("Key Dealer Metrics")
        print(f"  Dealer State    : {metrics.get('dealer_state_hint') or '—'}")
        print(f"  Gamma Proxy     : {fmt_num(metrics.get('gamma_proxy'), 2)}")
        print(f"  Spot (proxy)    : {fmt_num(metrics.get('spot'), 2)}")
        print(f"  Wall (max OI)   : {fmt_num(metrics.get('wall_strike'), 2)}")
        print(f"  Dist → Wall     : {fmt_pct(metrics.get('dist_to_wall_pct'), 2)}  (<= {PIN_DIST_PCT:.2f}% = pin-risk)")
        print("")
        print("Walls / Positioning")
        print(f"  Max Call OI     : {fmt_num(metrics.get('max_call_oi_strike'), 2)}")
        print(f"  Max Put OI      : {fmt_num(metrics.get('max_put_oi_strike'), 2)}")
        print(f"  PCR (OI)        : {fmt_num(metrics.get('pcr_oi'), 2)}")
        print(f"  PCR (Vol)       : {fmt_num(metrics.get('pcr_vol'), 2)}")

        if overlay:
            print("")
            print("Late-Day Overlay (optional)")
            if "dealer_late_day_mode" in overlay:
                print(f"  Mode            : {overlay.get('dealer_late_day_mode')}")
            if "dealer_pin_score" in overlay:
                print(f"  Pin Score       : {fmt_num(overlay.get('dealer_pin_score'), 1)}")
            if "dealer_expand_score" in overlay:
                print(f"  Expand Score    : {fmt_num(overlay.get('dealer_expand_score'), 1)}")
            if "dealer_whip_score" in overlay:
                print(f"  Whip Score      : {fmt_num(overlay.get('dealer_whip_score'), 1)}")

        if feats:
            print("")
            print("Structure (optional)")
            if "compression_flag" in feats:
                print(f"  Compression     : {feats.get('compression_flag')} (1=yes)")
            if "cluster_score" in feats:
                print(f"  Cluster Score   : {fmt_num(feats.get('cluster_score'), 3)}")
            if "intraday_range_pct" in feats:
                print(f"  Intraday Range% : {fmt_pct(feats.get('intraday_range_pct'), 2)}")
            if "true_range_pct" in feats:
                print(f"  True Range%     : {fmt_pct(feats.get('true_range_pct'), 2)}")
            if "day_type" in feats:
                print(f"  Labeled DayType : {feats.get('day_type')}")

        print("")
        print("Plan (3 bullets)")
        for b in bullets:
            print(f"  - {b}")
        print(line)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
