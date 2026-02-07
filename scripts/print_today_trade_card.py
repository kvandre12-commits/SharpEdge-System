#!/usr/bin/env python3
"""
Print Today's Trade Card (Decision Support)

Reads:
  - execution_state_daily (preferred)
  - falls back to intraday_trendday_prob + options_positioning_metrics + features_daily if needed

Outputs:
  - A clean console "trade card" for the latest session_date (or --date)

Usage:
  python scripts/print_today_trade_card.py
  python scripts/print_today_trade_card.py --date 2026-02-06
  SYMBOL=SPY SPY_DB_PATH=data/spy_truth.db python scripts/print_today_trade_card.py
"""

import os
import sys
import sqlite3
import argparse
from datetime import datetime
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

CUTOFF_NY = os.getenv("CUTOFF_HHMM", "11:30")

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return r is not None

def pick_latest_session_date(conn: sqlite3.Connection, symbol: str) -> str:
    # prefer features_daily if present
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

    # fallback to options_positioning_metrics
    if table_exists(conn, "options_positioning_metrics"):
        row = conn.execute(
            "SELECT MAX(session_date) FROM options_positioning_metrics WHERE underlying=?",
            (symbol,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])[:10]

    raise RuntimeError("Could not determine latest session_date (missing features_daily/options_positioning_metrics).")

def load_trade_card_from_execution_state(conn: sqlite3.Connection, symbol: str, session_date: str):
    if not table_exists(conn, "execution_state_daily"):
        return None
    df = pd.read_sql_query(
        """
        SELECT *
        FROM execution_state_daily
        WHERE symbol = ? AND session_date = ?
        LIMIT 1
        """,
        conn, params=(symbol, session_date),
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def load_fallback_card(conn: sqlite3.Connection, symbol: str, session_date: str):
    # Pull: intraday prob fused
    prob = {}
    if table_exists(conn, "intraday_trendday_prob"):
        dfp = pd.read_sql_query(
            """
            SELECT prob_trend, prob_range,
                   prob_trend_fused, prob_range_fused,
                   dealer_state_hint, gamma_proxy, wall_strike, dist_to_wall_pct
            FROM intraday_trendday_prob
            WHERE symbol=? AND session_date=? AND cutoff_ny=?
            LIMIT 1
            """,
            conn, params=(symbol, session_date, CUTOFF_NY),
        )
        if not dfp.empty:
            prob = dfp.iloc[0].to_dict()

    # Pull: dealer metrics (latest snapshot for date)
    dealer = {}
    if table_exists(conn, "options_positioning_metrics"):
        dfd = pd.read_sql_query(
            """
            WITH latest AS (
              SELECT MAX(snapshot_ts) AS snapshot_ts
              FROM options_positioning_metrics
              WHERE underlying=? AND session_date=?
            )
            SELECT dealer_state_hint, gamma_proxy,
                   max_total_oi_strike AS wall_strike,
                   spot,
                   CASE
                     WHEN spot IS NULL OR spot = 0 OR max_total_oi_strike IS NULL THEN NULL
                     ELSE ABS(spot - max_total_oi_strike) / spot * 100.0
                   END AS dist_to_wall_pct
            FROM options_positioning_metrics
            WHERE underlying=? AND session_date=?
              AND snapshot_ts = (SELECT snapshot_ts FROM latest)
            LIMIT 1
            """,
            conn, params=(symbol, session_date, symbol, session_date),
        )
        if not dfd.empty:
            dealer = dfd.iloc[0].to_dict()

    # Pull: daily features
    daily = {}
    if table_exists(conn, "features_daily"):
        cols = pd.read_sql_query("PRAGMA table_info(features_daily);", conn)["name"].tolist()
        date_col = "session_date" if "session_date" in cols else ("date" if "date" in cols else None)
        if date_col:
            dff = pd.read_sql_query(
                f"""
                SELECT
                  {date_col} AS session_date,
                  symbol,
                  cluster_score,
                  compression_flag
                FROM features_daily
                WHERE symbol=? AND {date_col}=?
                LIMIT 1
                """,
                conn, params=(symbol, session_date),
            )
            if not dff.empty:
                daily = dff.iloc[0].to_dict()

    # Combine with precedence: prob overrides dealer if fused already filled those fields
    card = {"session_date": session_date, "symbol": symbol}
    card.update(daily)
    card.update(dealer)
    card.update(prob)

    # If we don't have a bias, make one deterministically
    pt = card.get("prob_trend_fused", None)
    st = (card.get("dealer_state_hint") or "").lower()
    dist = card.get("dist_to_wall_pct", None)
    comp = int(card.get("compression_flag") or 0)

    final_bias = "BALANCED_SMALL"
    if (isinstance(dist, (int, float)) and dist is not None and dist <= 0.25) or st == "pin":
        final_bias = "PIN_FADE"
    elif isinstance(pt, (int, float)) and pt is not None and pt >= 0.65:
        final_bias = "EXPANSION_FOLLOW"
    elif isinstance(pt, (int, float)) and pt is not None and pt <= 0.45:
        final_bias = "WHIP_WAIT"
    else:
        # slight nudge if compression and mid trend prob
        if comp == 1 and isinstance(pt, (int, float)) and pt is not None and pt >= 0.55:
            final_bias = "BALANCED_SMALL (lean expansion)"

    card["final_bias"] = card.get("final_bias", final_bias)

    # basic score if not present
    if "execution_score" not in card or card["execution_score"] is None:
        score = 50.0
        if isinstance(pt, (int, float)) and pt is not None:
            score = max(0.0, min(100.0, float(pt) * 100.0))
        if st == "chase":
            score = min(100.0, score + 10.0)
        elif st == "unwind":
            score = max(0.0, score - 7.0)
        elif st == "pin":
            score = max(0.0, score - 15.0)
        if isinstance(dist, (int, float)) and dist is not None and dist <= 0.25:
            score = max(0.0, score - 10.0)
        card["execution_score"] = round(score, 1)

    return card

def fmt_pct(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "—"

def fmt_num(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def fmt_prob(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"

def print_card(card: dict):
    sd = str(card.get("session_date", "—"))
    sym = str(card.get("symbol", "—"))

    bias = str(card.get("final_bias", "—"))
    score = card.get("execution_score", None)

    pt = card.get("prob_trend_fused", card.get("prob_trend", None))
    pr = card.get("prob_range_fused", card.get("prob_range", None))

    dealer_state = card.get("dealer_state_hint", None)
    gamma_proxy = card.get("gamma_proxy", None)
    wall = card.get("wall_strike", card.get("dealer_wall_strike", None))
    dist = card.get("dist_to_wall_pct", card.get("dealer_dist_to_wall_pct", None))

    comp = card.get("compression_flag", None)
    cluster = card.get("cluster_score", None)

    # Actions
    actions = []
    if "PIN" in bias:
        actions = [
            "Plan: fade extremes back toward the wall/mean.",
            "Execution: smaller size, quicker exits; avoid chasing breakouts.",
            "Invalidation: clean acceptance away from wall + persistent expansion candles."
        ]
    elif "EXPANSION" in bias:
        actions = [
            "Plan: follow-through bias (trend/extension).",
            "Execution: enter on pullback after break; let winners run (don’t scalp yourself).",
            "Invalidation: reclaim of prior range + loss of directional momentum."
        ]
    elif "WHIP" in bias:
        actions = [
            "Plan: stand down until confirmation; trade second move, not first.",
            "Execution: smaller size; wait for break + retest or clear trend structure.",
            "Invalidation: first fakeout—avoid it; confirmation candle required."
        ]
    else:
        actions = [
            "Plan: balanced/low conviction—trade small, be selective.",
            "Execution: prefer A+ setups only; reduce frequency.",
            "Invalidation: if structure shifts to PIN or EXPANSION, upgrade plan."
        ]

    # Pretty print
    line = "═" * 72
    print(line)
    print(f"TRADE CARD  |  {sym}  |  session_date: {sd}  |  cutoff: {CUTOFF_NY} NY")
    print(line)
    print(f"Final Bias        : {bias}")
    print(f"Execution Score   : {fmt_num(score, 1) if score is not None else '—'} / 100")
    print("")
    print("Probabilities")
    print(f"  Trend (fused)   : {fmt_prob(pt)}")
    print(f"  Range (fused)   : {fmt_prob(pr)}")
    print("")
    print("Dealer Positioning")
    print(f"  State Hint      : {dealer_state or '—'}")
    print(f"  Gamma Proxy     : {fmt_num(gamma_proxy, 2)}")
    print(f"  Wall Strike     : {fmt_num(wall, 2)}")
    print(f"  Dist → Wall     : {fmt_pct(dist, 2)} (<= 0.25% = pin-risk)")
    print("")
    print("Volatility / Structure")
    print(f"  Compression     : {comp if comp is not None else '—'} (1=yes)")
    print(f"  Cluster Score   : {fmt_num(cluster, 3)}")
    print("")
    print("Plan (3 bullets)")
    for a in actions:
        print(f"  - {a}")
    print(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", dest="date", default="", help="YYYY-MM-DD (optional)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        session_date = args.date.strip() or pick_latest_session_date(conn, SYMBOL)

        card = load_trade_card_from_execution_state(conn, SYMBOL, session_date)
        if card is None:
            card = load_fallback_card(conn, SYMBOL, session_date)

        print_card(card)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
