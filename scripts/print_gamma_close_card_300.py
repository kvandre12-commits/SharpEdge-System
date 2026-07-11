#!/usr/bin/env python3
"""3:00pm Gamma Close Card (Decision Support)."""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
from typing import Any

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
PIN_DIST_PCT = float(os.getenv("PIN_DIST_PCT", "0.25"))


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


def column_names(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [
        row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    ]


def fetch_one_dict(
    conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
) -> dict[str, Any] | None:
    cur = conn.execute(query, params)
    row = cur.fetchone()
    if row is None:
        return None
    columns = [desc[0] for desc in cur.description]
    return dict(zip(columns, row, strict=False))


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "—"
    return f"{numeric:.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "—"
    return f"{numeric:.{digits}f}%"


def pick_latest_session_date(conn: sqlite3.Connection, symbol: str) -> str:
    if table_exists(conn, "options_positioning_metrics"):
        row = conn.execute(
            "SELECT MAX(session_date) FROM options_positioning_metrics WHERE underlying=?",
            (symbol,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])[:10]

    if table_exists(conn, "features_daily"):
        cols = column_names(conn, "features_daily")
        date_col = (
            "session_date"
            if "session_date" in cols
            else ("date" if "date" in cols else None)
        )
        if date_col:
            row = conn.execute(
                f"SELECT MAX({date_col}) FROM features_daily WHERE symbol=?",
                (symbol,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])[:10]

    raise RuntimeError("Could not determine latest session_date.")


def load_latest_metrics_for_date(
    conn: sqlite3.Connection, symbol: str, session_date: str
) -> dict[str, Any]:
    if not table_exists(conn, "options_positioning_metrics"):
        raise RuntimeError(
            "Missing options_positioning_metrics. Run your aggregation first."
        )

    metrics = fetch_one_dict(
        conn,
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
        (symbol, session_date, symbol, session_date),
    )
    if metrics is None:
        raise RuntimeError(
            f"No options_positioning_metrics rows for {symbol} on {session_date}."
        )

    spot = to_float(metrics.get("spot"))
    wall = to_float(metrics.get("wall_strike"))
    if spot not in (None, 0.0) and wall is not None:
        metrics["dist_to_wall_pct"] = abs(spot - wall) / spot * 100.0
    else:
        metrics["dist_to_wall_pct"] = None
    return metrics


def load_optional_overlay(
    conn: sqlite3.Connection, symbol: str, session_date: str
) -> dict[str, Any]:
    if not table_exists(conn, "overlays_daily"):
        return {}

    wanted = [
        "dealer_pin_score",
        "dealer_expand_score",
        "dealer_whip_score",
        "dealer_late_day_mode",
        "dealer_wall_strike",
        "dealer_dist_to_wall_pct",
    ]
    present = [name for name in wanted if name in column_names(conn, "overlays_daily")]
    if not present:
        return {}

    overlay = fetch_one_dict(
        conn,
        f"""
        SELECT {", ".join(present)}
        FROM overlays_daily
        WHERE symbol=? AND session_date=?
        LIMIT 1
        """,
        (symbol, session_date),
    )
    return overlay or {}


def load_optional_features(
    conn: sqlite3.Connection, symbol: str, session_date: str
) -> dict[str, Any]:
    if not table_exists(conn, "features_daily"):
        return {}

    cols = column_names(conn, "features_daily")
    date_col = (
        "session_date"
        if "session_date" in cols
        else ("date" if "date" in cols else None)
    )
    if not date_col:
        return {}

    keep = [
        name
        for name in [
            "cluster_score",
            "compression_flag",
            "intraday_range_pct",
            "true_range_pct",
            "day_type",
        ]
        if name in cols
    ]
    if not keep:
        return {}

    features = fetch_one_dict(
        conn,
        f"""
        SELECT {", ".join(keep)}
        FROM features_daily
        WHERE symbol=? AND {date_col}=?
        LIMIT 1
        """,
        (symbol, session_date),
    )
    return features or {}


def decide_plan(
    metrics: dict[str, Any], overlay: dict[str, Any]
) -> tuple[str, float, list[str]]:
    mode = str(overlay.get("dealer_late_day_mode") or "").upper()
    pin_s = to_float(overlay.get("dealer_pin_score"))
    exp_s = to_float(overlay.get("dealer_expand_score"))
    whp_s = to_float(overlay.get("dealer_whip_score"))

    dist = to_float(metrics.get("dist_to_wall_pct"))
    state_hint = str(metrics.get("dealer_state_hint") or "").lower()

    if mode in {"PIN", "EXPAND", "WHIP"}:
        bias = {
            "PIN": "PIN_FADE",
            "EXPAND": "EXPANSION_FOLLOW",
            "WHIP": "WHIP_WAIT",
        }[mode]
        scores = [score for score in [pin_s, exp_s, whp_s] if score is not None]
        conf = max(0.0, min(max(scores) if scores else 50.0, 100.0))
    else:
        conf = 55.0
        if dist is not None and dist <= PIN_DIST_PCT:
            bias = "PIN_FADE"
            conf = 72.0
        elif state_hint == "chase":
            bias = "EXPANSION_FOLLOW"
            conf = 70.0
        elif state_hint == "unwind":
            bias = "PIN_FADE"
            conf = 62.0
        else:
            bias = "WHIP_WAIT"

    if bias == "PIN_FADE":
        bullets = [
            "Core idea: dealer hedging tends to pull price back toward the wall/ATM late-day.",
            "Execution: fade edges → target wall; take profits quickly; avoid chasing breakouts.",
            "Trigger: if price holds away from wall with expanding candles/volume, stop fading.",
        ]
    elif bias == "EXPANSION_FOLLOW":
        bullets = [
            "Core idea: negative gamma / chase behavior can create late-day acceleration.",
            "Execution: join the move on pullback after break; hold runner into close.",
            "Trigger: if move stalls and snaps back toward wall, reduce and protect.",
        ]
    else:
        bullets = [
            "Core idea: late-day fakeouts are common—trade the second move, not the first.",
            "Execution: wait for break + retest / confirmation candle; smaller size.",
            "Trigger: if price pins tightly to wall, switch to PIN_FADE.",
        ]

    return bias, conf, bullets


def print_card(
    session_date: str,
    metrics: dict[str, Any],
    overlay: dict[str, Any],
    features: dict[str, Any],
    bias: str,
    conf: float,
    bullets: list[str],
) -> None:
    line = "═" * 72
    print(line)
    print(
        "GAMMA CLOSE CARD "
        f"| {SYMBOL} | session_date: {session_date} | snapshot_ts: {metrics.get('snapshot_ts')}"
    )
    print(line)
    print(f"Final Bias        : {bias}")
    print(f"Confidence        : {fmt_num(conf, 1)} / 100")
    print("")
    print("Key Dealer Metrics")
    print(f"  Dealer State    : {metrics.get('dealer_state_hint') or '—'}")
    print(f"  Gamma Proxy     : {fmt_num(metrics.get('gamma_proxy'), 2)}")
    print(f"  Spot (proxy)    : {fmt_num(metrics.get('spot'), 2)}")
    print(f"  Wall (max OI)   : {fmt_num(metrics.get('wall_strike'), 2)}")
    print(
        f"  Dist → Wall     : {fmt_pct(metrics.get('dist_to_wall_pct'), 2)}  "
        f"(<= {PIN_DIST_PCT:.2f}% = pin-risk)"
    )
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
            print(
                f"  Expand Score    : {fmt_num(overlay.get('dealer_expand_score'), 1)}"
            )
        if "dealer_whip_score" in overlay:
            print(f"  Whip Score      : {fmt_num(overlay.get('dealer_whip_score'), 1)}")
        if "dealer_wall_strike" in overlay:
            print(
                f"  Overlay Wall    : {fmt_num(overlay.get('dealer_wall_strike'), 2)}"
            )
        if "dealer_dist_to_wall_pct" in overlay:
            print(
                "  Overlay Dist%   : "
                f"{fmt_pct(overlay.get('dealer_dist_to_wall_pct'), 2)}"
            )

    if features:
        print("")
        print("Structure (optional)")
        if "compression_flag" in features:
            print(f"  Compression     : {features.get('compression_flag')} (1=yes)")
        if "cluster_score" in features:
            print(f"  Cluster Score   : {fmt_num(features.get('cluster_score'), 3)}")
        if "intraday_range_pct" in features:
            print(
                f"  Intraday Range% : {fmt_pct(features.get('intraday_range_pct'), 2)}"
            )
        if "true_range_pct" in features:
            print(f"  True Range%     : {fmt_pct(features.get('true_range_pct'), 2)}")
        if "day_type" in features:
            print(f"  Labeled DayType : {features.get('day_type')}")

    print("")
    print("Plan (3 bullets)")
    for bullet in bullets:
        print(f"  - {bullet}")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="YYYY-MM-DD (optional)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        session_date = args.date.strip() or pick_latest_session_date(conn, SYMBOL)
        metrics = load_latest_metrics_for_date(conn, SYMBOL, session_date)
        overlay = load_optional_overlay(conn, SYMBOL, session_date)
        features = load_optional_features(conn, SYMBOL, session_date)
        bias, conf, bullets = decide_plan(metrics, overlay)
        print_card(session_date, metrics, overlay, features, bias, conf, bullets)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
