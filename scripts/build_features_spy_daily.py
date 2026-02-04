#!/usr/bin/env python3
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# Intraday ORB / edge volatility knobs (15m bars)
INTRADAY_BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
ORB_BARS = int(os.getenv("ORB_BARS", "4"))  # 4x15m = 1 hour
EDGE_SHARE_FLAG = float(os.getenv("EDGE_SHARE_FLAG", "0.60"))

# Trend-day knobs
TREND_RANGE_PCT_MIN = float(os.getenv("TREND_RANGE_PCT_MIN", "0.012"))
TREND_CLOSE_POS_UP = float(os.getenv("TREND_CLOSE_POS_UP", "0.75"))
TREND_CLOSE_POS_DN = float(os.getenv("TREND_CLOSE_POS_DN", "0.25"))
TREND_MAX_WICK_PCT = float(os.getenv("TREND_MAX_WICK_PCT", "0.25"))
TREND_EDGE_SHARE_MAX = float(os.getenv("TREND_EDGE_SHARE_MAX", "0.60"))

FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v3_edge_orb_trend")

OUT_CSV = os.getenv("FEATURES_OUT_CSV", "outputs/spy_features_daily.csv")


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide with 0/NaN protection."""
    b2 = b.replace(0, np.nan)
    return (a / b2).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def read_truth_daily(con: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT
      date,
      symbol,
      open, high, low, close,
      volume
    FROM bars_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    if df.empty:
        raise RuntimeError("bars_daily returned 0 rows. Did you ingest daily first?")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def compute_orb_features(
    con: sqlite3.Connection,
    date_min: str,
    date_max: str,
) -> pd.DataFrame:
    """
    Computes open/close ORB ranges from intraday 15m bars.

    Expects intraday table columns:
      session_date, ts, symbol, high, low
    """
    q = f"""
    SELECT session_date, ts, high, low
    FROM {INTRADAY_BARS_TABLE}
    WHERE symbol = ?
      AND session_date BETWEEN ? AND ?
    ORDER BY session_date ASC, ts ASC
    """
    try:
        intr = pd.read_sql_query(q, con, params=(SYMBOL, date_min, date_max))
    except Exception:
        # Table missing or schema mismatch; return empty and features will default to 0
        return pd.DataFrame(columns=["date", "open_orb_range", "close_orb_range"])

    if intr.empty:
        return pd.DataFrame(columns=["date", "open_orb_range", "close_orb_range"])

    intr["ts"] = pd.to_datetime(intr["ts"], errors="coerce")
    intr = intr.dropna(subset=["ts"]).sort_values(["session_date", "ts"])

    rows = []
    for d, g in intr.groupby("session_date", sort=True):
        if len(g) < 2 * ORB_BARS:
            continue

        open_g = g.head(ORB_BARS)
        close_g = g.tail(ORB_BARS)

        open_range = float(open_g["high"].max() - open_g["low"].min())
        close_range = float(close_g["high"].max() - close_g["low"].min())

        rows.append({
            "date": str(d),
            "open_orb_range": open_range,
            "close_orb_range": close_range,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Feature build
# -----------------------------
def build_features(truth: pd.DataFrame, con: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    out = truth.copy()

    # Core returns / range
    out["prev_close"] = out["close"].shift(1)
    out["ret_1d"] = safe_div(out["close"] - out["prev_close"], out["prev_close"])

    out["intraday_range"] = out["high"] - out["low"]
    out["intraday_range_pct"] = safe_div(out["intraday_range"], out["close"])

    out["gap_open"] = out["open"] - out["prev_close"]
    out["gap_open_pct"] = safe_div(out["gap_open"], out["prev_close"])
    out["gap_abs_pct"] = out["gap_open_pct"].abs()

    # True range (daily)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - out["prev_close"]).abs()
    tr3 = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["true_range_pct"] = safe_div(out["true_range"], out["close"])

    # Vol proxy
    out["vol20"] = out["ret_1d"].rolling(20, min_periods=10).std().fillna(0.0)

    # -----------------------------
    # Edge volatility (open/close ORB)
    # -----------------------------
    out["open_orb_range"] = 0.0
    out["close_orb_range"] = 0.0
    out["open_orb_share"] = 0.0
    out["close_orb_share"] = 0.0
    out["edge_orb_share"] = 0.0
    out["edge_orb_bias"] = 0.0
    out["edge_orb_flag"] = 0

    if con is not None and not out.empty:
        date_min = str(out["date"].min())
        date_max = str(out["date"].max())
        orb = compute_orb_features(con, date_min=date_min, date_max=date_max)

        if not orb.empty:
            out = out.merge(orb, on="date", how="left", suffixes=("", "_orb"))
            if "open_orb_range" not in out.columns:  out["open_orb_range"] = 0.0
            if "close_orb_range" not in out.columns: out["close_orb_range"] = 0.0
            out["open_orb_range"] = out["open_orb_range"].fillna(0.0)
            out["close_orb_range"] = out["close_orb_range"].fillna(0.0)
            out["open_orb_share"] = safe_div(out["open_orb_range"], out["intraday_range"])
            out["close_orb_share"] = safe_div(out["close_orb_range"], out["intraday_range"])
            out["edge_orb_share"] = safe_div(out["open_orb_range"] + out["close_orb_range"], out["intraday_range"])
            out["edge_orb_bias"] = (out["close_orb_share"] - out["open_orb_share"]).fillna(0.0)
            out["edge_orb_flag"] = (out["edge_orb_share"] >= EDGE_SHARE_FLAG).astype(int)

    # -----------------------------
    # Trend vs non-trend day (structure only)
    # -----------------------------
    out["range_pct"] = safe_div(out["intraday_range"], out["close"])
    out["close_pos"] = safe_div(out["close"] - out["low"], out["intraday_range"])

    out["upper_wick_pct"] = safe_div(out["high"] - out[["open", "close"]].max(axis=1), out["intraday_range"])
    out["lower_wick_pct"] = safe_div(out[["open", "close"]].min(axis=1) - out["low"], out["intraday_range"])

    def classify_day(r):
        # trend up
        if (
            r["range_pct"] >= TREND_RANGE_PCT_MIN
            and r["close_pos"] >= TREND_CLOSE_POS_UP
            and r["upper_wick_pct"] <= TREND_MAX_WICK_PCT
            and r["edge_orb_share"] <= TREND_EDGE_SHARE_MAX
        ):
            return "trend_up"

        # trend down
        if (
            r["range_pct"] >= TREND_RANGE_PCT_MIN
            and r["close_pos"] <= TREND_CLOSE_POS_DN
            and r["lower_wick_pct"] <= TREND_MAX_WICK_PCT
            and r["edge_orb_share"] <= TREND_EDGE_SHARE_MAX
        ):
            return "trend_down"

        return "non_trend"

    out["day_type"] = out.apply(classify_day, axis=1)

    # -----------------------------
    # Your existing trigger / permission logic lives here.
    # I’m leaving placeholders to avoid breaking your pipeline.
    # If your original file has these columns, keep your original logic.
    # -----------------------------
    # If your prior script already sets these, keep them. Otherwise default safely.
    for c, default in [
        ("cluster_score", 0.0),
        ("trigger_gap_15", 0),
        ("trigger_range_15", 0),
        ("trigger_tr_15", 0),
        ("trigger_any_15", 0),
        ("trigger_cluster", 0),
        ("compression_flag", 0),
        ("next_day_expansion", 0),
        ("permission_strength", 0.0),
        ("permission_reason", ""),
        ("trade_permission", 0),
    ]:
        if c not in out.columns:
            out[c] = default

    # Feature version
    out["feature_version"] = FEATURE_VERSION

    # -----------------------------
    # Publish (THIS IS THE IMPORTANT PART)
    # -----------------------------
    cols = [
        "date", "symbol",
        "open", "high", "low", "close", "volume",
        "prev_close",
        "ret_1d",
        "gap_open_pct", "gap_abs_pct",
        "intraday_range_pct",
        "true_range_pct",
        "vol20",
        "cluster_score",

        "trigger_gap_15",
        "trigger_range_15",
        "trigger_tr_15",
        "trigger_any_15",
        "trigger_cluster",

        "compression_flag",
        "next_day_expansion",

        "permission_strength",
        "permission_reason",
        "trade_permission",

        # NEW: ORB / edge structure
        "open_orb_range",
        "close_orb_range",
        "open_orb_share",
        "close_orb_share",
        "edge_orb_share",
        "edge_orb_bias",
        "edge_orb_flag",

        # NEW: day structure
        "day_type",

        "feature_version",
    ]

    feats = out[cols].copy()
    return feats


def write_features_csv(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"OK: wrote {OUT_CSV} ({len(df):,} rows)")


def main():
    with sqlite3.connect(DB_PATH) as con:
        truth = read_truth_daily(con)
        feats = build_features(truth, con=con)  # ✅ PASS con so ORB runs
        write_features_csv(feats)


if __name__ == "__main__":
    main()
