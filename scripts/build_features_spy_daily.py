# scripts/build_features_spy_daily.py
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# Windows / knobs
VOL_WIN = int(os.getenv("VOL_WIN", "20"))
CLUSTER_WIN = int(os.getenv("CLUSTER_WIN", "10"))

LOOKBACK_COMPRESSION = int(os.getenv("LOOKBACK_COMPRESSION", "20"))
LOOKBACK_EXPANSION = int(os.getenv("LOOKBACK_EXPANSION", "20"))
COMP_PCTL = float(os.getenv("COMP_PCTL", "0.20"))  # lowest 20% => compression
EXP_PCTL = float(os.getenv("EXP_PCTL", "0.80"))    # highest 20% => expansion

# Trigger threshold (1.5% default)
TRIGGER_PCT = float(os.getenv("TRIGGER_PCT", "0.015"))

# Permission knobs
PERM_MIN_TRIGGERS_IN_WINDOW = int(os.getenv("PERM_MIN_TRIGGERS_IN_WINDOW", "2"))

# Intraday ORB / edge volatility knobs (15m bars)
INTRADAY_BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
ORB_BARS = int(os.getenv("ORB_BARS", "4"))  # 4 x 15m = 1 hour open + 1 hour close
EDGE_SHARE_FLAG = float(os.getenv("EDGE_SHARE_FLAG", "0.60"))

def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def read_truth(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    q = """
    SELECT
        date, symbol, open, high, low, close, volume, source, ingest_ts
    FROM bars_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(q, con, params=(symbol,))
    if df.empty:
        raise RuntimeError(f"No rows found in bars_daily for symbol={symbol}")

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def pct_rank_last(x: pd.Series) -> float:
    if x.isna().all():
        return np.nan
    last = x.iloc[-1]
    return float((x <= last).mean())

def compute_orb_features(
    con: sqlite3.Connection,
    symbol: str,
    intraday_table: str,
    orb_bars: int,
    date_min: str,
    date_max: str,
) -> pd.DataFrame:
    """
    Computes open/close ORB ranges using intraday 15m bars table (spy_bars_15m).
    Expects columns: session_date, ts, high, low, symbol
    Returns: date, open_orb_range, close_orb_range
    """
    q = f"""
    SELECT session_date, ts, high, low
    FROM {intraday_table}
    WHERE symbol = ?
      AND session_date BETWEEN ? AND ?
    ORDER BY session_date ASC, ts ASC
    """
    try:
        intr = pd.read_sql_query(q, con, params=(symbol, date_min, date_max))
    except Exception:
        # table missing or schema mismatch
        return pd.DataFrame(columns=["date", "open_orb_range", "close_orb_range"])

    if intr.empty:
        return pd.DataFrame(columns=["date", "open_orb_range", "close_orb_range"])

    # Ensure correct ordering
    intr["ts"] = pd.to_datetime(intr["ts"], errors="coerce")
    intr = intr.dropna(subset=["ts"]).sort_values(["session_date", "ts"])

    rows = []
    for d, g in intr.groupby("session_date", sort=True):
        if len(g) < 2 * orb_bars:
            continue

        open_g = g.head(orb_bars)
        close_g = g.tail(orb_bars)

        open_range = float(open_g["high"].max() - open_g["low"].min())
        close_range = float(close_g["high"].max() - close_g["low"].min())

        rows.append({
            "date": str(d),
            "open_orb_range": open_range,
            "close_orb_range": close_range,
        })

    return pd.DataFrame(rows)

def build_features(df: pd.DataFrame, con: sqlite3.Connection | None = None) -> pd.DataFrame:
    out = df.copy()

    # Prev close + returns
    out["prev_close"] = out["close"].shift(1)
    out["ret_1d"] = (out["close"] / out["prev_close"]) - 1.0

    # Gap
    out["gap_open_pct"] = (out["open"] / out["prev_close"]) - 1.0
    out["gap_abs_pct"] = out["gap_open_pct"].abs()

    # Range
    out["intraday_range"] = out["high"] - out["low"]
    out["intraday_range_pct"] = out["intraday_range"] / out["prev_close"]

    # True range
    hl = out["high"] - out["low"]
    hc = (out["high"] - out["prev_close"]).abs()
    lc = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = np.maximum(hl, np.maximum(hc, lc))
    out["true_range_pct"] = out["true_range"] / out["prev_close"]

    # Vol
    out["vol20"] = out["ret_1d"].rolling(VOL_WIN, min_periods=max(5, VOL_WIN // 2)).std()

    # Old cluster score (kept)
    out["big_day"] = (out["ret_1d"].abs() > out["vol20"]).astype(float)
    out["cluster_score"] = out["big_day"].rolling(CLUSTER_WIN, min_periods=1).sum()

    # -----------------------------
    # Range expansion triggers
    # -----------------------------
    out["trigger_gap_15"] = (out["gap_abs_pct"] >= TRIGGER_PCT).astype(int)
    out["trigger_range_15"] = (out["intraday_range_pct"].abs() >= TRIGGER_PCT).astype(int)
    out["trigger_tr_15"] = (out["true_range_pct"].abs() >= TRIGGER_PCT).astype(int)

    out["trigger_any_15"] = (
        (out["trigger_gap_15"] == 1)
        | (out["trigger_range_15"] == 1)
        | (out["trigger_tr_15"] == 1)
    ).astype(int)

    out["trigger_cluster"] = out["trigger_any_15"].rolling(CLUSTER_WIN, min_periods=1).sum()
    
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

        orb = compute_orb_features(
            con=con,
            symbol=SYMBOL,
            intraday_table=INTRADAY_BARS_TABLE,
            orb_bars=ORB_BARS,
            date_min=date_min,
            date_max=date_max,
        )

        if not orb.empty:
            out = out.merge(orb, on="date", how="left")
            out["open_orb_range"] = out["open_orb_range"].fillna(0.0)
            out["close_orb_range"] = out["close_orb_range"].fillna(0.0)

            denom = out["intraday_range"].replace(0, np.nan)
            out["open_orb_share"] = (out["open_orb_range"] / denom).fillna(0.0)
            out["close_orb_share"] = (out["close_orb_range"] / denom).fillna(0.0)
            out["edge_orb_share"] = ((out["open_orb_range"] + out["close_orb_range"]) / denom).fillna(0.0)
            out["edge_orb_bias"] = (out["close_orb_share"] - out["open_orb_share"]).fillna(0.0)
            out["edge_orb_flag"] = (out["edge_orb_share"] >= EDGE_SHARE_FLAG).astype(int)
    # -----------------------------
    # Compression + label (kept)
    # -----------------------------
    out["tr_pct_rank"] = out["true_range_pct"].rolling(
        LOOKBACK_COMPRESSION, min_periods=max(5, LOOKBACK_COMPRESSION // 2)
    ).apply(pct_rank_last, raw=False)
    out["compression_flag"] = (out["tr_pct_rank"] <= COMP_PCTL).astype(int)

    out["next_tr_pct"] = out["true_range_pct"].shift(-1)
    out["next_tr_pct_rank"] = out["next_tr_pct"].rolling(
        LOOKBACK_EXPANSION, min_periods=max(5, LOOKBACK_EXPANSION // 2)
    ).apply(pct_rank_last, raw=False)
    out["next_day_expansion"] = (out["next_tr_pct_rank"] >= EXP_PCTL).astype(int)

    # -----------------------------
    # Permission strength + B gate
    # Strength:
    #  +1 cluster regime (>= N triggers in window)
    #  +1 compression setup (compression_flag=1)
    #  +1 gap trigger (news kick)
    # Gate (B): cluster & compression
    # Plus override: gap & cluster
    # -----------------------------
    has_cluster = (out["trigger_cluster"] >= PERM_MIN_TRIGGERS_IN_WINDOW)
    has_compression = (out["compression_flag"] == 1)
    has_gap = (out["trigger_gap_15"] == 1)

    out["permission_strength"] = (
        has_cluster.astype(int)
        + has_compression.astype(int)
        + has_gap.astype(int)
    )

    out["trade_permission"] = (has_cluster & has_compression).astype(int)
    out["trade_permission"] = (
        (out["trade_permission"] == 1) | (has_gap & has_cluster)
    ).astype(int)

    out["permission_reason"] = np.select(
        [
            (out["permission_strength"] == 3),
            (out["permission_strength"] == 2),
            (out["permission_strength"] == 1),
        ],
        [
            "cluster+compression+gap",
            "cluster+compression",
            "cluster_only",
        ],
        default="none",
    )

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
    ]
    feats = out[cols].copy()

    feats["feature_ts"] = datetime.now(timezone.utc).isoformat()
    feats["feature_version"] = "v2_triggers_permission_strength"

    feats = feats.dropna(subset=["prev_close"]).reset_index(drop=True)

    # --- FINRA ATS pressure overlay (weekly, forward-filled) ---
    finra_path = "outputs/spy_finra_ats_weekly.csv"

    if os.path.exists(finra_path):
        finra = pd.read_csv(finra_path)

        # normalize FINRA date column
        if "date" not in finra.columns:
            for c in ["week", "week_start", "report_date", "trade_date"]:
                if c in finra.columns:
                    finra["date"] = finra[c]
                    break

        if "date" in finra.columns:
            finra["date"] = pd.to_datetime(finra["date"]).dt.date.astype(str)
            finra = finra.sort_values("date")

            # ensure ats_ratio always exists
            if "ats_ratio" not in finra.columns:
                finra["ats_ratio"] = np.nan

            feats = feats.merge(
                finra[["date", "ats_ratio"]],
                on="date",
                how="left",
            )

            feats["ats_ratio"] = feats["ats_ratio"].ffill()

            feats["ats_pressure_flag"] = (
                feats["ats_ratio"]
                >= feats["ats_ratio"]
                .rolling(20, min_periods=5)
                .quantile(0.80)
            ).astype(int)
        else:
            feats["ats_ratio"] = np.nan
            feats["ats_pressure_flag"] = 0
    else:
        feats["ats_ratio"] = np.nan
        feats["ats_pressure_flag"] = 0
    # -----------------------------
    # Trend day vs non-trend day
    # Uses: range size + close position + wick rejection + edge distribution
    # -----------------------------
    RANGE_PCT_MIN = float(os.getenv("TREND_RANGE_PCT_MIN", "0.012"))
    CLOSE_STRONG_UP = float(os.getenv("TREND_CLOSE_POS_UP", "0.75"))
    CLOSE_STRONG_DOWN = float(os.getenv("TREND_CLOSE_POS_DN", "0.25"))
    MAX_WICK_PCT = float(os.getenv("TREND_MAX_WICK_PCT", "0.25"))
    EDGE_SHARE_MAX = float(os.getenv("TREND_EDGE_SHARE_MAX", "0.60"))

    denom = out["intraday_range"].replace(0, np.nan)

    out["range_pct"] = (out["intraday_range"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["close_pos"] = ((out["close"] - out["low"]) / denom).fillna(0.0)

    out["upper_wick_pct"] = ((out["high"] - out[["open", "close"]].max(axis=1)) / denom).fillna(0.0)
    out["lower_wick_pct"] = ((out[["open", "close"]].min(axis=1) - out["low"]) / denom).fillna(0.0)

    def classify_day(r):
        # Trend up
        if (
            r["range_pct"] >= RANGE_PCT_MIN
            and r["close_pos"] >= CLOSE_STRONG_UP
            and r["upper_wick_pct"] <= MAX_WICK_PCT
            and r["edge_orb_share"] <= EDGE_SHARE_MAX
        ):
            return "trend_up"

        # Trend down
        if (
            r["range_pct"] >= RANGE_PCT_MIN
            and r["close_pos"] <= CLOSE_STRONG_DOWN
            and r["lower_wick_pct"] <= MAX_WICK_PCT
            and r["edge_orb_share"] <= EDGE_SHARE_MAX
        ):
            return "trend_down"

        return "non_trend"

    out["day_type"] = out.apply(classify_day, axis=1)
    return feats


def ensure_features_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS features_daily (
        date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        prev_close REAL,
        ret_1d REAL,
        gap_open_pct REAL,
        gap_abs_pct REAL,
        intraday_range_pct REAL,
        true_range_pct REAL,
        vol20 REAL,
        cluster_score REAL,

        trigger_gap_15 INTEGER,
        trigger_range_15 INTEGER,
        trigger_tr_15 INTEGER,
        trigger_any_15 INTEGER,
        trigger_cluster REAL,

        compression_flag INTEGER,
        next_day_expansion INTEGER,

        permission_strength INTEGER,
        permission_reason TEXT,
        trade_permission INTEGER,

        feature_ts TEXT,
        feature_version TEXT,
        PRIMARY KEY (symbol, date)
    )
    """)
    con.commit()

    # Forward adds (safe)
    existing = {r[1] for r in con.execute("PRAGMA table_info(features_daily)")}
    adds = {
        "trigger_gap_15": "INTEGER",
        "trigger_range_15": "INTEGER",
        "trigger_tr_15": "INTEGER",
        "trigger_any_15": "INTEGER",
        "trigger_cluster": "REAL",
        "permission_strength": "INTEGER",
        "permission_reason": "TEXT",
        "trade_permission": "INTEGER",
    }
    for col, typ in adds.items():
        if col not in existing:
            con.execute(f"ALTER TABLE features_daily ADD COLUMN {col} {typ}")
    
def write_csv(feats: pd.DataFrame):
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/spy_features_daily.csv"
    feats.to_csv(path, index=False)
    print(f"Wrote {path} ({len(feats)} rows)")


def write_latest_signal(feats: pd.DataFrame):
    if feats.empty:
        print("No features available; skipping latest signal.")
        return

    last = feats.sort_values("date").iloc[-1]

    signal = pd.DataFrame([{
        "date": last["date"],
        "symbol": last["symbol"],
        "close": float(last["close"]),
        "gap_open_pct": float(last["gap_open_pct"]),
        "true_range_pct": float(last["true_range_pct"]),
        "intraday_range_pct": float(last["intraday_range_pct"]),
        "compression_flag": int(last["compression_flag"]),
        "trigger_cluster": int(last["trigger_cluster"]),
        "permission_strength": int(last["permission_strength"]),
        "trade_permission": int(last["trade_permission"]),
        "permission_reason": last["permission_reason"],
        "feature_version": last["feature_version"],
    }])

    os.makedirs("outputs", exist_ok=True)
    signal.to_csv("outputs/latest_signal.csv", index=False)
    print("Wrote outputs/latest_signal.csv (1 row)")

def main():
    con = connect(DB_PATH)
    try:
        truth = read_truth(con, SYMBOL)
        feats = build_features(truth)
        write_csv(feats)
        write_latest_signal(feats)
        print("OK: features + latest signal built from truth.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
