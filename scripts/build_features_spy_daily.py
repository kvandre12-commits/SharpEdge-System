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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    con.commit()


def upsert_features(con: sqlite3.Connection, feats: pd.DataFrame):
    ensure_features_table(con)

    rows = feats.to_dict(orient="records")
    cols = list(feats.columns)

    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)

    update_cols = [c for c in cols if c not in ("symbol", "date")]
    set_clause = ",".join([f"{c}=excluded.{c}" for c in update_cols])

    sql = f"""
    INSERT INTO features_daily ({col_list})
    VALUES ({placeholders})
    ON CONFLICT(symbol, date) DO UPDATE SET
        {set_clause}
    """
    con.executemany(sql, [[r[c] for c in cols] for r in rows])
    con.commit()


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
        "compression_flag": int(last["compression_flag"]),
        "true_range_pct": float(last["true_range_pct"]),
        "intraday_range_pct": float(last["intraday_range_pct"]),
        "gap_open_pct": float(last["gap_open_pct"]),
        "trigger_cluster": int(last["trigger_cluster"]),
        "permission_strength": int(last["permission_strength"]),
        "trade_permission": int(last["trade_permission"]),
        "permission_reason": last["permission_reason"],
        "feature_version": last["feature_version"],
    }])

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/latest_signal.csv"
    signal.to_csv(path, index=False)
    print(f"Wrote {path} (1 row)")

    last = feats.sort_values("date").iloc[-1].copy()

    signal = pd.DataFrame([{
        "date": last["date"],
        "symbol": last.get("symbol", "SPY"),
        "compression": bool(last.get("is_compression", False)),
        "atr_pct": float(last.get("atr_pct", float("nan"))),
        "range_pct": float(last.get("range_pct", float("nan"))),
        "gap_pct": float(last.get("gap_pct", float("nan"))),
        "expansion_bias": (
            "UP" if last.get("expansion_up", False)
            else "DOWN" if last.get("expansion_down", False)
            else "NEUTRAL"
        )
    }])

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/latest_signal.csv"
    signal.to_csv(path, index=False)
    print(f"Wrote {path} (1 row)")


# ---- update main to call it ----
def main():
    con = connect(DB_PATH)
    try:
        truth = read_truth(con, SYMBOL)
        feats = build_features(truth)
        upsert_features(con, feats)
        write_csv(feats)
        write_latest_signal(feats)
        print("OK: features + latest signal built from truth.")
    finally:
        con.close()
if __name__ == "__main__":
