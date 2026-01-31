# scripts/build_features_spy_daily.py
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")  # adjust if your repo uses a different path
SYMBOL = os.getenv("SYMBOL", "SPY")

# Feature knobs (safe defaults)
LOOKBACK_COMPRESSION = int(os.getenv("LOOKBACK_COMPRESSION", "20"))   # window for "low percentile"
LOOKBACK_EXPANSION   = int(os.getenv("LOOKBACK_EXPANSION", "20"))     # window for "high percentile"
COMP_PCTL            = float(os.getenv("COMP_PCTL", "0.20"))          # lowest 20% => compression
EXP_PCTL             = float(os.getenv("EXP_PCTL", "0.80"))           # highest 20% => expansion
VOL_WIN              = int(os.getenv("VOL_WIN", "20"))                # vol estimate window
CLUSTER_WIN          = int(os.getenv("CLUSTER_WIN", "10"))            # clustering persistence window


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
    # Ensure types
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def pct_rank_last(x: pd.Series) -> float:
    """
    Percentile rank of the last element within the window.
    Returns NaN if window too small.
    """
    if x.isna().all():
        return np.nan
    last = x.iloc[-1]
    return float((x <= last).mean())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Previous close
    out["prev_close"] = out["close"].shift(1)

    # Daily return
    out["ret_1d"] = (out["close"] / out["prev_close"]) - 1.0

    # Gap features (open vs prev close)
    out["gap_open_pct"] = (out["open"] / out["prev_close"]) - 1.0
    out["gap_abs_pct"] = out["gap_open_pct"].abs()

    # Range features (normalized by prev close)
    out["intraday_range"] = out["high"] - out["low"]
    out["intraday_range_pct"] = out["intraday_range"] / out["prev_close"]

    # True range (classic)
    hl = out["high"] - out["low"]
    hc = (out["high"] - out["prev_close"]).abs()
    lc = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = np.maximum(hl, np.maximum(hc, lc))
    out["true_range_pct"] = out["true_range"] / out["prev_close"]

    # Volatility estimate
    out["vol20"] = out["ret_1d"].rolling(VOL_WIN, min_periods=max(5, VOL_WIN // 2)).std()

    # "Big day" flag and clustering persistence score
    out["big_day"] = (out["ret_1d"].abs() > out["vol20"]).astype(float)
    out["cluster_score"] = out["big_day"].rolling(CLUSTER_WIN, min_periods=1).sum()

    # Compression flag: percentile rank of today's true_range_pct in last LOOKBACK_COMPRESSION days
    out["tr_pct_rank"] = out["true_range_pct"].rolling(
        LOOKBACK_COMPRESSION, min_periods=max(5, LOOKBACK_COMPRESSION // 2)
    ).apply(pct_rank_last, raw=False)

    out["compression_flag"] = (out["tr_pct_rank"] <= COMP_PCTL).astype(int)

    # Next-day expansion label: look at tomorrow's percentile rank
    out["next_tr_pct"] = out["true_range_pct"].shift(-1)

    out["next_tr_pct_rank"] = out["next_tr_pct"].rolling(
        LOOKBACK_EXPANSION, min_periods=max(5, LOOKBACK_EXPANSION // 2)
    ).apply(pct_rank_last, raw=False)

    out["next_day_expansion"] = (out["next_tr_pct_rank"] >= EXP_PCTL).astype(int)

    # Keep only clean feature columns + identifiers
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
        "compression_flag",
        "next_day_expansion",
    ]
    feats = out[cols].copy()

    # Meta
    feats["feature_ts"] = datetime.now(timezone.utc).isoformat()
    feats["feature_version"] = "v1_gap_cluster_compress_expand"

    # Drop the first row (no prev_close) but keep the last row even if label is NaN (future unknown)
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
        compression_flag INTEGER,
        next_day_expansion INTEGER,
        feature_ts TEXT,
        feature_version TEXT,
        PRIMARY KEY (symbol, date)
    )
    """)
    con.commit()


def upsert_features(con: sqlite3.Connection, feats: pd.DataFrame):
    ensure_features_table(con)

    rows = feats.to_dict(orient="records")
    cols = list(feats.columns)

    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)

    # Update all non-key cols on conflict
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


def main():
    con = connect(DB_PATH)
    try:
        truth = read_truth(con, SYMBOL)
        feats = build_features(truth)
        upsert_features(con, feats)
        write_csv(feats)
        print("OK: features built from truth; truth unchanged.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
