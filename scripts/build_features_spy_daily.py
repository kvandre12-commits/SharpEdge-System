import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))

# --- knobs (keep small + sane) ---
VOL_WIN = int(os.getenv("FEAT_VOL_WIN", "20"))
CLUSTER_WIN = int(os.getenv("FEAT_CLUSTER_WIN", "10"))
TRIGGER_PCT = float(os.getenv("FEAT_TRIGGER_PCT", "0.015"))  # 1.5%
LOOKBACK_COMPRESSION = int(os.getenv("FEAT_LOOKBACK_COMPRESSION", "60"))
LOOKBACK_EXPANSION = int(os.getenv("FEAT_LOOKBACK_EXPANSION", "60"))
COMP_PCTL = float(os.getenv("FEAT_COMP_PCTL", "0.20"))
EXP_PCTL = float(os.getenv("FEAT_EXP_PCTL", "0.80"))

# ORB settings (15m bars):
ORB_BARS = int(os.getenv("FEAT_ORB_BARS", "4"))          # first 60m if 15m bars
CLOSE_BARS = int(os.getenv("FEAT_CLOSE_BARS", "4"))      # last 60m if 15m bars
EDGE_SHARE_FLAG = float(os.getenv("EDGE_SHARE_FLAG", "0.60"))


def connect(db_path: str) -> sqlite3.Connection:
    dbp = Path(db_path)
    if dbp.parent:
        dbp.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(dbp))
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide with 0/NaN protection."""
    b2 = b.replace(0, np.nan)
    return (a / b2).replace([np.inf, -np.inf], np.nan)


def pct_rank_last(window: pd.Series) -> float:
    """Percentile rank of the last element within the window (0..1)."""
    w = window.dropna()
    if len(w) < 3:
        return np.nan
    last = w.iloc[-1]
    return float((w <= last).mean())


def read_daily_truth(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """
    Tries common daily tables. Expected columns: date, symbol, open, high, low, close, volume
    """
    candidates = [
        "truth_daily",
        "ohlc_daily",
        "bars_daily",
        "spy_daily",
    ]
    table = next((t for t in candidates if table_exists(con, t)), None)
    if table is None:
        raise RuntimeError(f"No daily truth table found. Tried: {candidates}")

    df = pd.read_sql(
        f"""
        SELECT date, symbol, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ?
        ORDER BY date ASC
        """,
        con,
        params=(symbol,),
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def read_intraday_15m(con: sqlite3.Connection, symbol: str) -> Optional[pd.DataFrame]:
    """
    Tries common 15m tables. Expected columns: ts (or datetime), symbol, open, high, low, close, volume
    """
    candidates = [
        "truth_intraday_15m",
        "bars_intraday_15m",
        "bars_15m",
        "spy_intraday_15m",
    ]
    table = next((t for t in candidates if table_exists(con, t)), None)
    if table is None:
        return None

    df = pd.read_sql(
        f"""
        SELECT *
        FROM {table}
        WHERE symbol = ?
        ORDER BY 1 ASC
        """,
        con,
        params=(symbol,),
    )

    # Normalize timestamp column name
    ts_col = None
    for c in ["ts", "timestamp", "datetime", "time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        return None

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    df["session_date"] = df[ts_col].dt.tz_convert(None).dt.date.astype(str)
    df.rename(columns={ts_col: "ts"}, inplace=True)

    # Ensure required columns exist
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    return df


def compute_orb_features(intra: pd.DataFrame) -> pd.DataFrame:
    """
    For each session_date:
      - open_orb_range  = (max(high first N) - min(low first N)) / close(last bar)
      - close_orb_range = (max(high last M)  - min(low last M))  / close(last bar)
      - shares: open_orb_share, close_orb_share, edge_orb_share, bias
    """
    g = intra.groupby("session_date", sort=True)

    rows = []
    for session_date, d in g:
        d = d.sort_values("ts").reset_index(drop=True)

        # Skip tiny sessions
        if len(d) < max(ORB_BARS, CLOSE_BARS) + 1:
            continue

        close_ref = float(d["close"].iloc[-1]) if pd.notna(d["close"].iloc[-1]) else np.nan
        if not np.isfinite(close_ref) or close_ref == 0:
            continue

        head = d.iloc[:ORB_BARS]
        tail = d.iloc[-CLOSE_BARS:]

        open_rng = (head["high"].max() - head["low"].min()) / close_ref
        close_rng = (tail["high"].max() - tail["low"].min()) / close_ref

        total = open_rng + close_rng
        open_share = open_rng / total if total and np.isfinite(total) else np.nan
        close_share = close_rng / total if total and np.isfinite(total) else np.nan

        edge_share = np.nan
        edge_bias = np.nan
        edge_flag = 0
        if np.isfinite(open_share) and np.isfinite(close_share):
            edge_share = float(max(open_share, close_share))
            edge_bias = float(open_share - close_share)  # + => open-dominant, - => close-dominant
            edge_flag = int(edge_share >= EDGE_SHARE_FLAG)

        rows.append(
            {
                "date": session_date,
                "open_orb_range": float(open_rng),
                "close_orb_range": float(close_rng),
                "open_orb_share": float(open_share) if np.isfinite(open_share) else np.nan,
                "close_orb_share": float(close_share) if np.isfinite(close_share) else np.nan,
                "edge_orb_share": float(edge_share) if np.isfinite(edge_share) else np.nan,
                "edge_orb_bias": float(edge_bias) if np.isfinite(edge_bias) else np.nan,
                "edge_orb_flag": int(edge_flag),
            }
        )

    return pd.DataFrame(rows)


def classify_day_type(row: pd.Series) -> str:
    """
    Simple day-type classifier:
      - trend: large body relative to range and close near an extreme
      - range: otherwise
    """
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = h - l
    if not np.isfinite(rng) or rng <= 0:
        return "unknown"
    body = abs(c - o)
    body_share = body / rng

    # Close location in range (0..1)
    close_loc = (c - l) / rng
    near_extreme = (close_loc >= 0.80) or (close_loc <= 0.20)

    if body_share >= 0.55 and near_extreme:
        return "trend"
    return "range"


def build_features(df_daily: pd.DataFrame, con: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    out = df_daily.copy()

    # Prev close + returns
    out["prev_close"] = out["close"].shift(1)
    out["ret_1d"] = safe_div(out["close"], out["prev_close"]) - 1.0

    # Gap (interday)
    out["gap_open_pct"] = safe_div(out["open"], out["prev_close"]) - 1.0
    out["gap_abs_pct"] = out["gap_open_pct"].abs()

    # Range (intraday)
    out["intraday_range"] = out["high"] - out["low"]
    out["intraday_range_pct"] = safe_div(out["intraday_range"], out["close"])

    # True range
    hl = out["high"] - out["low"]
    hc = (out["high"] - out["prev_close"]).abs()
    lc = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = np.maximum(hl, np.maximum(hc, lc))
    out["true_range_pct"] = safe_div(out["true_range"], out["close"])

    # Volatility
    out["vol20"] = out["ret_1d"].rolling(VOL_WIN, min_periods=max(5, VOL_WIN // 2)).std()

    # Cluster score (simple: rolling sum of trigger days)
    out["trigger_gap_15"] = (out["gap_abs_pct"] >= TRIGGER_PCT).astype(int)
    out["trigger_range_15"] = (out["intraday_range_pct"].abs() >= TRIGGER_PCT).astype(int)
    out["trigger_tr_15"] = (out["true_range_pct"].abs() >= TRIGGER_PCT).astype(int)

    out["trigger_any_15"] = (
        (out["trigger_gap_15"] == 1) | (out["trigger_range_15"] == 1)
    ).astype(int)

    out["trigger_cluster"] = out["trigger_any_15"].rolling(CLUSTER_WIN, min_periods=1).sum()

    # A continuous "cluster_score" (scaled)
    out["cluster_score"] = safe_div(out["trigger_cluster"], pd.Series([CLUSTER_WIN] * len(out)))

    # Compression percentile rank (lower TR pct => more compressed)
    out["tr_pct_rank"] = out["true_range_pct"].rolling(
        LOOKBACK_COMPRESSION, min_periods=max(5, LOOKBACK_COMPRESSION // 2)
    ).apply(pct_rank_last, raw=False)
    out["compression_flag"] = (out["tr_pct_rank"] <= COMP_PCTL).astype(int)

    # Label: next day expansion (uses future data)
    out["next_tr_pct"] = out["true_range_pct"].shift(-1)
    out["next_tr_pct_rank"] = out["next_tr_pct"].rolling(
        LOOKBACK_EXPANSION, min_periods=max(5, LOOKBACK_EXPANSION // 2)
    ).apply(pct_rank_last, raw=False)
    out["label_next_day_expansion"] = (out["next_tr_pct_rank"] >= EXP_PCTL).astype(int)

    # Permission strength (simple + deterministic)
    # 0..3-ish: cluster + triggers + compression
    out["permission_strength"] = (
        (out["cluster_score"].fillna(0) >= 0.30).astype(int)
        + out["trigger_any_15"].fillna(0).astype(int)
        + out["compression_flag"].fillna(0).astype(int)
    ).astype(int)

    def reason(row: pd.Series) -> str:
        parts = []
        if row.get("cluster_score", 0) >= 0.30:
            parts.append("cluster")
        if row.get("trigger_any_15", 0) == 1:
            parts.append("trigger")
        if row.get("compression_flag", 0) == 1:
            parts.append("compression")
        return ",".join(parts) if parts else "none"

    out["permission_reason"] = out.apply(reason, axis=1)
    out["trade_permission"] = (out["permission_strength"] >= 2).astype(int)

    # Day type
    out["day_type"] = out.apply(classify_day_type, axis=1)

    feats = out[
        [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "prev_close",
            "ret_1d",
            "gap_open_pct",
            "gap_abs_pct",
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
            "label_next_day_expansion",
            "permission_strength",
            "permission_reason",
            "trade_permission",
            "day_type",
        ]
    ].copy()

    # Optional ORB (15m)
    if con is not None:
        intra = read_intraday_15m(con, SYMBOL)
        if intra is not None and not intra.empty:
            orb = compute_orb_features(intra)
            if not orb.empty:
                feats = feats.merge(orb, on="date", how="left")
            else:
                feats["open_orb_range"] = np.nan
                feats["close_orb_range"] = np.nan
                feats["open_orb_share"] = np.nan
                feats["close_orb_share"] = np.nan
                feats["edge_orb_share"] = np.nan
                feats["edge_orb_bias"] = np.nan
                feats["edge_orb_flag"] = 0
        else:
            feats["open_orb_range"] = np.nan
            feats["close_orb_range"] = np.nan
            feats["open_orb_share"] = np.nan
            feats["close_orb_share"] = np.nan
            feats["edge_orb_share"] = np.nan
            feats["edge_orb_bias"] = np.nan
            feats["edge_orb_flag"] = 0
    else:
        feats["open_orb_range"] = np.nan
        feats["close_orb_range"] = np.nan
        feats["open_orb_share"] = np.nan
        feats["close_orb_share"] = np.nan
        feats["edge_orb_share"] = np.nan
        feats["edge_orb_bias"] = np.nan
        feats["edge_orb_flag"] = 0

    feats["feature_ts"] = datetime.now(timezone.utc).isoformat()
    feats["feature_version"] = "v1_clean_daily_features_orb_trend"

    # Drop the first row where prev_close is NaN
    feats = feats.dropna(subset=["prev_close"]).reset_index(drop=True)
    return feats


def ensure_features_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
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
            label_next_day_expansion INTEGER,

            permission_strength INTEGER,
            permission_reason TEXT,
            trade_permission INTEGER,

            day_type TEXT,

            open_orb_range REAL,
            close_orb_range REAL,
            open_orb_share REAL,
            close_orb_share REAL,
            edge_orb_share REAL,
            edge_orb_bias REAL,
            edge_orb_flag INTEGER,

            feature_ts TEXT,
            feature_version TEXT,

            PRIMARY KEY (symbol, date)
        )
        """
    )
    con.commit()


def upsert_features(con: sqlite3.Connection, feats: pd.DataFrame) -> None:
    if feats.empty:
        return

    feats2 = feats.copy()
    feats2["date"] = pd.to_datetime(feats2["date"]).dt.strftime("%Y-%m-%d")

    cols = list(feats2.columns)
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    updates = ",".join([f"{c}=excluded.{c}" for c in cols if c not in ["symbol", "date"]])

    sql = f"""
    INSERT INTO features_daily ({col_list})
    VALUES ({placeholders})
    ON CONFLICT(symbol, date) DO UPDATE SET
      {updates}
    """
    con.executemany(sql, feats2[cols].to_records(index=False).tolist())
    con.commit()


def write_csv(feats: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_DIR / "features_daily.csv", index=False)


def write_latest_features(feats: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if feats.empty:
        return
    feats.tail(1).to_csv(OUT_DIR / "latest_features_daily.csv", index=False)


def main() -> None:
    con = connect(DB_PATH)
    try:
        ensure_features_table(con)
        truth = read_daily_truth(con, SYMBOL)
        feats = build_features(truth, con=con)
        upsert_features(con, feats)
        write_csv(feats)
        write_latest_features(feats)
        print("OK: features_daily built + upserted + CSVs written.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
