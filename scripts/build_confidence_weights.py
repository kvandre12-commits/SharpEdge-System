#!/usr/bin/env python3
"""
SharpEdge 2.1 — Confidence Weighting Layer

Purpose:
  Convert conditional expectancy research rows into deployment-confidence rows.

This is NOT a signal engine.
It is a statistical trust engine / deployment confidence filter / capital preservation layer.

Input:
  - conditional_expectancy_matrix

Outputs:
  - SQLite table: confidence_matrix
  - outputs/confidence_matrix.csv
  - outputs/top_confidence_states.csv

Design rules:
  - Fail loudly if the upstream expectancy matrix is missing or empty.
  - Penalize low sample counts.
  - Cap confidence for MICRO/BOOTSTRAP samples.
  - Reward stable fill paths and reliable fill behavior.
  - Penalize drawdown, MAE, failed fills, and stop-out proxies.
  - Never let high raw expectancy dominate when sample/risk quality is poor.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
OUTDIR = os.getenv("OUTDIR", "outputs")
IN_TABLE = os.getenv("CONDITIONAL_EXPECTANCY_TABLE", "conditional_expectancy_matrix")
OUT_TABLE = os.getenv("CONFIDENCE_MATRIX_TABLE", "confidence_matrix")

# Sample bucket thresholds. These are deployment-confidence thresholds, not research thresholds.
MICRO_N = int(os.getenv("CONF_MICRO_N", "5"))
BOOTSTRAP_N = int(os.getenv("CONF_BOOTSTRAP_N", "20"))
DEVELOPING_N = int(os.getenv("CONF_DEVELOPING_N", "40"))
SUPPORTED_N = int(os.getenv("CONF_SUPPORTED_N", "40"))

# Confidence caps by sample maturity. Low sample states may inform research, not aggressive deployment.
MICRO_CAP = float(os.getenv("CONF_MICRO_CAP", "25"))
BOOTSTRAP_CAP = float(os.getenv("CONF_BOOTSTRAP_CAP", "55"))
DEVELOPING_CAP = float(os.getenv("CONF_DEVELOPING_CAP", "80"))
SUPPORTED_CAP = float(os.getenv("CONF_SUPPORTED_CAP", "100"))

# Composite weights, expressed as score points before sample cap.
W_SAMPLE = float(os.getenv("CONF_W_SAMPLE", "20"))
W_FILL = float(os.getenv("CONF_W_FILL", "20"))
W_EXPECTANCY = float(os.getenv("CONF_W_EXPECTANCY", "15"))
W_TRADABILITY = float(os.getenv("CONF_W_TRADABILITY", "15"))
W_PATH = float(os.getenv("CONF_W_PATH", "15"))
W_RISK = float(os.getenv("CONF_W_RISK", "15"))

TOP_N = int(os.getenv("CONF_TOP_N", "50"))

HIGH_THRESH = float(os.getenv("CONF_HIGH_THRESH", "75"))
MEDIUM_THRESH = float(os.getenv("CONF_MEDIUM_THRESH", "55"))
LOW_THRESH = float(os.getenv("CONF_LOW_THRESH", "35"))


HIGH_QUALITY_PATHS = {
    "DIRECT_FILL": 1.00,
    "SQUEEZE_THEN_FILL": 0.90,
    "ROTATIONAL_BALANCE_THEN_FILL": 0.85,
    "BALANCE_THEN_FILL": 0.80,
    "RECLAIM_THEN_FILL": 0.85,
}

MID_QUALITY_PATHS = {
    "PARTIAL_FILL": 0.55,
    "SLOW_GRIND_FILL": 0.60,
    "DELAYED_FILL": 0.60,
    "UNRESOLVED": 0.45,
    "UNKNOWN": 0.45,
}

LOW_QUALITY_PATHS = {
    "FAILED_FILL_CONTINUATION": 0.10,
    "PARTIAL_FILL_REJECT": 0.20,
    "LIQUIDITY_VACUUM_CONTINUATION": 0.05,
    "ACCEPTED_CONTINUATION": 0.20,
    "NO_FILL_CONTINUATION": 0.10,
}


IDENTITY_COLS = [
    "event_type",
    "gap_direction",
    "fill_path_type",
    "vol_state",
    "open_regime_label",
]

REQUIRED_NUMERIC_COLS = [
    "n",
    "fill_rate",
    "direct_fill_rate",
    "failed_fill_rate",
    "avg_MAE_pct",
    "payoff_ratio",
    "expectancy",
    "sortino_ratio",
    "max_drawdown",
    "stop_out_rate_proxy",
    "tradability_score",
]


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if pd.isna(x) or not np.isfinite(x):
        return lo
    return float(max(lo, min(hi, x)))


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def ensure_identity_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in IDENTITY_COLS:
        if c not in out.columns:
            out[c] = "UNKNOWN"
        out[c] = out[c].fillna("UNKNOWN").astype(str)
    return out


def sample_bucket(n: int) -> str:
    if n < MICRO_N:
        return "MICRO_SAMPLE"
    if n < BOOTSTRAP_N:
        return "BOOTSTRAP"
    if n < DEVELOPING_N:
        return "DEVELOPING"
    return "SUPPORTED"


def sample_cap(bucket: str) -> float:
    if bucket == "MICRO_SAMPLE":
        return MICRO_CAP
    if bucket == "BOOTSTRAP":
        return BOOTSTRAP_CAP
    if bucket == "DEVELOPING":
        return DEVELOPING_CAP
    return SUPPORTED_CAP


def sample_weight(n: int) -> float:
    """
    Smoothly increases toward 1 as sample size approaches SUPPORTED_N.
    This avoids a hard cliff while still respecting sample maturity.
    """
    if n <= 0:
        return 0.0
    return clamp(np.sqrt(n / max(1, SUPPORTED_N)))


def path_quality(path: str, direct_fill_rate: float, failed_fill_rate: float) -> float:
    p = str(path or "UNKNOWN").upper().strip()
    if p in HIGH_QUALITY_PATHS:
        base = HIGH_QUALITY_PATHS[p]
    elif p in LOW_QUALITY_PATHS:
        base = LOW_QUALITY_PATHS[p]
    elif p in MID_QUALITY_PATHS:
        base = MID_QUALITY_PATHS[p]
    else:
        base = 0.45

    # Let observed behavior gently modify the archetype score.
    direct_bonus = 0.15 * clamp(direct_fill_rate)
    failed_penalty = 0.35 * clamp(failed_fill_rate)
    return clamp(base + direct_bonus - failed_penalty)


def normalize_positive(series: pd.Series) -> pd.Series:
    """Rank-style normalization for arbitrary score scales. Deterministic and robust to outliers."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    return s.fillna(s.min()).rank(pct=True).fillna(0.0)


def expectancy_quality(expectancy: pd.Series) -> pd.Series:
    """
    Positive expectancy is useful, negative expectancy should not receive trust.
    Rank only positive rows so a single huge outlier cannot dominate by magnitude alone.
    """
    e = pd.to_numeric(expectancy, errors="coerce").fillna(0.0)
    pos = e.clip(lower=0.0)
    if (pos > 0).sum() == 0:
        return pd.Series(0.0, index=expectancy.index)
    return normalize_positive(pos)


def risk_quality(row: pd.Series) -> float:
    fill_rate = clamp(row.get("fill_rate", 0.0))
    failed_fill_rate = clamp(row.get("failed_fill_rate", 0.0))
    stop_out = clamp(row.get("stop_out_rate_proxy", 0.0))

    mae = row.get("avg_MAE_pct", 0.0)
    if pd.isna(mae):
        mae = 0.0
    mae = abs(float(mae))

    max_dd = row.get("max_drawdown", 0.0)
    if pd.isna(max_dd):
        max_dd = 0.0
    max_dd = abs(float(max_dd))

    payoff_ratio = row.get("payoff_ratio", 0.0)
    if pd.isna(payoff_ratio) or not np.isfinite(payoff_ratio):
        payoff_ratio = 0.0
    payoff_quality = clamp(float(payoff_ratio) / 2.0)  # 2:1 or better gets full asymmetry credit

    # MAE and DD may be pct or R-like units. These transforms saturate penalties without exploding.
    mae_penalty = clamp(mae / 0.02)       # 2% adverse excursion = full penalty
    dd_penalty = clamp(max_dd / 0.05)     # 5% cumulative drawdown/R-dd = full penalty

    score = (
        0.30 * fill_rate
        + 0.25 * payoff_quality
        + 0.20 * (1.0 - failed_fill_rate)
        + 0.15 * (1.0 - mae_penalty)
        + 0.05 * (1.0 - dd_penalty)
        + 0.05 * (1.0 - stop_out)
    )
    return clamp(score)


def confidence_label(score: float) -> str:
    if score >= HIGH_THRESH:
        return "HIGH"
    if score >= MEDIUM_THRESH:
        return "MEDIUM"
    if score >= LOW_THRESH:
        return "LOW"
    return "NO_CONFIDENCE"


def deployment_ready(label: str, bucket: str) -> int:
    if bucket in {"MICRO_SAMPLE", "BOOTSTRAP"}:
        return 0
    return int(label in {"HIGH", "MEDIUM"})


def deployment_tier(label: str, bucket: str) -> str:
    if bucket == "MICRO_SAMPLE":
        return "RESEARCH_ONLY"
    if bucket == "BOOTSTRAP":
        return "PROBE_ONLY"
    if label == "HIGH" and bucket == "SUPPORTED":
        return "NORMAL_OR_AGGRESSIVE_ELIGIBLE"
    if label in {"HIGH", "MEDIUM"} and bucket in {"DEVELOPING", "SUPPORTED"}:
        return "NORMAL_ELIGIBLE"
    if label == "LOW":
        return "WATCHLIST_ONLY"
    return "NO_TRADE"


def missing_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def load_matrix(con: sqlite3.Connection) -> pd.DataFrame:
    if not table_exists(con, IN_TABLE):
        raise RuntimeError(
            f"Missing required table: {IN_TABLE}. Run scripts/build_conditional_expectancy_matrix.py first."
        )

    df = pd.read_sql_query(f"SELECT * FROM {IN_TABLE}", con)
    if df.empty:
        raise RuntimeError(f"{IN_TABLE} returned 0 rows. Cannot build confidence weights.")

    if "n" not in df.columns:
        raise RuntimeError(f"{IN_TABLE} missing required column: n")

    # Non-critical numeric columns default to zero so the layer is forward-compatible.
    for c in REQUIRED_NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = safe_num(df, c, default=0.0)

    df["n"] = df["n"].fillna(0).astype(int)
    return ensure_identity_cols(df)


def build_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sample_bucket"] = out["n"].apply(sample_bucket)
    out["sample_weight"] = out["n"].apply(sample_weight)
    out["sample_cap"] = out["sample_bucket"].apply(sample_cap)

    out["path_quality_score"] = [
        path_quality(p, d, f)
        for p, d, f in zip(
            out["fill_path_type"],
            out["direct_fill_rate"].fillna(0.0),
            out["failed_fill_rate"].fillna(0.0),
        )
    ]

    out["risk_quality_score"] = out.apply(risk_quality, axis=1)
    out["expectancy_quality_score"] = expectancy_quality(out["expectancy"])
    out["tradability_quality_score"] = normalize_positive(out["tradability_score"])
    out["fill_quality_score"] = out["fill_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)

    raw = (
        W_SAMPLE * out["sample_weight"]
        + W_FILL * out["fill_quality_score"]
        + W_EXPECTANCY * out["expectancy_quality_score"]
        + W_TRADABILITY * out["tradability_quality_score"]
        + W_PATH * out["path_quality_score"]
        + W_RISK * out["risk_quality_score"]
    )

    out["raw_confidence_score"] = raw.clip(lower=0.0, upper=100.0)
    out["confidence_score"] = np.minimum(out["raw_confidence_score"], out["sample_cap"]).round(2)
    out["confidence_label"] = out["confidence_score"].apply(confidence_label)
    out["deployment_ready"] = [
        deployment_ready(label, bucket)
        for label, bucket in zip(out["confidence_label"], out["sample_bucket"])
    ]
    out["deployment_tier"] = [
        deployment_tier(label, bucket)
        for label, bucket in zip(out["confidence_label"], out["sample_bucket"])
    ]

    out["confidence_notes"] = np.select(
        [
            out["sample_bucket"].eq("MICRO_SAMPLE"),
            out["sample_bucket"].eq("BOOTSTRAP"),
            out["failed_fill_rate"].fillna(0.0).ge(0.50),
            out["risk_quality_score"].lt(0.35),
            out["confidence_label"].eq("HIGH"),
        ],
        [
            "micro sample: research only",
            "bootstrap sample: probe/research only",
            "failed-fill rate elevated",
            "risk quality weak",
            "supported statistical confidence",
        ],
        default="confidence acceptable but monitor sample/risk stability",
    )

    out["confidence_ts"] = datetime.now(timezone.utc).isoformat()
    out["confidence_version"] = "sharpedge_2_1_confidence_weights_v1"

    sort_cols = [
        "confidence_score",
        "deployment_ready",
        "risk_quality_score",
        "path_quality_score",
        "n",
        "expectancy",
    ]
    out = out.sort_values(sort_cols, ascending=[False, False, False, False, False, False]).reset_index(drop=True)
    return out


def write_outputs(con: sqlite3.Connection, out: pd.DataFrame) -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    out.to_sql(OUT_TABLE, con, if_exists="replace", index=False)
    out.to_csv(os.path.join(OUTDIR, "confidence_matrix.csv"), index=False)

    top = out.sort_values(
        ["deployment_ready", "confidence_score", "n", "expectancy"],
        ascending=[False, False, False, False],
    ).head(TOP_N)
    top.to_csv(os.path.join(OUTDIR, "top_confidence_states.csv"), index=False)


def main() -> None:
    con = connect()
    try:
        matrix = load_matrix(con)
        confidence = build_confidence(matrix)
        write_outputs(con, confidence)

        print(f"OK: wrote {OUT_TABLE} rows={len(confidence)}")
        print("OK: wrote outputs/confidence_matrix.csv")
        print("OK: wrote outputs/top_confidence_states.csv")

        label_counts = confidence["confidence_label"].value_counts(dropna=False).to_dict()
        bucket_counts = confidence["sample_bucket"].value_counts(dropna=False).to_dict()
        ready_count = int(confidence["deployment_ready"].sum())
        print(f"OK: confidence labels={label_counts}")
        print(f"OK: sample buckets={bucket_counts}")
        print(f"OK: deployment_ready={ready_count}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
