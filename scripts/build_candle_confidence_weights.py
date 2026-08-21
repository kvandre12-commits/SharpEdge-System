#!/usr/bin/env python3
"""SharpEdge candle 2.1 — confidence weighting layer.

Purpose:
  Convert candle-conditioned expectancy rows into deployment-confidence rows.

This is NOT a signal engine and never grants execution authority. It is a trust
filter: sample maturity, tier specificity, target/stop reliability, expectancy,
and adverse-excursion risk decide whether a candle state is research-only,
probe-only, watchlist, or cockpit-surface eligible.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
OUTDIR = Path(os.getenv("OUTDIR", "outputs"))
IN_TABLE = os.getenv(
    "CANDLE_EXPECTANCY_MATRIX_TABLE", "candle_conditioned_expectancy_matrix"
)
OUT_TABLE = os.getenv("CANDLE_CONFIDENCE_TABLE", "candle_confidence_matrix")

MICRO_N = int(os.getenv("CANDLE_CONF_MICRO_N", "8"))
BOOTSTRAP_N = int(os.getenv("CANDLE_CONF_BOOTSTRAP_N", "20"))
DEVELOPING_N = int(os.getenv("CANDLE_CONF_DEVELOPING_N", "40"))
SUPPORTED_N = int(os.getenv("CANDLE_CONF_SUPPORTED_N", "80"))

MICRO_CAP = float(os.getenv("CANDLE_CONF_MICRO_CAP", "25"))
BOOTSTRAP_CAP = float(os.getenv("CANDLE_CONF_BOOTSTRAP_CAP", "50"))
DEVELOPING_CAP = float(os.getenv("CANDLE_CONF_DEVELOPING_CAP", "75"))
SUPPORTED_CAP = float(os.getenv("CANDLE_CONF_SUPPORTED_CAP", "100"))

W_SAMPLE = float(os.getenv("CANDLE_CONF_W_SAMPLE", "20"))
W_TARGET = float(os.getenv("CANDLE_CONF_W_TARGET", "20"))
W_EXPECTANCY = float(os.getenv("CANDLE_CONF_W_EXPECTANCY", "20"))
W_RISK = float(os.getenv("CANDLE_CONF_W_RISK", "15"))
W_RESOLUTION = float(os.getenv("CANDLE_CONF_W_RESOLUTION", "10"))
W_TIER = float(os.getenv("CANDLE_CONF_W_TIER", "10"))
W_CLARITY = float(os.getenv("CANDLE_CONF_W_CLARITY", "5"))

HIGH_THRESH = float(os.getenv("CANDLE_CONF_HIGH_THRESH", "75"))
MEDIUM_THRESH = float(os.getenv("CANDLE_CONF_MEDIUM_THRESH", "55"))
LOW_THRESH = float(os.getenv("CANDLE_CONF_LOW_THRESH", "35"))
TOP_N = int(os.getenv("CANDLE_CONF_TOP_N", "50"))

IDENTITY_COLS = [
    "match_tier",
    "event_name",
    "event_direction",
    "nearest_reference_name",
    "nearest_reference_relation",
    "reference_distance_bucket",
    "acceptance_state",
    "volume_confirmation",
    "vol_state",
    "macro_state",
    "dp_state",
    "regime_label",
    "open_regime_label",
    "time_bucket",
]

REQUIRED_NUMERIC_COLS = [
    "n",
    "target_before_stop_rate",
    "stop_before_target_rate",
    "same_bar_rate",
    "no_resolution_rate",
    "up_target_first_rate",
    "down_target_first_rate",
    "avg_realized_R",
    "avg_favorable_excursion_pct",
    "avg_adverse_excursion_pct",
]

TIER_QUALITY = {
    "tier_1_full": 1.00,
    "tier_2_execution": 0.85,
    "tier_3_core": 0.70,
    "tier_4_event_only": 0.55,
}

TIER_CAP = {
    "tier_1_full": 100.0,
    "tier_2_execution": 90.0,
    "tier_3_core": 75.0,
    "tier_4_event_only": 60.0,
}

DIRECTIONAL = {"CALLS", "PUTS"}


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return lo
    return float(max(lo, min(hi, value)))


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def normalize_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if (values > 0).sum() == 0:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def sample_bucket(n: int) -> str:
    if n < MICRO_N:
        return "MICRO_SAMPLE"
    if n < BOOTSTRAP_N:
        return "BOOTSTRAP"
    if n < DEVELOPING_N:
        return "DEVELOPING"
    if n < SUPPORTED_N:
        return "SUPPORTED"
    return "DEEP_SAMPLE"


def sample_cap(bucket: str) -> float:
    if bucket == "MICRO_SAMPLE":
        return MICRO_CAP
    if bucket == "BOOTSTRAP":
        return BOOTSTRAP_CAP
    if bucket == "DEVELOPING":
        return DEVELOPING_CAP
    return SUPPORTED_CAP


def sample_weight(n: int) -> float:
    if n <= 0:
        return 0.0
    return clamp(np.sqrt(n / max(SUPPORTED_N, 1)))


def confidence_label(score: float) -> str:
    if score >= HIGH_THRESH:
        return "HIGH"
    if score >= MEDIUM_THRESH:
        return "MEDIUM"
    if score >= LOW_THRESH:
        return "LOW"
    return "NO_CONFIDENCE"


def deployment_tier(
    label: str,
    bucket: str,
    match_tier: str,
    direction: str,
    positive_edge: bool,
) -> str:
    if bucket == "MICRO_SAMPLE":
        return "RESEARCH_ONLY"
    if bucket == "BOOTSTRAP":
        return "PROBE_ONLY"
    if match_tier == "tier_4_event_only":
        return "CONTEXT_ONLY"
    if direction not in DIRECTIONAL:
        return "WATCHLIST_ONLY"
    if not positive_edge:
        return "WATCHLIST_ONLY"
    if label == "HIGH" and bucket in {"SUPPORTED", "DEEP_SAMPLE"}:
        return "COCKPIT_SURFACE_ELIGIBLE"
    if label in {"HIGH", "MEDIUM"} and bucket in {
        "DEVELOPING",
        "SUPPORTED",
        "DEEP_SAMPLE",
    }:
        return "WATCHLIST_OR_PROBE_ELIGIBLE"
    if label == "LOW":
        return "WATCHLIST_ONLY"
    return "NO_TRADE"


def deployment_ready(
    label: str,
    bucket: str,
    match_tier: str,
    direction: str,
    positive_edge: bool,
) -> int:
    tier = deployment_tier(label, bucket, match_tier, direction, positive_edge)
    return int(tier in {"COCKPIT_SURFACE_ELIGIBLE", "WATCHLIST_OR_PROBE_ELIGIBLE"})


def load_matrix(con: sqlite3.Connection) -> pd.DataFrame:
    if not table_exists(con, IN_TABLE):
        raise RuntimeError(
            f"Missing required table: {IN_TABLE}. Run scripts/build_candle_expectancy_pipeline.py first."
        )
    df = pd.read_sql_query(f"SELECT * FROM {IN_TABLE}", con)
    if df.empty:
        raise RuntimeError(f"{IN_TABLE} returned 0 rows")
    if "n" not in df.columns:
        raise RuntimeError(f"{IN_TABLE} missing n column")

    out = df.copy()
    for col in IDENTITY_COLS:
        if col not in out.columns:
            out[col] = "ANY" if col != "event_name" else "UNKNOWN"
        out[col] = out[col].fillna("ANY").astype(str)
    for col in REQUIRED_NUMERIC_COLS:
        out[col] = safe_num(out, col, 0.0).fillna(0.0)
    out["n"] = out["n"].astype(int)
    return out


def risk_quality(df: pd.DataFrame) -> pd.Series:
    stop = df["stop_before_target_rate"].clip(lower=0.0, upper=1.0)
    same_bar = df["same_bar_rate"].clip(lower=0.0, upper=1.0)
    adverse = df["avg_adverse_excursion_pct"].abs().fillna(0.0)
    favorable = df["avg_favorable_excursion_pct"].abs().fillna(0.0)
    adverse_penalty = (adverse / 0.003).clip(lower=0.0, upper=1.0)
    asymmetry = (favorable / adverse.replace(0.0, 0.0001) / 2.0).clip(
        lower=0.0, upper=1.0
    )
    return (
        0.35 * (1.0 - stop)
        + 0.25 * asymmetry
        + 0.25 * (1.0 - adverse_penalty)
        + 0.15 * (1.0 - same_bar)
    ).clip(lower=0.0, upper=1.0)


def target_quality(df: pd.DataFrame) -> pd.Series:
    directional = df["event_direction"].isin(DIRECTIONAL)
    directional_quality = df["target_before_stop_rate"].clip(lower=0.0, upper=1.0)
    two_sided_best = df[["up_target_first_rate", "down_target_first_rate"]].max(axis=1)
    two_sided_clarity = (
        df["up_target_first_rate"] - df["down_target_first_rate"]
    ).abs()
    neutral_quality = (0.65 * two_sided_best + 0.35 * two_sided_clarity).clip(
        lower=0.0, upper=1.0
    )
    return directional_quality.where(directional, neutral_quality)


def build_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_bucket"] = out["n"].apply(sample_bucket)
    out["sample_weight"] = out["n"].apply(sample_weight)
    out["sample_cap"] = out["sample_bucket"].apply(sample_cap)
    out["tier_quality_score"] = out["match_tier"].map(TIER_QUALITY).fillna(0.45)
    out["tier_cap"] = out["match_tier"].map(TIER_CAP).fillna(50.0)
    out["target_quality_score"] = target_quality(out)
    out["risk_quality_score"] = risk_quality(out)
    out["resolution_quality_score"] = (1.0 - out["no_resolution_rate"]).clip(
        lower=0.0, upper=1.0
    )
    out["expectancy_quality_score"] = normalize_positive(out["avg_realized_R"])
    out["directional_clarity_score"] = (
        (out["up_target_first_rate"] - out["down_target_first_rate"])
        .abs()
        .clip(lower=0.0, upper=1.0)
    )

    raw = (
        W_SAMPLE * out["sample_weight"]
        + W_TARGET * out["target_quality_score"]
        + W_EXPECTANCY * out["expectancy_quality_score"]
        + W_RISK * out["risk_quality_score"]
        + W_RESOLUTION * out["resolution_quality_score"]
        + W_TIER * out["tier_quality_score"]
        + W_CLARITY * out["directional_clarity_score"]
    )
    total_weight = max(
        W_SAMPLE + W_TARGET + W_EXPECTANCY + W_RISK + W_RESOLUTION + W_TIER + W_CLARITY,
        1.0,
    )
    out["raw_confidence_score"] = (raw / total_weight * 100.0).clip(
        lower=0.0, upper=100.0
    )
    out["confidence_score"] = np.minimum.reduce(
        [out["raw_confidence_score"], out["sample_cap"], out["tier_cap"]]
    ).round(2)
    out["confidence_label"] = out["confidence_score"].apply(confidence_label)
    out["positive_edge"] = (
        out["event_direction"].isin(DIRECTIONAL)
        & out["avg_realized_R"].gt(0.0)
        & out["target_before_stop_rate"].gt(out["stop_before_target_rate"])
    )
    out["deployment_tier"] = [
        deployment_tier(label, bucket, tier, direction, positive_edge)
        for label, bucket, tier, direction, positive_edge in zip(
            out["confidence_label"],
            out["sample_bucket"],
            out["match_tier"],
            out["event_direction"],
            out["positive_edge"],
        )
    ]
    out["deployment_ready"] = [
        deployment_ready(label, bucket, tier, direction, positive_edge)
        for label, bucket, tier, direction, positive_edge in zip(
            out["confidence_label"],
            out["sample_bucket"],
            out["match_tier"],
            out["event_direction"],
            out["positive_edge"],
        )
    ]
    out["confidence_notes"] = np.select(
        [
            out["sample_bucket"].eq("MICRO_SAMPLE"),
            out["sample_bucket"].eq("BOOTSTRAP"),
            out["match_tier"].eq("tier_4_event_only"),
            ~out["positive_edge"],
            out["risk_quality_score"].lt(0.35),
            out["no_resolution_rate"].ge(0.50),
            out["confidence_label"].eq("HIGH"),
        ],
        [
            "micro sample: research only",
            "bootstrap sample: probe only",
            "event-only context: not specific enough for deployment",
            "non-positive directional edge: watch/research only",
            "risk quality weak",
            "no-resolution rate elevated",
            "supported candle expectancy confidence",
        ],
        default="confidence acceptable; monitor tier/sample stability",
    )
    out["confidence_ts"] = datetime.now(timezone.utc).isoformat()
    out["confidence_version"] = "sharpedge_candle_2_1_confidence_v1"
    return out.sort_values(
        [
            "deployment_ready",
            "confidence_score",
            "risk_quality_score",
            "target_quality_score",
            "n",
            "avg_realized_R",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def write_outputs(con: sqlite3.Connection, confidence: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    confidence.to_sql(OUT_TABLE, con, if_exists="replace", index=False)
    confidence.to_csv(OUTDIR / "candle_confidence_matrix.csv", index=False)
    top = confidence.sort_values(
        ["deployment_ready", "confidence_score", "n", "avg_realized_R"],
        ascending=[False, False, False, False],
    ).head(TOP_N)
    top.to_csv(OUTDIR / "top_candle_confidence_states.csv", index=False)


def main() -> int:
    con = connect()
    try:
        matrix = load_matrix(con)
        confidence = build_confidence(matrix)
        write_outputs(con, confidence)
        print(f"OK: wrote {OUT_TABLE} rows={len(confidence)}")
        print("OK: wrote outputs/candle_confidence_matrix.csv")
        print("OK: wrote outputs/top_candle_confidence_states.csv")
        print(
            "OK: confidence labels="
            f"{confidence['confidence_label'].value_counts(dropna=False).to_dict()}"
        )
        print(
            "OK: sample buckets="
            f"{confidence['sample_bucket'].value_counts(dropna=False).to_dict()}"
        )
        print(f"OK: deployment_ready={int(confidence['deployment_ready'].sum())}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
