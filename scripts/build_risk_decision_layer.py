#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

MIN_SAMPLE_STANDARD = int(os.getenv("RISK_MIN_SAMPLE_STANDARD", "30"))
MIN_SAMPLE_AGGRESSIVE = int(os.getenv("RISK_MIN_SAMPLE_AGGRESSIVE", "75"))

VOL_COMPRESS_THRESH = float(os.getenv("VOL_COMPRESS_THRESH", "0.80"))
MACRO_STRESS_THRESH = float(os.getenv("MACRO_STRESS_THRESH", "0.70"))

OUTPUT_CSV = "outputs/risk_decision_layer.csv"
TOP_OUTPUT_CSV = "outputs/top_risk_allocations.csv"


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def table_exists(con, table: str) -> bool:
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None


def ensure_table(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS risk_decision_layer (
      date TEXT NOT NULL,
      symbol TEXT NOT NULL,

      deployment_state TEXT,
      deployment_confidence REAL,
      position_size_multiplier REAL,
      capital_risk_pct REAL,

      confidence_score REAL,
      path_quality_score REAL,
      risk_quality_score REAL,
      expectancy REAL,
      payoff_ratio REAL,

      mae_pct REAL,
      fill_rate REAL,
      failed_fill_rate REAL,
      squeeze_risk REAL,

      vol_state TEXT,
      macro_state TEXT,
      dp_state TEXT,
      open_regime_label TEXT,
      dealer_state_hint TEXT,

      no_trade_score REAL,
      confidence_of_inaction REAL,
      deployment_denied_reason TEXT,

      tradability_score REAL,
      sample_bucket TEXT,
      sample_n INTEGER,

      decision_ts TEXT,
      PRIMARY KEY(symbol, date)
    )
    """)
    con.commit()


def load_table(con, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table}", con)


def safe_col(df: pd.DataFrame, col: str, default=np.nan):
    if col not in df.columns:
        df[col] = default
    return df[col]


def normalize(x, low, high):
    if pd.isna(x):
        return 0.0
    return float(max(0.0, min(1.0, (x - low) / (high - low))))


def deployment_bucket(row):
    conf = row["deployment_confidence"]
    trad = row["tradability_score"]
    sample_n = row["sample_n"]
    denied = row["confidence_of_inaction"]

    if denied >= 0.65:
        return "NO_TRADE", 0.0, 0.0

    if conf < 0.35 or trad < 0.35:
        return "WATCH", 0.0, 0.0

    if sample_n < MIN_SAMPLE_STANDARD:
        return "PROBE", 0.50, 0.05

    if conf >= 0.80 and trad >= 0.75 and sample_n >= MIN_SAMPLE_AGGRESSIVE:
        return "AGGRESSIVE", 1.50, 0.15

    if conf >= 0.60 and trad >= 0.55:
        return "STANDARD", 1.00, 0.10

    return "PROBE", 0.50, 0.05


def main():
    con = connect()

    try:
        ensure_table(con)

        required = ["confidence_matrix", "conditional_expectancy_matrix"]
        missing = [t for t in required if not table_exists(con, t)]
        if missing:
            raise RuntimeError(f"Missing required upstream tables: {missing}")

        conf = load_table(con, "confidence_matrix")
        cond = load_table(con, "conditional_expectancy_matrix")

        if conf.empty:
            raise RuntimeError("confidence_matrix is empty")

        if cond.empty:
            raise RuntimeError("conditional_expectancy_matrix is empty")

        base = conf.copy()

        merge_keys = [k for k in [
            "event_type",
            "regime_id",
            "open_regime_label",
            "vol_state",
            "vol_trend_state",
            "macro_state",
            "dp_state"
        ] if k in conf.columns and k in cond.columns]

        if merge_keys:
            base = base.merge(cond, on=merge_keys, how="left", suffixes=("", "_cond"))

        # Optional context layers
        if table_exists(con, "regime_daily"):
            reg = pd.read_sql_query(
                "SELECT date, symbol, vol_state, macro_state, dp_state, macro_stress FROM regime_daily WHERE symbol=?",
                con,
                params=(SYMBOL,)
            )
            base = base.merge(reg, on=[c for c in ["date", "symbol", "vol_state", "macro_state", "dp_state"] if c in base.columns and c in reg.columns], how="left")

        if table_exists(con, "open_resolution_regime"):
            op = pd.read_sql_query(
                "SELECT session_date as date, underlying as symbol, open_regime_label, regime_confidence FROM open_resolution_regime WHERE underlying=?",
                con,
                params=(SYMBOL,)
            )
            base = base.merge(op, on=[c for c in ["date", "symbol", "open_regime_label"] if c in base.columns and c in op.columns], how="left")

        if table_exists(con, "options_positioning_metrics"):
            opt = pd.read_sql_query(
                "SELECT session_date as date, underlying as symbol, dealer_state_hint, pcr_oi FROM options_positioning_metrics WHERE underlying=?",
                con,
                params=(SYMBOL,)
            )
            base = base.merge(opt, on=[c for c in ["date", "symbol"] if c in base.columns and c in opt.columns], how="left")

        safe_col(base, "confidence_score", 0.0)
        safe_col(base, "path_quality_score", 0.0)
        safe_col(base, "risk_quality_score", 0.0)
        safe_col(base, "expectancy", 0.0)
        safe_col(base, "payoff_ratio", 0.0)
        safe_col(base, "avg_MAE_pct", 0.0)
        safe_col(base, "fill_rate", 0.0)
        safe_col(base, "failed_fill_rate", 0.0)
        safe_col(base, "squeeze_risk", 0.0)
        safe_col(base, "sample_bucket", "UNKNOWN")
        safe_col(base, "n", 0)
        safe_col(base, "macro_stress", 0.0)

        base["sample_n"] = pd.to_numeric(base["n"], errors="coerce").fillna(0).astype(int)

        expectancy_score = base["expectancy"].apply(lambda x: normalize(x, -0.01, 0.03))
        payoff_score = base["payoff_ratio"].apply(lambda x: normalize(x, 0.5, 3.0))
        fill_score = base["fill_rate"].apply(lambda x: normalize(x, 0.30, 0.90))

        mae_penalty = 1.0 - base["avg_MAE_pct"].apply(lambda x: normalize(abs(x), 0.002, 0.03))
        fail_penalty = 1.0 - base["failed_fill_rate"].apply(lambda x: normalize(x, 0.05, 0.60))
        squeeze_penalty = 1.0 - base["squeeze_risk"].apply(lambda x: normalize(x, 0.05, 0.50))

        base["tradability_score"] = (
            expectancy_score * 0.25 +
            payoff_score * 0.15 +
            fill_score * 0.15 +
            base["path_quality_score"].fillna(0) * 0.20 +
            base["risk_quality_score"].fillna(0) * 0.15 +
            mae_penalty * 0.10
        ).clip(0, 1)

        instability = (
            (1.0 - fail_penalty) * 0.35 +
            (1.0 - squeeze_penalty) * 0.25 +
            base["macro_stress"].fillna(0) * 0.20
        )

        base["deployment_confidence"] = (
            base["confidence_score"].fillna(0) * 0.40 +
            base["path_quality_score"].fillna(0) * 0.20 +
            base["risk_quality_score"].fillna(0) * 0.20 +
            base["tradability_score"] * 0.20
        ).clip(0, 1)

        base["confidence_of_inaction"] = (
            instability * 0.50 +
            (1.0 - base["tradability_score"]) * 0.30 +
            (1.0 - base["deployment_confidence"]) * 0.20
        ).clip(0, 1)

        reasons = []
        for _, r in base.iterrows():
            deny = []

            if r["sample_n"] < MIN_SAMPLE_STANDARD:
                deny.append("LOW_SAMPLE")

            if r["macro_stress"] >= MACRO_STRESS_THRESH:
                deny.append("MACRO_STRESS")

            if r["failed_fill_rate"] >= 0.45:
                deny.append("FAILED_FILL_INSTABILITY")

            if r["avg_MAE_pct"] >= 0.02:
                deny.append("HIGH_MAE")

            if r["squeeze_risk"] >= 0.35:
                deny.append("SQUEEZE_RISK")

            reasons.append(",".join(deny) if deny else "NONE")

        base["deployment_denied_reason"] = reasons
        base["no_trade_score"] = base["confidence_of_inaction"]

        decisions = base.apply(deployment_bucket, axis=1)

        base["deployment_state"] = [d[0] for d in decisions]
        base["position_size_multiplier"] = [d[1] for d in decisions]
        base["capital_risk_pct"] = [d[2] for d in decisions]

        vol_compress = base["vol_state"].astype(str).str.lower().eq("high")
        macro_compress = base["macro_stress"].fillna(0) >= VOL_COMPRESS_THRESH

        compress_mask = vol_compress | macro_compress

        base.loc[compress_mask, "position_size_multiplier"] *= 0.50
        base.loc[compress_mask, "capital_risk_pct"] *= 0.50

        base["decision_ts"] = datetime.now(timezone.utc).isoformat()

        out_cols = [
            "date",
            "symbol",
            "deployment_state",
            "deployment_confidence",
            "position_size_multiplier",
            "capital_risk_pct",
            "confidence_score",
            "path_quality_score",
            "risk_quality_score",
            "expectancy",
            "payoff_ratio",
            "avg_MAE_pct",
            "fill_rate",
            "failed_fill_rate",
            "squeeze_risk",
            "vol_state",
            "macro_state",
            "dp_state",
            "open_regime_label",
            "dealer_state_hint",
            "no_trade_score",
            "confidence_of_inaction",
            "deployment_denied_reason",
            "tradability_score",
            "sample_bucket",
            "sample_n",
            "decision_ts"
        ]

        out = base.copy()
        out["mae_pct"] = out["avg_MAE_pct"]

        records = out[[
            "date","symbol","deployment_state","deployment_confidence",
            "position_size_multiplier","capital_risk_pct",
            "confidence_score","path_quality_score","risk_quality_score",
            "expectancy","payoff_ratio","mae_pct","fill_rate",
            "failed_fill_rate","squeeze_risk","vol_state",
            "macro_state","dp_state","open_regime_label",
            "dealer_state_hint","no_trade_score",
            "confidence_of_inaction","deployment_denied_reason",
            "tradability_score","sample_bucket","sample_n","decision_ts"
        ]]

        con.executemany(
            """
            INSERT OR REPLACE INTO risk_decision_layer (
              date,symbol,deployment_state,deployment_confidence,
              position_size_multiplier,capital_risk_pct,
              confidence_score,path_quality_score,risk_quality_score,
              expectancy,payoff_ratio,mae_pct,fill_rate,
              failed_fill_rate,squeeze_risk,vol_state,
              macro_state,dp_state,open_regime_label,
              dealer_state_hint,no_trade_score,
              confidence_of_inaction,deployment_denied_reason,
              tradability_score,sample_bucket,sample_n,decision_ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            records.to_records(index=False).tolist()
        )

        con.commit()

        os.makedirs("outputs", exist_ok=True)

        records.to_csv(OUTPUT_CSV, index=False)

        top = records.sort_values(
            ["deployment_confidence", "tradability_score"],
            ascending=[False, False]
        ).head(25)

        top.to_csv(TOP_OUTPUT_CSV, index=False)

        print(f"OK: wrote {OUTPUT_CSV} rows={len(records)}")
        print(f"OK: wrote {TOP_OUTPUT_CSV} rows={len(top)}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
