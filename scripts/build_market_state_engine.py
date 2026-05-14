#!/usr/bin/env python3
"""
System Edge — Market State Compression Engine

Compress raw market metrics into a small number of interpretable environment
states. This is not a prediction engine; it is the language layer between raw
metrics and execution decisions.
"""

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_TXT = os.getenv("MARKET_STATE_OUT_TXT", "outputs/market_state_engine.txt")
OUT_CSV = os.getenv("MARKET_STATE_OUT_CSV", "outputs/market_state_daily.csv")
STATE_VERSION = "v1_state_compression_interpretable"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS market_state_daily (
  date TEXT NOT NULL,
  symbol TEXT NOT NULL,
  regime_state TEXT,
  dealer_state TEXT,
  auction_state TEXT,
  environment_state TEXT,
  compression_score REAL,
  exhaustion_score REAL,
  confidence REAL,
  state_version TEXT,
  created_at TEXT,
  PRIMARY KEY(symbol, date)
)
"""


def table_exists(con, table):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def columns(con, table):
    if not table_exists(con, table):
        return set()
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def ensure_schema(con):
    con.execute(CREATE_SQL)
    have = columns(con, "market_state_daily")
    for col, typ in {
        "regime_state": "TEXT",
        "dealer_state": "TEXT",
        "auction_state": "TEXT",
        "environment_state": "TEXT",
        "compression_score": "REAL",
        "exhaustion_score": "REAL",
        "confidence": "REAL",
        "state_version": "TEXT",
        "created_at": "TEXT",
    }.items():
        if col not in have:
            con.execute(f"ALTER TABLE market_state_daily ADD COLUMN {col} {typ}")
    con.commit()


def select_or_null(cols, name, alias=None, default="NULL"):
    alias = alias or name
    if name in cols:
        return name if alias == name else f"{name} AS {alias}"
    return f"{default} AS {alias}"


def read_regime(con):
    if not table_exists(con, "regime_daily"):
        raise RuntimeError("Missing regime_daily. Run build_regime_spy_daily.py first.")
    cols = columns(con, "regime_daily")
    q = f"""
    SELECT
      date,
      symbol,
      {select_or_null(cols, 'regime_label')},
      {select_or_null(cols, 'vol_state')},
      {select_or_null(cols, 'vol_trend_state')},
      {select_or_null(cols, 'compression_flag', default='0')},
      {select_or_null(cols, 'dp_state')},
      {select_or_null(cols, 'macro_state')}
    FROM regime_daily
    WHERE symbol=?
    ORDER BY date
    """
    return pd.read_sql_query(q, con, params=(SYMBOL,))


def read_features(con):
    if not table_exists(con, "features_daily"):
        return pd.DataFrame(columns=["date", "symbol", "day_type", "edge_orb_flag", "trigger_cluster"])
    cols = columns(con, "features_daily")
    q = f"""
    SELECT
      date,
      symbol,
      {select_or_null(cols, 'day_type')},
      {select_or_null(cols, 'edge_orb_flag', default='0')},
      {select_or_null(cols, 'trigger_cluster', default='0')}
    FROM features_daily
    WHERE symbol=?
    ORDER BY date
    """
    return pd.read_sql_query(q, con, params=(SYMBOL,))


def read_surface(con):
    if not table_exists(con, "options_surface_state"):
        return pd.DataFrame(columns=["date", "symbol", "transition_state", "gamma_concentration", "wall_drift", "near_spot_density", "surface_skew"])
    cols = columns(con, "options_surface_state")
    q = f"""
    SELECT
      session_date AS date,
      underlying AS symbol,
      {select_or_null(cols, 'transition_state')},
      {select_or_null(cols, 'gamma_concentration', default='0')},
      {select_or_null(cols, 'wall_drift', default='0')},
      {select_or_null(cols, 'near_spot_density', default='0')},
      {select_or_null(cols, 'surface_skew', default='0')}
    FROM options_surface_state
    WHERE underlying=?
    ORDER BY session_date
    """
    return pd.read_sql_query(q, con, params=(SYMBOL,))


def read_signals(con):
    if not table_exists(con, "signals_daily"):
        return pd.DataFrame(columns=["date", "symbol", "trade_gate", "pressure_state", "early_bucket"])
    cols = columns(con, "signals_daily")
    q = f"""
    SELECT
      date,
      symbol,
      {select_or_null(cols, 'trade_gate', default='0')},
      {select_or_null(cols, 'pressure_state')},
      {select_or_null(cols, 'early_bucket')}
    FROM signals_daily
    WHERE symbol=?
    ORDER BY date
    """
    return pd.read_sql_query(q, con, params=(SYMBOL,))


def classify_regime(row):
    compression = int(row.get("compression_flag") or 0)
    vol_state = str(row.get("vol_state") or "unknown").lower()
    vol_trend = str(row.get("vol_trend_state") or "unknown").lower()
    day_type = str(row.get("day_type") or "").lower()
    surface = str(row.get("transition_state") or "").upper()

    if "PIN" in surface or compression == 1:
        return "PIN_COMPRESSION"
    if "trend" in day_type and vol_trend == "rising":
        return "TREND_EXPANSION"
    if vol_state == "high" and vol_trend in {"rising", "flat"}:
        return "VOLATILE_UNWIND"
    if "FAILED" in surface:
        return "FAILED_ACCEPTANCE"
    return "BALANCED_ROTATION"


def classify_dealer(row):
    surface = str(row.get("transition_state") or "").upper()
    gamma = float(row.get("gamma_concentration") or 0.0)
    density = float(row.get("near_spot_density") or 0.0)
    skew = float(row.get("surface_skew") or 0.0)

    if "DOWNSIDE" in surface or skew < -0.25:
        return "SHORT_GAMMA"
    if "PIN" in surface or density >= 0.45:
        return "LONG_GAMMA_PIN"
    if "UPSIDE" in surface or (gamma >= 0.55 and skew > 0.15):
        return "UPSIDE_CHASE"
    if "DEGRADED" in surface:
        return "LIQUIDITY_VACUUM"
    return "DEFENSIVE"


def classify_auction(row):
    pressure = str(row.get("pressure_state") or "").upper()
    day_type = str(row.get("day_type") or "").lower()
    gate = int(row.get("trade_gate") or 0)
    edge = int(row.get("edge_orb_flag") or 0)

    if "BUY" in pressure and gate == 1:
        return "INITIATIVE_BUYING"
    if "SELL" in pressure and gate == 1:
        return "RESPONSIVE_SELLING"
    if edge == 1 and "trend" in day_type:
        return "TRAPPED_SHORTS" if "up" in day_type else "TRAPPED_LONGS"
    if gate == 0:
        return "BALANCE"
    return "FAILED_BREAKOUT"


def confidence_score(row):
    score = 0.35
    if pd.notna(row.get("regime_label")):
        score += 0.15
    if pd.notna(row.get("transition_state")):
        score += 0.15
    if int(row.get("trade_gate") or 0) == 1:
        score += 0.15
    score += min(float(row.get("near_spot_density") or 0.0), 1.0) * 0.10
    score += min(abs(float(row.get("surface_skew") or 0.0)), 1.0) * 0.10
    return round(min(score, 1.0), 4)


def build_frame(con):
    df = read_regime(con)
    if df.empty:
        raise RuntimeError("regime_daily has no rows for symbol")
    for opt in (read_features(con), read_surface(con), read_signals(con)):
        if not opt.empty:
            df = df.merge(opt, on=["date", "symbol"], how="left")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, row in df.iterrows():
        regime_state = classify_regime(row)
        dealer_state = classify_dealer(row)
        auction_state = classify_auction(row)
        rows.append({
            "date": str(row["date"]),
            "symbol": str(row["symbol"]),
            "regime_state": regime_state,
            "dealer_state": dealer_state,
            "auction_state": auction_state,
            "environment_state": f"{regime_state} | {dealer_state} | {auction_state}",
            "compression_score": float(row.get("near_spot_density") or 0.0),
            "exhaustion_score": float(row.get("wall_drift") or 0.0),
            "confidence": confidence_score(row),
            "state_version": STATE_VERSION,
            "created_at": now,
        })
    return pd.DataFrame(rows)


def write_outputs(con, out):
    cols = ["date", "symbol", "regime_state", "dealer_state", "auction_state", "environment_state", "compression_score", "exhaustion_score", "confidence", "state_version", "created_at"]
    upsert = f"""
    INSERT INTO market_state_daily ({','.join(cols)})
    VALUES ({','.join(['?'] * len(cols))})
    ON CONFLICT(symbol, date) DO UPDATE SET
      regime_state=excluded.regime_state,
      dealer_state=excluded.dealer_state,
      auction_state=excluded.auction_state,
      environment_state=excluded.environment_state,
      compression_score=excluded.compression_score,
      exhaustion_score=excluded.exhaustion_score,
      confidence=excluded.confidence,
      state_version=excluded.state_version,
      created_at=excluded.created_at
    """
    con.executemany(upsert, out[cols].to_records(index=False).tolist())
    con.commit()

    os.makedirs("outputs", exist_ok=True)
    out[cols].to_csv(OUT_CSV, index=False)
    latest = out.sort_values("date").iloc[-1]
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(
            "SYSTEM EDGE — MARKET STATE ENGINE\n"
            "===================================\n\n"
            f"date: {latest['date']}\n"
            f"regime_state: {latest['regime_state']}\n"
            f"dealer_state: {latest['dealer_state']}\n"
            f"auction_state: {latest['auction_state']}\n"
            f"environment_state: {latest['environment_state']}\n"
            f"confidence: {latest['confidence']:.2f}\n"
            f"state_version: {latest['state_version']}\n"
        )


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_schema(con)
        out = build_frame(con)
        write_outputs(con, out)
        print(out.tail(5).to_string(index=False))
        print("\nOK: state compression engine complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
