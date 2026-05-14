#!/usr/bin/env python3
"""
System Edge — Market State Compression Engine

Purpose:
Compress raw market metrics into a small number of interpretable
environment states.

This is NOT a prediction engine.
It is an environment classification layer.

Raw metrics:
- features_daily
- regime_daily
- options_positioning_metrics
- open_resolution_regime

Compressed outputs:
- regime_state
- dealer_state
- auction_state
- environment_state

This becomes the language layer used by:
- execution cards
- controller agents
- permission engine
- Discord summaries
- future model supervision
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")


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

    created_at TEXT,

    PRIMARY KEY(date, symbol)
)
"""


QUERY = """
SELECT
    r.date,
    r.symbol,

    r.regime_label,
    r.vol_state,
    r.vol_trend_state,
    r.compression_flag,
    r.dp_state,

    f.day_type,
    f.edge_orb_flag,

    o.dealer_state_hint,
    o.gamma_concentration,
    o.wall_drift,
    o.near_spot_density,

    s.trade_gate,
    s.pressure_state

FROM regime_daily r
LEFT JOIN features_daily f
    ON r.date = f.date
   AND r.symbol = f.symbol
LEFT JOIN options_surface_state o
    ON r.date = o.session_date
   AND r.symbol = o.underlying
LEFT JOIN signals_daily s
    ON r.date = s.date
   AND r.symbol = s.symbol

WHERE r.symbol = ?
ORDER BY r.date
"""


def classify_regime(row):
    compression = row.get("compression_flag", 0)
    vol_state = str(row.get("vol_state", "")).upper()
    vol_trend = str(row.get("vol_trend_state", "")).upper()

    if compression == 1:
        return "PIN_COMPRESSION"

    if "HIGH" in vol_state and "UP" in vol_trend:
        return "VOLATILE_UNWIND"

    if row.get("day_type") == "trend":
        return "TREND_EXPANSION"

    return "BALANCED_ROTATION"


def classify_dealer(row):
    dealer = str(row.get("dealer_state_hint", "UNKNOWN")).upper()
    gamma = float(row.get("gamma_concentration") or 0)
    density = float(row.get("near_spot_density") or 0)

    if "SHORT" in dealer:
        return "SHORT_GAMMA"

    if density > 0.7:
        return "LONG_GAMMA_PIN"

    if gamma > 0.6:
        return "UPSIDE_CHASE"

    return "DEFENSIVE"


def classify_auction(row):
    pressure = str(row.get("pressure_state", "")).upper()
    gate = int(row.get("trade_gate") or 0)

    if "BUY" in pressure and gate == 1:
        return "INITIATIVE_BUYING"

    if "SELL" in pressure and gate == 1:
        return "RESPONSIVE_SELLING"

    if gate == 0:
        return "BALANCE"

    return "FAILED_BREAKOUT"


def combine_environment(regime_state, dealer_state, auction_state):
    return f"{regime_state} | {dealer_state} | {auction_state}"



def main():
    con = sqlite3.connect(DB_PATH)
    con.execute(CREATE_SQL)

    df = pd.read_sql_query(QUERY, con, params=(SYMBOL,))

    if df.empty:
        raise SystemExit("No rows available for state engine")

    out = []

    for _, row in df.iterrows():
        regime_state = classify_regime(row)
        dealer_state = classify_dealer(row)
        auction_state = classify_auction(row)

        compression_score = float(row.get("near_spot_density") or 0)
        exhaustion_score = float(row.get("wall_drift") or 0)

        confidence = min(
            1.0,
            0.4
            + compression_score * 0.2
            + abs(exhaustion_score) * 0.1
            + (0.2 if row.get("trade_gate") == 1 else 0)
        )

        environment_state = combine_environment(
            regime_state,
            dealer_state,
            auction_state,
        )

        out.append({
            "date": row["date"],
            "symbol": row["symbol"],
            "regime_state": regime_state,
            "dealer_state": dealer_state,
            "auction_state": auction_state,
            "environment_state": environment_state,
            "compression_score": compression_score,
            "exhaustion_score": exhaustion_score,
            "confidence": confidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    out_df = pd.DataFrame(out)

    out_df.to_sql(
        "market_state_daily",
        con,
        if_exists="replace",
        index=False,
    )

    latest = out_df.tail(1).iloc[0]

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/market_state_engine.txt", "w") as f:
        f.write(
            "SYSTEM EDGE — MARKET STATE ENGINE\n"
            "===================================\n\n"
            f"date: {latest['date']}\n"
            f"regime_state: {latest['regime_state']}\n"
            f"dealer_state: {latest['dealer_state']}\n"
            f"auction_state: {latest['auction_state']}\n"
            f"environment_state: {latest['environment_state']}\n"
            f"confidence: {latest['confidence']:.2f}\n"
        )

    print(out_df.tail(5).to_string(index=False))
    print("\nState compression engine complete.")

    con.close()


if __name__ == "__main__":
    main()
