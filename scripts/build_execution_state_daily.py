#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUT_DIR = os.getenv("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PROB_TREND_STRONG = float(os.getenv("PROB_TREND_STRONG", "0.65"))
PROB_TREND_WEAK   = float(os.getenv("PROB_TREND_WEAK", "0.45"))

# If your signals table uses a different name/column, tweak here
SIGNALS_TABLE = os.getenv("SIGNALS_TABLE", "signals_daily")
SIGNAL_COL    = os.getenv("SIGNAL_COL", "signal_strength")  # best guess; adjust if yours differs
SIGNAL_STRONG = float(os.getenv("SIGNAL_STRONG", "0.70"))   # only used if present

def table_exists(conn, name):
    r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)).fetchone()
    return r is not None

def ensure_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS execution_state_daily (
      session_date TEXT NOT NULL,
      symbol TEXT NOT NULL,

      prob_trend_fused REAL,
      prob_range_fused REAL,

      dealer_state_hint TEXT,
      gamma_proxy REAL,
      wall_strike REAL,
      dist_to_wall_pct REAL,

      cluster_score REAL,
      compression_flag INTEGER,

      signal_strength REAL,

      execution_score REAL,
      final_bias TEXT,

      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (session_date, symbol)
    );
    """)
    conn.commit()

def load_core(conn):
    # detect date column in features_daily
    cols = pd.read_sql_query("PRAGMA table_info(features_daily);", conn)["name"].tolist()
    date_col = "session_date" if "session_date" in cols else "date"

    f = pd.read_sql_query(
        f"""
        SELECT
          {date_col} AS session_date,
          symbol,
          COALESCE(cluster_score, 0.0) AS cluster_score,
          COALESCE(compression_flag, 0) AS compression_flag
        FROM features_daily
        WHERE symbol = ?
        """,
        conn, params=(SYMBOL,)
    )
    f["session_date"] = pd.to_datetime(f["session_date"]).dt.date.astype(str)

    # intraday prob (use cutoff 11:30 row)
    p = pd.read_sql_query(
        """
        SELECT
          session_date,
          symbol,
          prob_trend_fused,
          prob_range_fused
        FROM intraday_trendday_prob
        WHERE symbol = ? AND cutoff_ny = '11:30'
        """,
        conn, params=(SYMBOL,)
    )
    p["session_date"] = pd.to_datetime(p["session_date"]).dt.date.astype(str)

    # latest dealer metrics per session_date
    m = pd.read_sql_query(
        """
        WITH latest AS (
          SELECT session_date, MAX(snapshot_ts) AS snapshot_ts
          FROM options_positioning_metrics
          WHERE underlying = ?
          GROUP BY session_date
        )
        SELECT
          m.session_date,
          m.underlying AS symbol,
          m.dealer_state_hint,
          m.gamma_proxy,
          m.max_total_oi_strike AS wall_strike,
          CASE
            WHEN m.spot IS NULL OR m.spot = 0 OR m.max_total_oi_strike IS NULL THEN NULL
            ELSE ABS(m.spot - m.max_total_oi_strike) / m.spot * 100.0
          END AS dist_to_wall_pct
        FROM options_positioning_metrics m
        JOIN latest l
          ON l.session_date = m.session_date
         AND l.snapshot_ts = m.snapshot_ts
        WHERE m.underlying = ?
        """,
        conn, params=(SYMBOL, SYMBOL)
    )
    m["session_date"] = pd.to_datetime(m["session_date"]).dt.date.astype(str)

    # signal strength (optional)
    if table_exists(conn, SIGNALS_TABLE):
        s_cols = pd.read_sql_query(f"PRAGMA table_info({SIGNALS_TABLE});", conn)["name"].tolist()
        # best effort: if signal column not present, set NaN
        if SIGNAL_COL in s_cols:
            s = pd.read_sql_query(
                f"""
                SELECT session_date, symbol, {SIGNAL_COL} AS signal_strength
                FROM {SIGNALS_TABLE}
                WHERE symbol = ?
                """,
                conn, params=(SYMBOL,)
            )
            s["session_date"] = pd.to_datetime(s["session_date"]).dt.date.astype(str)
        else:
            s = pd.DataFrame(columns=["session_date","symbol","signal_strength"])
    else:
        s = pd.DataFrame(columns=["session_date","symbol","signal_strength"])

    # merge
    df = f.merge(p, on=["session_date","symbol"], how="left") \
          .merge(m, on=["session_date","symbol"], how="left") \
          .merge(s, on=["session_date","symbol"], how="left")

    return df

def decide(row):
    pt = row.get("prob_trend_fused")
    pr = row.get("prob_range_fused")
    st = (str(row.get("dealer_state_hint") or "")).lower()
    dist = row.get("dist_to_wall_pct")
    comp = int(row.get("compression_flag") or 0)
    sig = row.get("signal_strength")

    # base score from trend probability
    score = 0.0
    if pd.notna(pt):
        score += float(pt) * 70.0
    else:
        score += 35.0  # neutral if missing

    # dealer state adjustments
    if st == "chase":
        score += 12.0
    elif st == "unwind":
        score -= 8.0
    elif st == "pin":
        score -= 15.0

    # pin proximity adjustment
    if pd.notna(dist) and float(dist) <= 0.25:
        score -= 18.0

    # compression increases likelihood of expansion *if* trend probability supports it
    if comp == 1 and pd.notna(pt) and float(pt) >= 0.55:
        score += 8.0

    # ignition boost if available
    if pd.notna(sig) and float(sig) >= SIGNAL_STRONG:
        score += 10.0

    score = float(np.clip(score, 0.0, 100.0))

    # final bias rules (simple + deterministic)
    if (pd.notna(dist) and float(dist) <= 0.25) or st == "pin":
        final = "PIN_FADE"
    elif pd.notna(pt) and float(pt) >= PROB_TREND_STRONG:
        # direction selection is NOT handled here (needs intraday sign); keep it “expansion” bias.
        final = "EXPANSION_FOLLOW"
    elif pd.notna(pt) and float(pt) <= PROB_TREND_WEAK:
        final = "WHIP_WAIT"
    else:
        final = "BALANCED_SMALL"

    return score, final

def upsert(conn, df):
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute("""
        INSERT INTO execution_state_daily (
          session_date, symbol,
          prob_trend_fused, prob_range_fused,
          dealer_state_hint, gamma_proxy, wall_strike, dist_to_wall_pct,
          cluster_score, compression_flag,
          signal_strength,
          execution_score, final_bias
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_date, symbol) DO UPDATE SET
          prob_trend_fused=excluded.prob_trend_fused,
          prob_range_fused=excluded.prob_range_fused,
          dealer_state_hint=excluded.dealer_state_hint,
          gamma_proxy=excluded.gamma_proxy,
          wall_strike=excluded.wall_strike,
          dist_to_wall_pct=excluded.dist_to_wall_pct,
          cluster_score=excluded.cluster_score,
          compression_flag=excluded.compression_flag,
          signal_strength=excluded.signal_strength,
          execution_score=excluded.execution_score,
          final_bias=excluded.final_bias
        """, (
            r["session_date"], r["symbol"],
            float(r["prob_trend_fused"]) if pd.notna(r.get("prob_trend_fused")) else None,
            float(r["prob_range_fused"]) if pd.notna(r.get("prob_range_fused")) else None,
            r.get("dealer_state_hint", None),
            float(r["gamma_proxy"]) if pd.notna(r.get("gamma_proxy")) else None,
            float(r["wall_strike"]) if pd.notna(r.get("wall_strike")) else None,
            float(r["dist_to_wall_pct"]) if pd.notna(r.get("dist_to_wall_pct")) else None,
            float(r.get("cluster_score", 0.0)) if pd.notna(r.get("cluster_score")) else 0.0,
            int(r.get("compression_flag", 0) or 0),
            float(r["signal_strength"]) if pd.notna(r.get("signal_strength")) else None,
            float(r["execution_score"]),
            r["final_bias"],
        ))
    conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_table(conn)

    df = load_core(conn)
    if df.empty:
        print("No rows produced. Check features_daily / intraday_trendday_prob / options_positioning_metrics.")
        return

    # compute decisions
    scores = df.apply(lambda r: decide(r), axis=1, result_type="expand")
    df["execution_score"] = scores[0]
    df["final_bias"] = scores[1]

    upsert(conn, df)

    out = df.sort_values("session_date")
    out.to_csv(os.path.join(OUT_DIR, "execution_state_daily.csv"), index=False)
    out.tail(1).to_csv(os.path.join(OUT_DIR, "latest_execution_state_daily.csv"), index=False)

    print(f"OK: wrote {len(out)} rows -> outputs/execution_state_daily.csv")

if __name__ == "__main__":
    main()
