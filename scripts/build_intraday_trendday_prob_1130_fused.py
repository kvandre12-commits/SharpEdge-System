#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

INTRADAY_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
INTRADAY_TS_COL = os.getenv("INTRADAY_TS_COL", "ts")

CUTOFF_HHMM = os.getenv("CUTOFF_HHMM", "11:30")
MIN_TRAIN_DAYS = int(os.getenv("MIN_TRAIN_DAYS", "120"))

# logistic fit knobs
L2 = float(os.getenv("LOGREG_L2", "1.0"))
LR = float(os.getenv("LOGREG_LR", "0.2"))
STEPS = int(os.getenv("LOGREG_STEPS", "400"))

# fusion knobs
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))  # 0.25% (matches your aggregator default) 2
CHASE_MULT = float(os.getenv("CHASE_MULT", "1.25"))
UNWIND_MULT = float(os.getenv("UNWIND_MULT", "0.85"))
PIN_SUPPRESS = float(os.getenv("PIN_SUPPRESS", "0.60"))

OUT_DIR = os.getenv("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

NY = ZoneInfo("America/New_York")

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def standardize(X):
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig == 0, 1.0, sig)
    return (X - mu) / sig, mu, sig

def fit_logreg(X, y, l2=L2, lr=LR, steps=STEPS):
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        gw = (X.T @ (p - y)) / n + l2 * w
        gb = np.mean(p - y)
        w -= lr * gw
        b -= lr * gb
    return w, b

def cutoff_time():
    hh, mm = CUTOFF_HHMM.split(":")
    return time(int(hh), int(mm))

def ensure_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS intraday_trendday_prob (
      session_date TEXT NOT NULL,
      symbol TEXT NOT NULL,
      cutoff_ny TEXT NOT NULL,

      prob_trend REAL NOT NULL,
      prob_range REAL NOT NULL,

      -- fused adjustments
      prob_trend_fused REAL,
      prob_range_fused REAL,
      dealer_state_hint TEXT,
      gamma_proxy REAL,
      wall_strike REAL,
      dist_to_wall_pct REAL,

      -- features used
      ret_open_to_cutoff REAL,
      orbrange_pct REAL,
      orb_break_strength REAL,
      range_pct_to_cutoff REAL,
      true_range_pct_to_cutoff REAL,
      hhll_persistence REAL,
      vwap_proxy REAL,

      model_version TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (session_date, symbol, cutoff_ny)
    );
    """)
    conn.commit()

def load_daily_labels(conn, symbol):
    cols = pd.read_sql_query("PRAGMA table_info(features_daily);", conn)["name"].tolist()
    date_col = "session_date" if "session_date" in cols else "date"
    df = pd.read_sql_query(
        f"""
        SELECT {date_col} AS session_date, symbol, day_type
        FROM features_daily
        WHERE symbol = ? AND day_type IS NOT NULL
        ORDER BY session_date;
        """,
        conn, params=(symbol,)
    )
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date.astype(str)
    df["y"] = (df["day_type"].astype(str).str.lower() == "trend").astype(int)
    return df[["session_date", "symbol", "y"]]

def load_intraday(conn, symbol):
    df = pd.read_sql_query(
        f"""
        SELECT {INTRADAY_TS_COL} AS ts, symbol, open, high, low, close, volume
        FROM {INTRADAY_TABLE}
        WHERE symbol = ?
        ORDER BY ts;
        """,
        conn, params=(symbol,)
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")

    df["ts_ny"] = df["ts"].dt.tz_convert(NY)
    df["session_date"] = df["ts_ny"].dt.date.astype(str)
    df["time_ny"] = df["ts_ny"].dt.time
    return df

def compute_intraday_features(intra_df):
    ctime = cutoff_time()
    rows = []
    for session_date, g in intra_df.groupby("session_date", sort=True):
        g = g.sort_values("ts_ny")
        g = g[(g["time_ny"] >= time(9, 30)) & (g["time_ny"] <= time(16, 0))]
        if g.empty:
            continue

        g_cut = g[g["time_ny"] <= ctime]
        if len(g_cut) < 3:
            continue

        o = float(g_cut.iloc[0]["open"])
        h = float(g_cut["high"].max())
        l = float(g_cut["low"].min())
        c = float(g_cut.iloc[-1]["close"])

        g_orb = g_cut[(g_cut["time_ny"] >= time(9, 30)) & (g_cut["time_ny"] <= time(10, 30))]
        if len(g_orb) >= 2:
            orb_h = float(g_orb["high"].max())
            orb_l = float(g_orb["low"].min())
        else:
            orb_h = float(g_cut.iloc[0]["high"])
            orb_l = float(g_cut.iloc[0]["low"])

        orbrange = max(1e-9, orb_h - orb_l)
        orbrange_pct = (orbrange / o) * 100.0

        if c >= orb_h:
            orb_break_strength = (c - orb_h) / orbrange
        elif c <= orb_l:
            orb_break_strength = -(orb_l - c) / orbrange
        else:
            orb_break_strength = 0.0

        ret_open_to_cutoff = (c / o - 1.0) * 100.0
        range_pct_to_cutoff = ((h - l) / o) * 100.0
        true_range_pct_to_cutoff = (max(h - l, abs(h - o), abs(l - o)) / o) * 100.0

        closes = g_cut["close"].astype(float).values
        deltas = np.diff(closes)
        net_dir = np.sign(c - o)
        if len(deltas) > 0 and net_dir != 0:
            hhll_persistence = float(np.mean(np.sign(deltas) == net_dir))
        else:
            hhll_persistence = 0.0

        vwap_proxy = float(np.sign(c - o))

        rows.append({
            "session_date": session_date,
            "symbol": SYMBOL,
            "cutoff_ny": CUTOFF_HHMM,
            "ret_open_to_cutoff": ret_open_to_cutoff,
            "orbrange_pct": orbrange_pct,
            "orb_break_strength": orb_break_strength,
            "range_pct_to_cutoff": range_pct_to_cutoff,
            "true_range_pct_to_cutoff": true_range_pct_to_cutoff,
            "hhll_persistence": hhll_persistence,
            "vwap_proxy": vwap_proxy,
        })
    return pd.DataFrame(rows)

def load_dealer_metrics_latest(conn, symbol):
    # For each session_date pick the latest snapshot_ts within that date
    df = pd.read_sql_query(
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
          m.snapshot_ts,
          m.max_total_oi_strike AS wall_strike,
          m.gamma_proxy,
          m.dealer_state_hint,
          m.spot
        FROM options_positioning_metrics m
        JOIN latest l
          ON l.session_date = m.session_date
         AND l.snapshot_ts = m.snapshot_ts
        WHERE m.underlying = ?
        """,
        conn, params=(symbol, symbol)
    )
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date.astype(str)
    return df

def fuse_probs(scored, dealer_df):
    out = scored.merge(dealer_df[["session_date","symbol","wall_strike","gamma_proxy","dealer_state_hint","spot"]],
                       on=["session_date","symbol"], how="left")
    # dist to wall using spot if available else close proxy (use ret + open not saved; so use spot only)
    out["dist_to_wall_pct"] = np.where(
        out["wall_strike"].notna() & out["spot"].notna() & (out["spot"] != 0),
        (np.abs(out["spot"] - out["wall_strike"]) / out["spot"]) * 100.0,
        np.nan
    )

    mult = np.ones(len(out), dtype=float)
    # dealer state
    st = out["dealer_state_hint"].fillna("").astype(str).str.lower().values
    mult *= np.where(st == "chase", CHASE_MULT, 1.0)
    mult *= np.where(st == "unwind", UNWIND_MULT, 1.0)

    # pin suppression if near wall
    near_pin = (out["dist_to_wall_pct"].values / 100.0) <= PIN_THRESH_PCT
    mult *= np.where(near_pin, PIN_SUPPRESS, 1.0)

    p = out["prob_trend"].astype(float).values
    p_fused = np.clip(p * mult, 0.0, 0.99)

    out["prob_trend_fused"] = p_fused
    out["prob_range_fused"] = 1.0 - p_fused
    return out

def upsert(conn, df):
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute("""
        INSERT INTO intraday_trendday_prob (
          session_date, symbol, cutoff_ny,
          prob_trend, prob_range,
          prob_trend_fused, prob_range_fused,
          dealer_state_hint, gamma_proxy, wall_strike, dist_to_wall_pct,
          ret_open_to_cutoff, orbrange_pct, orb_break_strength,
          range_pct_to_cutoff, true_range_pct_to_cutoff,
          hhll_persistence, vwap_proxy,
          model_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_date, symbol, cutoff_ny) DO UPDATE SET
          prob_trend=excluded.prob_trend,
          prob_range=excluded.prob_range,
          prob_trend_fused=excluded.prob_trend_fused,
          prob_range_fused=excluded.prob_range_fused,
          dealer_state_hint=excluded.dealer_state_hint,
          gamma_proxy=excluded.gamma_proxy,
          wall_strike=excluded.wall_strike,
          dist_to_wall_pct=excluded.dist_to_wall_pct,
          ret_open_to_cutoff=excluded.ret_open_to_cutoff,
          orbrange_pct=excluded.orbrange_pct,
          orb_break_strength=excluded.orb_break_strength,
          range_pct_to_cutoff=excluded.range_pct_to_cutoff,
          true_range_pct_to_cutoff=excluded.true_range_pct_to_cutoff,
          hhll_persistence=excluded.hhll_persistence,
          vwap_proxy=excluded.vwap_proxy,
          model_version=excluded.model_version
        """, (
            r["session_date"], r["symbol"], r["cutoff_ny"],
            float(r["prob_trend"]), float(r["prob_range"]),
            float(r.get("prob_trend_fused", np.nan)) if pd.notna(r.get("prob_trend_fused", np.nan)) else None,
            float(r.get("prob_range_fused", np.nan)) if pd.notna(r.get("prob_range_fused", np.nan)) else None,
            r.get("dealer_state_hint", None),
            float(r["gamma_proxy"]) if pd.notna(r.get("gamma_proxy", np.nan)) else None,
            float(r["wall_strike"]) if pd.notna(r.get("wall_strike", np.nan)) else None,
            float(r["dist_to_wall_pct"]) if pd.notna(r.get("dist_to_wall_pct", np.nan)) else None,
            float(r["ret_open_to_cutoff"]),
            float(r["orbrange_pct"]),
            float(r["orb_break_strength"]),
            float(r["range_pct_to_cutoff"]),
            float(r["true_range_pct_to_cutoff"]),
            float(r["hhll_persistence"]),
            float(r["vwap_proxy"]),
            r["model_version"],
        ))
    conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_table(conn)

    labels = load_daily_labels(conn, SYMBOL)
    intra = load_intraday(conn, SYMBOL)
    intrafeat = compute_intraday_features(intra)

    df_train = intrafeat.merge(labels, on=["session_date","symbol"], how="inner")
    if len(df_train) < MIN_TRAIN_DAYS:
        raise RuntimeError(f"Not enough labeled rows to train: {len(df_train)} < {MIN_TRAIN_DAYS}")

    feature_names = [
        "ret_open_to_cutoff","orbrange_pct","orb_break_strength",
        "range_pct_to_cutoff","true_range_pct_to_cutoff",
        "hhll_persistence","vwap_proxy"
    ]

    X = df_train[feature_names].astype(float).values
    y = df_train["y"].astype(int).values

    Xs, mu, sig = standardize(X)
    w, b = fit_logreg(Xs, y)

    X_all = intrafeat[feature_names].astype(float).values
    X_all_s = (X_all - mu) / sig
    p = sigmoid(X_all_s @ w + b)

    scored = intrafeat.copy()
    scored["prob_trend"] = p
    scored["prob_range"] = 1.0 - p
    scored["model_version"] = f"logreg_np_fused_cutoff_{CUTOFF_HHMM}_l2{L2}_lr{LR}_steps{STEPS}"

    dealer = load_dealer_metrics_latest(conn, SYMBOL)
    fused = fuse_probs(scored, dealer)

    upsert(conn, fused)

    fused = fused.sort_values("session_date")
    fused.to_csv(os.path.join(OUT_DIR, "trendday_prob_1130_fused.csv"), index=False)
    fused.tail(1).to_csv(os.path.join(OUT_DIR, "latest_trendday_prob_1130_fused.csv"), index=False)

    print(f"OK: wrote {len(fused)} rows -> outputs/trendday_prob_1130_fused.csv")

if __name__ == "__main__":
    main()
