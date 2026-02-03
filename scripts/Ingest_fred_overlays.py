#!/usr/bin/env python3
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import requests

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_START = os.getenv("FRED_START", "2000-01-01")

# Keep it small + high-signal to start
SERIES = {
    "VIXCLS": "vix",     # CBOE VIX close
    "VIX9D": "vix9d",    # 9-day VIX
    "DGS10": "dgs10",    # 10Y Treasury yield
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def ensure_tables(con: sqlite3.Connection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_daily (
          date TEXT NOT NULL,
          series_id TEXT NOT NULL,
          value REAL,
          source TEXT DEFAULT 'FRED',
          ingest_ts TEXT,
          PRIMARY KEY(date, series_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS overlays_daily (
          date TEXT NOT NULL,
          symbol TEXT NOT NULL,
          overlay_type TEXT NOT NULL,
          overlay_strength REAL NOT NULL,
          notes TEXT,
          PRIMARY KEY (symbol, date, overlay_type)
        );
        """
    )
    con.commit()


def z_to_strength(z: float) -> float:
    # 0 until z>=1.0, ramps to 1 by z>=2.5 (matches your FINRA overlay logic)
    if z is None or pd.isna(z):
        return 0.0
    return float(max(0.0, min(1.0, (z - 1.0) / 1.5)))


def fetch_series_observations(series_id: str, start: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "frequency": "d",
        "observation_start": start,
    }
    r = requests.get(FRED_URL, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])

    rows = []
    for o in obs:
        v = o.get("value")
        if v is None or v == ".":
            continue
        rows.append({"date": o["date"], "series_id": series_id, "value": float(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def upsert_macro(con: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    ingest_ts = datetime.now(timezone.utc).isoformat()
    df2 = df.copy()
    df2["date"] = df2["date"].dt.strftime("%Y-%m-%d")
    df2["source"] = "FRED"
    df2["ingest_ts"] = ingest_ts

    con.executemany(
        """
        INSERT INTO macro_daily (date, series_id, value, source, ingest_ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, series_id) DO UPDATE SET
          value=excluded.value,
          source=excluded.source,
          ingest_ts=excluded.ingest_ts
        """,
        df2[["date", "series_id", "value", "source", "ingest_ts"]].values.tolist(),
    )
    con.commit()


def read_trading_days(con: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT date
    FROM bars_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    d = pd.read_sql_query(q, con, params=(SYMBOL,))
    d["date"] = pd.to_datetime(d["date"])
    return d


def compute_and_write_overlays(con: sqlite3.Connection):
    # Pull macro series we ingested
    q = """
    SELECT date, series_id, value
    FROM macro_daily
    WHERE series_id IN ('VIXCLS','VIX9D','DGS10')
    ORDER BY date ASC
    """
    m = pd.read_sql_query(q, con)
    if m.empty:
        raise RuntimeError("macro_daily is empty. Did FRED ingest succeed?")
    m["date"] = pd.to_datetime(m["date"])
    pivot = m.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").sort_index()

    # Align to actual trading days (bars_daily), forward-fill for holidays/weekends where needed
    days = read_trading_days(con)
    if days.empty:
        raise RuntimeError("bars_daily is empty. Run ingest_spy_daily first.")
    aligned = days.merge(pivot, left_on="date", right_index=True, how="left").sort_values("date")
    aligned[["VIXCLS", "VIX9D", "DGS10"]] = aligned[["VIXCLS", "VIX9D", "DGS10"]].ffill()

    # Derived features
    aligned["vix_term"] = aligned["VIX9D"] / aligned["VIXCLS"]
    aligned["dgs10_chg_5d"] = aligned["DGS10"].diff(5)

    # Z-scores (rolling)
    def zscore(s: pd.Series, win: int = 252) -> pd.Series:
        mu = s.rolling(win, min_periods=max(30, win // 4)).mean()
        sd = s.rolling(win, min_periods=max(30, win // 4)).std()
        return (s - mu) / sd

    aligned["vix_z"] = zscore(aligned["VIXCLS"])
    aligned["vix9d_z"] = zscore(aligned["VIX9D"])
    aligned["vix_term_z"] = zscore(aligned["vix_term"])
    aligned["rates_z"] = zscore(aligned["dgs10_chg_5d"], win=252)

    # Map to overlay_strength
    overlays = [
        ("vix", "VIXCLS", "vix_z"),
        ("vix9d", "VIX9D", "vix9d_z"),
        ("vix_term", "vix_term", "vix_term_z"),
        ("rates10y", "dgs10_chg_5d", "rates_z"),
    ]

    rows = []
    for overlay_type, val_col, z_col in overlays:
        for _, r in aligned.iterrows():
            z = r.get(z_col)
            val = r.get(val_col)
            strength = z_to_strength(z)
            date_str = r["date"].strftime("%Y-%m-%d")
            note = f"{overlay_type}:{val_col}={val:.4f} z={z:.2f}" if pd.notna(z) and pd.notna(val) else f"{overlay_type}:missing"
            rows.append((date_str, SYMBOL, overlay_type, float(strength), note))

    con.executemany(
        """
        INSERT INTO overlays_daily (date, symbol, overlay_type, overlay_strength, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date, overlay_type) DO UPDATE SET
          overlay_strength=excluded.overlay_strength,
          notes=excluded.notes
        """,
        rows,
    )
    con.commit()


def main():
    if not FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY env var (set it as a GitHub secret).")

    con = connect()
    try:
        ensure_tables(con)

        # 1) ingest raw macro series
        all_frames = []
        for series_id in SERIES.keys():
            df = fetch_series_observations(series_id, FRED_START)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            raise RuntimeError("No FRED data returned. Check series IDs, API key, or network.")

        raw = pd.concat(all_frames, ignore_index=True)
        upsert_macro(con, raw)

        # 2) compute and write overlays_daily
        compute_and_write_overlays(con)

        # Optional: write a quick output snapshot
        os.makedirs("outputs", exist_ok=True)
        out = pd.read_sql_query(
            "SELECT * FROM overlays_daily WHERE symbol=? AND overlay_type IN ('vix','vix9d','vix_term','rates10y') ORDER BY date DESC LIMIT 15",
            con, params=(SYMBOL,)
        )
        out.to_csv("outputs/spy_macro_overlays_tail.csv", index=False)
        print("OK: wrote outputs/spy_macro_overlays_tail.csv")

    finally:
        con.close()


if __name__ == "__main__":
    main()
