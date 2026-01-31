import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOL = "SPY"
DB_PATH = "data/truth.db"
CSV_OUT = "data/bars_daily.csv"

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("raw", exist_ok=True)

def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS bars_daily (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        source TEXT,
        ingest_ts TEXT,
        PRIMARY KEY (symbol, date)
    )
    """)
    return con

def fetch_spy_daily():
    # yfinance sometimes returns MultiIndex columns (tuples). Flatten them safely.
sym = SYMBOL[0] if isinstance(SYMBOL, (list, tuple)) else SYMBOL

df = yf.download(sym, period="2y", interval="1d", auto_adjust=False)
df = df.reset_index()
# Flatten MultiIndex columns from yfinance (tuples like ('Open','SPY'))
def _clean_col(c):
    if isinstance(c, tuple):
        c = "_".join([str(x) for x in c if x is not None and str(x) != ""])
    return str(c).strip().lower().replace(" ", "_")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        if isinstance(c, tuple):
            c = "_".join(str(x) for x in c if x)
        return str(c).strip().lower().replace(" ", "_")

    df.columns = [norm(c) for c in df.columns]
    return df


if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    # Example: ('Open','SPY') -> 'open_spy'
   # Normalize columns from yfinance (sometimes MultiIndex -> tuples)
df.columns = [
    "_".join([str(x) for x in c if x not in (None, "")]).strip()
    if isinstance(c, tuple) else str(c).strip()
    for c in df.columns
]
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Pick the right column names whether yfinance returns "open" or "spy_open"/"open_spy"
def pick(*names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing columns. Have: {df.columns.tolist()}")

date_col = pick("date", "datetime")
open_col = pick("open", "spy_open", "open_spy")
high_col = pick("high", "spy_high", "high_spy")
low_col  = pick("low",  "spy_low",  "low_spy")
close_col= pick("close","spy_close","close_spy")
vol_col  = pick("volume","spy_volume","volume_spy")

df[date_col] = pd.to_datetime(df[date_col]).dt.date.astype(str)

out = df[[date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
out.columns = ["date", "open", "high", "low", "close", "volume"]
 df.columns = [
        "_".join([str(x) for x in col if x is not None and str(x) != ""])
        for col in df.columns.to_list()
    ]
else:
    df.columns = [str(c) for c in df.columns]

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
# yfinance usually uses "Date" before cleaning -> becomes "date"
df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["symbol"] = SYMBOL
    out["source"] = "yfinance"
    out["ingest_ts"] = datetime.now(timezone.utc).isoformat()
    return out

def upsert(con, rows):
    con.executemany("""
        INSERT INTO bars_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source,
            ingest_ts=excluded.ingest_ts
    """, rows[["symbol","date","open","high","low","close","volume","source","ingest_ts"]].itertuples(index=False))
    con.commit()

def export_csv(con):
    df = pd.read_sql_query("SELECT * FROM bars_daily ORDER BY date", con)
    df.to_csv(CSV_OUT, index=False)

def main():
    ensure_dirs()
    con = connect_db()
    pulled = fetch_spy_daily()
    upsert(con, pulled)
    export_csv(con)
    con.close()

if __name__ == "__main__":
    main()
