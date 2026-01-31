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

df.columns = [_clean_col(c) for c in df.columns]

if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    # Example: ('Open','SPY') -> 'open_spy'
    df.columns = [
        "_".join([str(x) for x in col if x is not None and str(x) != ""])
        for col in df.columns.to_list()
    ]
else:
    df.columns = [str(c) for c in df.columns]

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

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
