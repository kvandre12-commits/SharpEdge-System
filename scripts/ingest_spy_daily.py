import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOL = "SPY"
DB_PATH = "data/spy_truth.db"
SOURCE = "yfinance"


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    return con


def ensure_truth_table(con: sqlite3.Connection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bars_daily (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            ingest_ts TEXT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    con.commit()


def fetch_spy_daily(period: str = "2y") -> pd.DataFrame:
    df = yf.download(SYMBOL, period=period, interval="1d", auto_adjust=False)

    # yfinance can return MultiIndex columns sometimes; flatten them safely.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    # Normalize column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # yfinance usually uses 'date' for daily; sometimes 'datetime'
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["symbol"] = SYMBOL
    out["source"] = SOURCE
    out["ingest_ts"] = datetime.now(timezone.utc).isoformat()

    # Basic truth sanity: drop rows with missing OHLC
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return out


def upsert_truth(con: sqlite3.Connection, truth: pd.DataFrame):
    ensure_truth_table(con)

    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "source", "ingest_ts"]
    rows = truth[cols].to_records(index=False).tolist()

    con.executemany(
        """
        INSERT INTO bars_daily (date, symbol, open, high, low, close, volume, source, ingest_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source,
            ingest_ts=excluded.ingest_ts
        """,
        rows,
    )
    con.commit()


def write_truth_csv(con: sqlite3.Connection):
    os.makedirs("outputs", exist_ok=True)
    q = """
    SELECT date, symbol, open, high, low, close, volume, source, ingest_ts
    FROM bars_daily
    WHERE symbol = ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(q, con, params=(SYMBOL,))
    path = "outputs/spy_truth_daily.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def main():
    con = connect(DB_PATH)
    try:
        truth = fetch_spy_daily(period="2y")
        upsert_truth(con, truth)
        write_truth_csv(con)

        # Quick visibility in logs
        last = truth.tail(1)
        print("OK: truth ingested. Last row:")
        print(last.to_string(index=False))
    finally:
        con.close()


if __name__ == "__main__":
    main()
