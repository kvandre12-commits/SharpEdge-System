from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import pandas as pd
import yfinance as yf

# If you already set SYMBOL somewhere else, you can remove this default.
SYMBOL = "SPY"


# -------------------------
# Shared helpers
# -------------------------

def _norm_col(c) -> str:
    """Tuple-safe column normalization (handles MultiIndex from yfinance)."""
    if isinstance(c, tuple):
        c = "_".join(str(x) for x in c if x not in (None, ""))
    return str(c).strip().lower().replace(" ", "_")


def _pick(df: pd.DataFrame, *names: str) -> str:
    """Pick the first existing column name from a list of candidates."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing columns. Have: {df.columns.tolist()}")


# -------------------------
# Daily ingest (fixed)
# -------------------------

def fetch_spy_daily(symbol: str | list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV for SPY (or a passed symbol) from yfinance.

    Output columns:
      date, open, high, low, close, volume, symbol, source, ingest_ts
    """
    sym = symbol if symbol is not None else SYMBOL
    sym = sym[0] if isinstance(sym, (list, tuple)) else sym

    df = yf.download(sym, period="2y", interval="1d", auto_adjust=False, progress=False)
    df = df.reset_index()

    # ✅ FIX: normalize columns (this was dead code in the old version)
    df.columns = [_norm_col(c) for c in df.columns]

    date_col  = _pick(df, "date", "datetime")
    open_col  = _pick(df, "open", "spy_open", "open_spy")
    high_col  = _pick(df, "high", "spy_high", "high_spy")
    low_col   = _pick(df, "low",  "spy_low",  "low_spy")
    close_col = _pick(df, "close","spy_close","close_spy")
    vol_col   = _pick(df, "volume","spy_volume","volume_spy")

    # keep date SQL-friendly
    df[date_col] = pd.to_datetime(df[date_col]).dt.date.astype(str)

    out = df[[date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]

    out["symbol"] = sym
    out["source"] = "yfinance"
    out["ingest_ts"] = datetime.now(timezone.utc).isoformat()

    return out


# -------------------------
# Intraday ingest (new)
# -------------------------

@dataclass(frozen=True)
class IntradayConfig:
    interval: str = "5m"           # "1m", "2m", "5m", "15m", "30m", "60m"/"1h", "90m"
    include_prepost: bool = False
    pause_seconds: float = 0.25    # gentle pacing to reduce throttling risk


def _max_chunk_days(interval: str) -> int:
    """
    Practical chunk sizes for Yahoo/yfinance intraday pulls.
    These aren’t strict guarantees, but they keep requests realistic.
    """
    interval = interval.lower()
    if interval == "1m":
        return 7
    if interval in {"2m", "5m", "15m", "30m", "90m"}:
        return 60
    if interval in {"60m", "1h"}:
        return 730
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_spy_intraday(
    start: str | datetime,
    end: str | datetime,
    symbol: str | list[str] | tuple[str, ...] | None = None,
    cfg: IntradayConfig = IntradayConfig(),
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV candles from yfinance in chunks.

    Output columns:
      ts_utc, date, time, open, high, low, close, volume, symbol, source, ingest_ts

    Notes:
    - start/end are required for deterministic ingestion.
    - Output ts_utc is a string for easy SQL ingestion.
    """
    sym = symbol if symbol is not None else SYMBOL
    sym = sym[0] if isinstance(sym, (list, tuple)) else sym

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    max_days = _max_chunk_days(cfg.interval)

    chunks: list[pd.DataFrame] = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + pd.Timedelta(days=max_days), end_dt)

        df = yf.download(
            sym,
            start=cursor,
            end=chunk_end,
            interval=cfg.interval,
            auto_adjust=False,
            progress=False,
            prepost=cfg.include_prepost,
            group_by="column",
            threads=True,
        )

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [_norm_col(c) for c in df.columns]

            # yfinance sometimes calls the timestamp column "datetime", sometimes "date"
            ts_col = "datetime" if "datetime" in df.columns else "date" if "date" in df.columns else None
            if ts_col is None:
                raise KeyError(f"Missing timestamp column. Have: {df.columns.tolist()}")

            open_col  = _pick(df, "open", "spy_open", "open_spy")
            high_col  = _pick(df, "high", "spy_high", "high_spy")
            low_col   = _pick(df, "low",  "spy_low",  "low_spy")
            close_col = _pick(df, "close","spy_close","close_spy")
            vol_col   = _pick(df, "volume","spy_volume","volume_spy")

            # Force timestamps to UTC for consistent keys/joins in SQL
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

            out = pd.DataFrame({
                "ts_utc": ts.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
                "date": ts.dt.strftime("%Y-%m-%d"),
                "time": ts.dt.strftime("%H:%M:%S"),
                "open": pd.to_numeric(df[open_col], errors="coerce"),
                "high": pd.to_numeric(df[high_col], errors="coerce"),
                "low": pd.to_numeric(df[low_col], errors="coerce"),
                "close": pd.to_numeric(df[close_col], errors="coerce"),
                "volume": pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype("int64"),
            })

            out["symbol"] = sym
            out["source"] = "yfinance"
            out["ingest_ts"] = datetime.now(timezone.utc).isoformat()

            # drop bad timestamps + any empty ohlc rows
            out = out[out["ts_utc"].notna()]
            out = out.dropna(subset=["open", "high", "low", "close"])

            chunks.append(out)

        if cfg.pause_seconds:
            time.sleep(cfg.pause_seconds)

        cursor = chunk_end

    if not chunks:
        return pd.DataFrame(columns=[
            "ts_utc","date","time","open","high","low","close","volume","symbol","source","ingest_ts"
        ])

    final = pd.concat(chunks, ignore_index=True)

    # de-dupe in case chunk boundaries overlap
    final = final.drop_duplicates(subset=["symbol", "ts_utc"]).sort_values(["symbol", "ts_utc"])
    final = final.reset_index(drop=True)

    return final


# -------------------------
# Quick sanity run (optional)
# -------------------------

if __name__ == "__main__":
    d = fetch_spy_daily()
    print(d.tail())

    i = fetch_spy_intraday("2026-01-27", "2026-01-31", cfg=IntradayConfig(interval="5m"))
    print(i.head())
    print(i.tail())
