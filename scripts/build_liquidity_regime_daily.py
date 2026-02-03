#!/usr/bin/env python3
"""
Build daily liquidity regime events from DAILY bars (bars_daily).
Writes to liquidity_regime_events in SQLite.

Regimes:
- FAILED_BREAKDOWN: low < prior_low and close > prior_low and TR/ATR >= threshold
- FAILED_BREAKOUT:  high > prior_high and close < prior_high and TR/ATR >= threshold
- CLEAN_BREAKOUT / CLEAN_BREAKDOWN
- RANGE_COMPRESSION
"""
import argparse
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

# thresholds (same knobs as your intraday version)
FAILED_MIN_RANGE_ATR = float(os.getenv("FAILED_MIN_RANGE_ATR", "1.25"))
CLEAN_MIN_RANGE_ATR = float(os.getenv("CLEAN_MIN_RANGE_ATR", "1.00"))
COMPRESSION_MAX_RANGE_ATR = float(os.getenv("COMPRESSION_MAX_RANGE_ATR", "0.75"))

ATR_LOOKBACK = int(os.getenv("ATR_LOOKBACK", "14"))


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS liquidity_regime_events (
          snapshot_ts TEXT NOT NULL,
          session_date TEXT NOT NULL,
          underlying TEXT NOT NULL,

          prior_key_high REAL,
          prior_key_low  REAL,

          session_open  REAL,
          session_high  REAL,
          session_low   REAL,
          session_close REAL,

          true_range REAL,
          atr_14 REAL,
          range_atr_ratio REAL,

          broke_above_high INTEGER,
          broke_below_low  INTEGER,

          failed_breakout INTEGER,
          failed_breakdown INTEGER,

          reclaimed_level INTEGER,
          rejected_level INTEGER,

          regime_type TEXT,
          regime_confidence REAL,
          notes TEXT,

          PRIMARY KEY (underlying, session_date)
        )
        """
    )
    con.commit()


def fetch_bars(con: sqlite3.Connection, symbol: str, bars_table: str) -> List[Dict]:
    q = f"""
    SELECT
      date AS session_date,
      open  AS session_open,
      high  AS session_high,
      low   AS session_low,
      close AS session_close
    FROM {bars_table}
    WHERE symbol = ?
    ORDER BY date ASC
    """
    rows = con.execute(q, (symbol,)).fetchall()
    cols = [c[0] for c in con.execute(q, (symbol,)).description]
    return [dict(zip(cols, r)) for r in rows]


def compute_true_range(prev_close: Optional[float], high: float, low: float) -> float:
    if prev_close is None:
        return float(high - low)
    return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))


def rolling_sma(values: List[float], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out


def classify_regime(
    prior_high: float,
    prior_low: float,
    o: float, h: float, l: float, c: float,
    tr: float, atr: Optional[float]
) -> Tuple[str, Dict[str, int], float, str]:
    flags = {
        "broke_above_high": int(h > prior_high),
        "broke_below_low": int(l < prior_low),
        "failed_breakout": 0,
        "failed_breakdown": 0,
        "reclaimed_level": 0,
        "rejected_level": 0,
    }

    if atr is None or atr <= 0:
        return ("UNCLASSIFIED", flags, 0.0, "ATR unavailable; need lookback sessions")

    ratio = tr / atr

    broke_above = (h > prior_high)
    broke_below = (l < prior_low)
    close_back_below_prior_high = (c < prior_high)
    close_back_above_prior_low = (c > prior_low)

    reclaimed = broke_below and close_back_above_prior_low
    rejected = broke_above and close_back_below_prior_high

    flags["reclaimed_level"] = int(reclaimed)
    flags["rejected_level"] = int(rejected)

    notes = []
    confidence = 0.0

    # FAILED regimes
    if reclaimed and ratio >= FAILED_MIN_RANGE_ATR:
        flags["failed_breakdown"] = 1
        confidence = 60.0
        if ratio >= 1.5:
            confidence += 15.0
        if abs(c - prior_low) / max(1e-6, atr) >= 0.25:
            confidence += 10.0
        notes.append("swept prior low and reclaimed")
        return ("FAILED_BREAKDOWN", flags, min(confidence, 100.0), "; ".join(notes))

    if rejected and ratio >= FAILED_MIN_RANGE_ATR:
        flags["failed_breakout"] = 1
        confidence = 60.0
        if ratio >= 1.5:
            confidence += 15.0
        if abs(prior_high - c) / max(1e-6, atr) >= 0.25:
            confidence += 10.0
        notes.append("swept prior high and rejected")
        return ("FAILED_BREAKOUT", flags, min(confidence, 100.0), "; ".join(notes))

    # CLEAN continuation
    if broke_above and (c > prior_high) and ratio >= CLEAN_MIN_RANGE_ATR:
        notes.append("broke and held above prior high")
        return ("CLEAN_BREAKOUT", flags, 50.0 + min((ratio - 1.0) * 20.0, 30.0), "; ".join(notes))

    if broke_below and (c < prior_low) and ratio >= CLEAN_MIN_RANGE_ATR:
        notes.append("broke and held below prior low")
        return ("CLEAN_BREAKDOWN", flags, 50.0 + min((ratio - 1.0) * 20.0, 30.0), "; ".join(notes))

    # COMPRESSION
    if ratio <= COMPRESSION_MAX_RANGE_ATR and (not broke_above) and (not broke_below):
        notes.append("range compression")
        return ("RANGE_COMPRESSION", flags, 40.0, "; ".join(notes))

    return ("UNCLASSIFIED", flags, 20.0, "no strong regime match")


def upsert(con: sqlite3.Connection, event: Dict) -> None:
    con.execute(
        """
        INSERT INTO liquidity_regime_events (
          snapshot_ts, session_date, underlying,
          prior_key_high, prior_key_low,
          session_open, session_high, session_low, session_close,
          true_range, atr_14, range_atr_ratio,
          broke_above_high, broke_below_low,
          failed_breakout, failed_breakdown,
          reclaimed_level, rejected_level,
          regime_type, regime_confidence, notes
        ) VALUES (
          :snapshot_ts, :session_date, :underlying,
          :prior_key_high, :prior_key_low,
          :session_open, :session_high, :session_low, :session_close,
          :true_range, :atr_14, :range_atr_ratio,
          :broke_above_high, :broke_below_low,
          :failed_breakout, :failed_breakdown,
          :reclaimed_level, :rejected_level,
          :regime_type, :regime_confidence, :notes
        )
        ON CONFLICT(underlying, session_date) DO UPDATE SET
          snapshot_ts=excluded.snapshot_ts,
          prior_key_high=excluded.prior_key_high,
          prior_key_low=excluded.prior_key_low,
          session_open=excluded.session_open,
          session_high=excluded.session_high,
          session_low=excluded.session_low,
          session_close=excluded.session_close,
          true_range=excluded.true_range,
          atr_14=excluded.atr_14,
          range_atr_ratio=excluded.range_atr_ratio,
          broke_above_high=excluded.broke_above_high,
          broke_below_low=excluded.broke_below_low,
          failed_breakout=excluded.failed_breakout,
          failed_breakdown=excluded.failed_breakdown,
          reclaimed_level=excluded.reclaimed_level,
          rejected_level=excluded.rejected_level,
          regime_type=excluded.regime_type,
          regime_confidence=excluded.regime_confidence,
          notes=excluded.notes
        """,
        event,
    )


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)
        ap = argparse.ArgumentParser()
ap.add_argument("--bars-table", default="bars_daily")
args = ap.parse_args()

days = fetch_bars(con, SYMBOL, args.bars_table)
        if len(days) < ATR_LOOKBACK + 1:
            print(f"Not enough daily bars for ATR{ATR_LOOKBACK}. Have {len(days)}. Still writing when possible.")

        # TR + ATR series
        trs: List[float] = []
        prev_close: Optional[float] = None
        for d in days:
            tr = compute_true_range(prev_close, float(d["session_high"]), float(d["session_low"]))
            trs.append(tr)
            prev_close = float(d["session_close"])

        atrs = rolling_sma(trs, ATR_LOOKBACK)
        snapshot_ts = iso_utc_now()

        for i in range(1, len(days)):
            cur = days[i]
            prev = days[i - 1]

            prior_high = float(prev["session_high"])
            prior_low = float(prev["session_low"])

            o = float(cur["session_open"])
            h = float(cur["session_high"])
            l = float(cur["session_low"])
            c = float(cur["session_close"])

            tr = float(trs[i])
            atr = atrs[i]
            ratio = (tr / atr) if (atr is not None and atr > 0) else None

            regime_type, flags, conf, notes = classify_regime(
                prior_high, prior_low, o, h, l, c, tr, atr
            )

            event = {
                "snapshot_ts": snapshot_ts,
                "session_date": cur["session_date"],
                "underlying": SYMBOL,

                "prior_key_high": prior_high,
                "prior_key_low": prior_low,

                "session_open": o,
                "session_high": h,
                "session_low": l,
                "session_close": c,

                "true_range": tr,
                "atr_14": float(atr) if atr is not None else None,
                "range_atr_ratio": float(ratio) if ratio is not None else None,

                "broke_above_high": flags["broke_above_high"],
                "broke_below_low": flags["broke_below_low"],

                "failed_breakout": flags["failed_breakout"],
                "failed_breakdown": flags["failed_breakdown"],

                "reclaimed_level": flags["reclaimed_level"],
                "rejected_level": flags["rejected_level"],

                "regime_type": regime_type,
                "regime_confidence": float(conf),
                "notes": notes,
            }

            upsert(con, event)

        con.commit()
        print("OK: liquidity_regime_events updated from bars_daily.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
