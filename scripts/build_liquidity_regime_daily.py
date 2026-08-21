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

try:
    from scripts.utils.pipeline_state import write_state
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import write_state

import sys as _sys

# Canonical auction classifier — SINGLE SOURCE OF TRUTH shared with the live
# cockpit (cockpit/auction_regime.py). We import it here so the historical
# backtest table and the live cockpit read can never diverge. Hard-fail if the
# module cannot be located rather than silently falling back to stale logic.
_COCKPIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cockpit")
if _COCKPIT_DIR not in _sys.path:
    _sys.path.insert(0, _COCKPIT_DIR)
from auction_regime import (  # noqa: E402
    classify_regime,
    compute_true_range,
    rolling_sma,
)

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
    """
    Works for:
    - daily bars (date)
    - intraday bars (session_date)
    """
    # detect which date column exists
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({bars_table})").fetchall()]

    if "session_date" in cols:
        date_col = "session_date"
    elif "date" in cols:
        date_col = "date"
    else:
        raise RuntimeError(f"{bars_table} has no date/session_date column")

    q = f"""
    SELECT
      {date_col} AS session_date,
      open  AS session_open,
      high  AS session_high,
      low   AS session_low,
      close AS session_close
    FROM {bars_table}
    WHERE symbol = ?
    ORDER BY {date_col} ASC
    """

    rows = con.execute(q, (symbol,)).fetchall()
    cols = [c[0] for c in con.execute(q, (symbol,)).description]
    return [dict(zip(cols, r)) for r in rows]


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


def output_state(con: sqlite3.Connection) -> dict:
    row = con.execute(
        """
        SELECT COUNT(*), MIN(session_date), MAX(session_date), MAX(snapshot_ts)
        FROM liquidity_regime_events
        WHERE underlying=?
        """,
        (SYMBOL,),
    ).fetchone()
    return {
        "rows": row[0] or 0,
        "earliest_session_date": row[1],
        "latest_session_date": row[2],
        "latest_snapshot_ts": row[3],
    }


def main():
    # ---- parse args FIRST (outside try) ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars-table", default="bars_daily")
    args = ap.parse_args()
    bars_table = args.bars_table

    # ---- then do DB work ----
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)

        before = output_state(con)
        days = fetch_bars(con, SYMBOL, bars_table)
        if len(days) < ATR_LOOKBACK + 1:
            print(f"Not enough bars for ATR{ATR_LOOKBACK}. Have {len(days)}. Still writing when possible.")

        # TR + ATR series
        trs: List[float] = []
        prev_close: Optional[float] = None
        for d in days:
            tr = compute_true_range(
                prev_close,
                float(d["session_high"]),
                float(d["session_low"]),
            )
            trs.append(tr)
            prev_close = float(d["session_close"])

        atrs = rolling_sma(trs, ATR_LOOKBACK)
        snapshot_ts = iso_utc_now()

        wrote = 0
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
            wrote += 1

        con.commit()
        after = output_state(con)
        write_state(
            "liquidity_regime",
            {
                "symbol": SYMBOL,
                "bars_table": bars_table,
                "input_rows": len(days),
                "upserted_rows": wrote,
                "before": before,
                "after": after,
            },
        )
        print(f"OK: liquidity_regime_events updated from {bars_table}. upserted={wrote} latest={after.get('latest_session_date')}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
