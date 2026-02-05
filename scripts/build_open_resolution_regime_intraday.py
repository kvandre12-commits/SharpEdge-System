#!/usr/bin/env python3
"""
Build OPEN resolution regime using intraday bars + true key levels.

Inputs:
- spy_bars_15m (or env INTRADAY_BARS_TABLE)
- liquidity_regime_events.prior_key_low (true key level)

Outputs:
- open_resolution_regime (SQLite)
"""
import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")

NY = ZoneInfo("America/New_York")

# Premarket "initiative flush" knobs (tune later)
PM_RETURN_THRESH = float(os.getenv("PM_RETURN_THRESH", "-0.003"))      # -0.30%
PM_RANGE_RATIO_THRESH = float(os.getenv("PM_RANGE_RATIO_THRESH", "0.0025"))  # 0.25%

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_table(con: sqlite3.Connection) -> None:
    # Table is created by migration; keeping a safe create here is fine too.
    con.execute("""
    CREATE TABLE IF NOT EXISTS open_resolution_regime (
      snapshot_ts TEXT NOT NULL,
      session_date TEXT NOT NULL,
      underlying TEXT NOT NULL,

      pm_open REAL,
      pm_high REAL,
      pm_low REAL,
      pm_close REAL,
      pm_return REAL,
      pm_range REAL,
      pm_range_ratio REAL,
      pm_initiative_flush INTEGER,

      break_level REAL,
      flush_low REAL,

      rth_first_ts TEXT,
      rth_first_open REAL,
      rth_first_high REAL,
      rth_first_low REAL,
      rth_first_close REAL,

      failed_breakdown_open INTEGER,
      accepted_breakdown_open INTEGER,
      open_regime_label TEXT,
      regime_confidence REAL,
      notes TEXT,

      PRIMARY KEY (underlying, session_date)
    )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_open_resolution_regime_date ON open_resolution_regime(underlying, session_date)")
    con.commit()

def parse_ts_utc(ts_text: str) -> datetime:
    # stored like 2025-02-04T14:45:00Z
    return datetime.fromisoformat(ts_text.replace("Z", "+00:00")).astimezone(timezone.utc)

def fetch_sessions(con: sqlite3.Connection) -> List[str]:
    rows = con.execute(
        f"SELECT DISTINCT session_date FROM {BARS_TABLE} WHERE symbol = ? ORDER BY session_date ASC",
        (SYMBOL,),
    ).fetchall()
    return [r[0] for r in rows]

def fetch_bars_for_session(con: sqlite3.Connection, session_date: str) -> List[Dict[str, Any]]:
    rows = con.execute(
        f"""
        SELECT ts, open, high, low, close, volume
        FROM {BARS_TABLE}
        WHERE symbol = ? AND session_date = ?
        ORDER BY ts ASC
        """,
        (SYMBOL, session_date),
    ).fetchall()
    out = []
    for ts, o, h, l, c, v in rows:
        out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return out

PM_UP_RETURN_THRESH = float(os.getenv("PM_UP_RETURN_THRESH", "0.003"))  # +0.30%

def get_true_key_levels(con: sqlite3.Connection, session_date: str) -> Dict[str, Optional[float]]:
    row = con.execute(
        """
        SELECT prior_key_low, prior_key_high
        FROM liquidity_regime_events
        WHERE underlying = ? AND session_date = ?
        """,
        (SYMBOL, session_date),
    ).fetchone()
    if not row:
        return {"prior_key_low": None, "prior_key_high": None}
    return {
        "prior_key_low": float(row[0]) if row[0] is not None else None,
        "prior_key_high": float(row[1]) if row[1] is not None else None,
    }

def first_rth_n_bars(bars: List[Dict[str, Any]], n: int = 2) -> List[Dict[str, Any]]:
    rth = [b for b in bars if in_ny_time(b["ts"], "09:30", "10:00")]
    return rth[:n]
    
def in_ny_time(ts_text: str, start_hhmm: str, end_hhmm: str) -> bool:
    dt_utc = parse_ts_utc(ts_text)
    dt_ny = dt_utc.astimezone(NY)
    t = dt_ny.time()
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    return (t >= datetime(2000,1,1,sh,sm).time()) and (t < datetime(2000,1,1,eh,em).time())

def compute_premarket_stats(bars: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    pm = [b for b in bars if in_ny_time(b["ts"], "04:00", "09:30")]
    if not pm:
        return {"pm_open": None, "pm_high": None, "pm_low": None, "pm_close": None,
                "pm_return": None, "pm_range": None, "pm_range_ratio": None}

    pm_open = float(pm[0]["open"])
    pm_close = float(pm[-1]["close"])
    pm_high = float(max(b["high"] for b in pm))
    pm_low  = float(min(b["low"]  for b in pm))
    pm_range = pm_high - pm_low
    pm_return = (pm_close / pm_open - 1.0) if pm_open else None
    pm_range_ratio = (pm_range / pm_open) if pm_open else None

    return {
        "pm_open": pm_open,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "pm_close": pm_close,
        "pm_return": pm_return,
        "pm_range": pm_range,
        "pm_range_ratio": pm_range_ratio,
    }

def first_rth_bar(bars: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rth = [b for b in bars if in_ny_time(b["ts"], "09:30", "09:45")]
    if not rth:
        return None
    # With 15m bars, there should be exactly one bar. Return the first.
    return rth[0]

def classify(session_date: str,
             pm: Dict[str, Optional[float]],
             keys: Dict[str, Optional[float]],
             rths: List[Dict[str, Any]]) -> Dict[str, Any]:

    notes = []
    pm_return = pm["pm_return"]
    pm_rr = pm["pm_range_ratio"]

    # Setup qualifier: same as before, but direction-aware
    has_range = (pm_rr is not None and pm_rr >= PM_RANGE_RATIO_THRESH)
    flush_down = (pm_return is not None and pm_return <= PM_RETURN_THRESH)
    rip_up     = (pm_return is not None and pm_return >= PM_UP_RETURN_THRESH)

    setup_dir = "NONE"
    if has_range and flush_down:
        setup_dir = "DOWN"
    elif has_range and rip_up:
        setup_dir = "UP"

    if setup_dir == "NONE":
        return {
            "pm_initiative_flush": 0,
            "setup_dir": "NONE",
            "key_source": None,
            "failed_breakdown_open": 0,
            "accepted_breakdown_open": 0,
            "open_regime_label": "NO_SETUP",
            "regime_confidence": 10.0,
            "notes": "no initiative flush/rip setup",
            "break_level": None,
            "flush_low": pm["pm_low"],
        }

    # Choose true key level based on setup direction
    break_level = None
    key_source = None

    if setup_dir == "DOWN":
        break_level = keys.get("prior_key_low")
        key_source = "PRIOR_KEY_LOW"
    else:
        break_level = keys.get("prior_key_high")
        key_source = "PRIOR_KEY_HIGH"

    if break_level is None:
        break_level = pm["pm_open"]
        key_source = "FALLBACK_PM_OPEN"
        notes.append("break_level fallback=pm_open (true key missing)")

    # Need at least the first bar
    if not rths:
        return {
            "pm_initiative_flush": 1,
            "setup_dir": setup_dir,
            "key_source": key_source,
            "failed_breakdown_open": 0,
            "accepted_breakdown_open": 0,
            "open_regime_label": "MISSING_RTH",
            "regime_confidence": 0.0,
            "notes": "missing first RTH bar",
            "break_level": break_level,
            "flush_low": pm["pm_low"],
        }

    b1 = rths[0]
    b2 = rths[1] if len(rths) > 1 else None

    o1, h1, l1, c1 = map(float, (b1["open"], b1["high"], b1["low"], b1["close"]))
    c2 = float(b2["close"]) if b2 else None

    flush_low = pm["pm_low"]
    flush_high = pm["pm_high"]

    # Direction-specific logic
    failed = 0
    accepted = 0
    label = "UNRESOLVED_OPEN"
    conf = 35.0

    if setup_dir == "DOWN":
        swept = (l1 < break_level)
        reclaimed = swept and (c1 > break_level)

        # Acceptance now requires 2 bars (reduces fake opens)
        accept_close_1 = (c1 < break_level)
        accept_close_2 = (c2 is not None and c2 < break_level)

        if reclaimed:
            failed = 1
            label = "FAILED_BREAKDOWN_OPEN"
            conf = 72.0
            notes.append("swept below prior_key_low and closed back above")
        elif accept_close_1 and accept_close_2:
            accepted = 1
            label = "ACCEPTED_BREAKDOWN_OPEN_2BAR"
            conf = 70.0
            notes.append("accepted below prior_key_low for 2 bars")
        elif flush_low is not None and c1 < float(flush_low) and (c2 is None or c2 < float(flush_low)):
            accepted = 1
            label = "ACCEPTED_BREAKDOWN_OPEN_STRONG"
            conf = 75.0
            notes.append("accepted below premarket flush low")
        else:
            notes.append("no reclaim/accept yet (down setup)")

    else:  # UP setup
        swept = (h1 > break_level)
        rejected = swept and (c1 < break_level)

        accept_close_1 = (c1 > break_level)
        accept_close_2 = (c2 is not None and c2 > break_level)

        if rejected:
            failed = 1
            label = "FAILED_BREAKOUT_OPEN"
            conf = 72.0
            notes.append("swept above prior_key_high and closed back below")
        elif accept_close_1 and accept_close_2:
            accepted = 1
            label = "ACCEPTED_BREAKOUT_OPEN_2BAR"
            conf = 70.0
            notes.append("accepted above prior_key_high for 2 bars")
        elif flush_high is not None and c1 > float(flush_high) and (c2 is None or c2 > float(flush_high)):
            accepted = 1
            label = "ACCEPTED_BREAKOUT_OPEN_STRONG"
            conf = 75.0
            notes.append("accepted above premarket high")
        else:
            notes.append("no reject/accept yet (up setup)")

    return {
        "pm_initiative_flush": 1,
        "setup_dir": setup_dir,
        "key_source": key_source,
        "failed_breakdown_open": failed,
        "accepted_breakdown_open": accepted,
        "open_regime_label": label,
        "regime_confidence": float(min(conf, 100.0)),
        "notes": "; ".join(notes),
        "break_level": break_level,
        "flush_low": float(flush_low) if flush_low is not None else None,
    }
                 
def upsert(con: sqlite3.Connection, row: Dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO open_resolution_regime (
          snapshot_ts, session_date, underlying,
          pm_open, pm_high, pm_low, pm_close, pm_return, pm_range, pm_range_ratio, pm_initiative_flush,
          break_level, flush_low,
          rth_first_ts, rth_first_open, rth_first_high, rth_first_low, rth_first_close,
          failed_breakdown_open, accepted_breakdown_open, open_regime_label, regime_confidence, notes
        ) VALUES (
          :snapshot_ts, :session_date, :underlying,
          :pm_open, :pm_high, :pm_low, :pm_close, :pm_return, :pm_range, :pm_range_ratio, :pm_initiative_flush,
          :break_level, :flush_low,
          :rth_first_ts, :rth_first_open, :rth_first_high, :rth_first_low, :rth_first_close,
          :failed_breakdown_open, :accepted_breakdown_open, :open_regime_label, :regime_confidence, :notes
        )
        ON CONFLICT(underlying, session_date) DO UPDATE SET
          snapshot_ts=excluded.snapshot_ts,
          pm_open=excluded.pm_open,
          pm_high=excluded.pm_high,
          pm_low=excluded.pm_low,
          pm_close=excluded.pm_close,
          pm_return=excluded.pm_return,
          pm_range=excluded.pm_range,
          pm_range_ratio=excluded.pm_range_ratio,
          pm_initiative_flush=excluded.pm_initiative_flush,
          break_level=excluded.break_level,
          flush_low=excluded.flush_low,
          rth_first_ts=excluded.rth_first_ts,
          rth_first_open=excluded.rth_first_open,
          rth_first_high=excluded.rth_first_high,
          rth_first_low=excluded.rth_first_low,
          rth_first_close=excluded.rth_first_close,
          failed_breakdown_open=excluded.failed_breakdown_open,
          accepted_breakdown_open=excluded.accepted_breakdown_open,
          open_regime_label=excluded.open_regime_label,
          regime_confidence=excluded.regime_confidence,
          notes=excluded.notes
        """,
        row,
    )

def main():
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con)

        sessions = fetch_sessions(con)
        if not sessions:
            print(f"No sessions found in {BARS_TABLE}.")
            return

        snapshot_ts = iso_utc_now()

        for session_date in sessions:
            bars = fetch_bars_for_session(con, session_date)
            if not bars:
                continue

            pm = compute_premarket_stats(bars)
            keys = get_true_key_levels(con, session_date)
            rths = first_rth_n_bars(bars, n=2)
            cls = classify(session_date, pm, keys, rths)

            row = {
                "snapshot_ts": snapshot_ts,
                "session_date": session_date,
                "underlying": SYMBOL,

                "pm_open": pm["pm_open"],
                "pm_high": pm["pm_high"],
                "pm_low": pm["pm_low"],
                "pm_close": pm["pm_close"],
                "pm_return": pm["pm_return"],
                "pm_range": pm["pm_range"],
                "pm_range_ratio": pm["pm_range_ratio"],
                "pm_initiative_flush": cls["pm_initiative_flush"],

                "break_level": cls["break_level"],
                "flush_low": cls["flush_low"],

                "rth_first_ts": rth["ts"] if rth else None,
                "rth_first_open": float(rth["open"]) if rth else None,
                "rth_first_high": float(rth["high"]) if rth else None,
                "rth_first_low": float(rth["low"]) if rth else None,
                "rth_first_close": float(rth["close"]) if rth else None,

                "failed_breakdown_open": cls["failed_breakdown_open"],
                "accepted_breakdown_open": cls["accepted_breakdown_open"],
                "open_regime_label": cls["open_regime_label"],
                "regime_confidence": cls["regime_confidence"],"setup_dir": cls.get("setup_dir"),
                "key_source": cls.get("key_source"),
                "notes": cls["notes"],
            }

            upsert(con, row)

        con.commit()
        print("OK: open_resolution_regime updated.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
