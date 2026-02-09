#!/usr/bin/env python3
import os
import sqlite3
import numpy 
from datetime import datetime, timezone
from typing import Optional, Tuple, List

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
UNDERLYING = os.getenv("SYMBOL", "SPY")  # keep consistent with your pipeline
DTE_MIN = int(os.getenv("DTE_MIN", "0"))
DTE_MAX = int(os.getenv("DTE_MAX", "3"))

# If provided, only aggregate this snapshot_ts (ISO8601). Otherwise auto-detect new ones.
ONLY_SNAPSHOT_TS = os.getenv("SNAPSHOT_TS", "").strip()

# Optional: compute a crude dealer state hint
COMPUTE_STATE = os.getenv("COMPUTE_DEALER_STATE", "1").strip() == "1"
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))  # 0.25% default

def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def table_exists(con: sqlite3.Connection, name: str) -> bool:
    r = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return r is not None

def column_names(con: sqlite3.Connection, table: str) -> set:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    # row: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in rows}

def pick_daily_table(con: sqlite3.Connection) -> Optional[str]:
    # Prefer bars_daily (your current code), then fall back to other common tables.
    candidates = ["bars_daily", "truth_daily", "ohlc_daily", "spy_daily"]
    for t in candidates:
        if table_exists(con, t):
            cols = column_names(con, t)
            if {"date", "symbol", "close"}.issubset(cols):
                return t
    return None

def get_spot_for_session(con: sqlite3.Connection, session_date: str, underlying: str) -> Optional[float]:
    # Uses daily close as "spot" proxy. Later you can switch to bars_15m or 1m.
    t = pick_daily_table(con)
    if not t:
        return None
    row = con.execute(
        f"""
        SELECT close
        FROM {t}
        WHERE symbol = ? AND date = ?
        LIMIT 1
        """,
        (underlying, session_date),
    ).fetchone()
    if not row:
        return None
    try:
        return float(row[0]) if row[0] is not None else None
    except Exception:
        return None

def compute_flip_strike(strikes: List[float], net_gex: List[float], spot: float) -> Optional[float]:
    if spot is None or not np.isfinite(spot) or len(strikes) < 2:
        return None

    s = np.array(strikes, dtype=float)
    g = np.array(net_gex, dtype=float)

    # sort by strike
    order = np.argsort(s)
    s = s[order]
    g = g[order]

    # first bracket around spot
    idx = np.searchsorted(s, spot)
    candidates = []

    if 0 < idx < len(s):
        candidates.append((idx - 1, idx))

    # search outward for nearest sign change
    for r in range(1, len(s)):
        lo = idx - r
        hi = idx + r
        if 0 <= lo < len(s) - 1:
            candidates.append((lo, lo + 1))
        if 1 <= hi < len(s):
            candidates.append((hi - 1, hi))

    seen = set()
    for a, b in candidates:
        if (a, b) in seen:
            continue
        seen.add((a, b))

        g1, g2 = g[a], g[b]
        if not (np.isfinite(g1) and np.isfinite(g2)):
            continue

        # exact hits
        if g1 == 0:
            return float(s[a])
        if g2 == 0:
            return float(s[b])

        # need a sign change
        if np.sign(g1) == np.sign(g2):
            continue

        x1, x2 = s[a], s[b]
        if g2 == g1:
            return float((x1 + x2) / 2.0)

        # linear interpolation to g=0
        x0 = x1 + (0.0 - g1) * (x2 - x1) / (g2 - g1)
        return float(x0)

    return None

def ensure_metrics_table(con: sqlite3.Connection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_positioning_metrics (
          metrics_id        INTEGER PRIMARY KEY AUTOINCREMENT,
          snapshot_ts       TEXT NOT NULL,      -- ISO8601 UTC
          session_date      TEXT NOT NULL,      -- YYYY-MM-DD
          underlying        TEXT NOT NULL DEFAULT 'SPY',

          dte_min           INTEGER NOT NULL DEFAULT 0,
          dte_max           INTEGER NOT NULL DEFAULT 3,

          spot              REAL,
          atm_strike        REAL,

          max_total_oi_strike  REAL,
          max_call_oi_strike   REAL,
          max_put_oi_strike    REAL,

          total_call_oi     INTEGER,
          total_put_oi      INTEGER,
          pcr_oi            REAL,

          total_call_vol    INTEGER,
          total_put_vol     INTEGER,
          pcr_vol           REAL,

          gamma_proxy       REAL,
          dealer_state_hint TEXT,

          UNIQUE (snapshot_ts, underlying, dte_min, dte_max)
        );
        """
    )

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_opt_metrics_session
          ON options_positioning_metrics (session_date, snapshot_ts);
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_opt_metrics_horizon
          ON options_positioning_metrics (session_date, dte_min, dte_max, snapshot_ts);
        """
    )
    con.commit()

def get_spot_for_session(con: sqlite3.Connection, session_date: str, underlying: str) -> Optional[float]:
    # Uses daily close as "spot" proxy. Later you can switch to bars_15m or 1m.
    row = con.execute(
        """
        SELECT close
        FROM bars_daily
        WHERE symbol = ? AND date = ?
        LIMIT 1
        """,
        (underlying, session_date),
    ).fetchone()
    if not row:
        return None
    try:
        return float(row[0]) if row[0] is not None else None
    except Exception:
        return None

def pick_new_snapshot_ts(con: sqlite3.Connection) -> List[str]:
    if ONLY_SNAPSHOT_TS:
        return [ONLY_SNAPSHOT_TS]

    # Find snapshot_ts present in raw snapshots but not yet aggregated for (underlying, dte_min, dte_max)
    rows = con.execute(
        """
        SELECT DISTINCT s.snapshot_ts
        FROM options_chain_snapshots s
        LEFT JOIN options_positioning_metrics m
          ON m.snapshot_ts = s.snapshot_ts
         AND m.underlying = s.underlying
         AND m.dte_min = ?
         AND m.dte_max = ?
        WHERE s.underlying = ?
          AND m.metrics_id IS NULL
        ORDER BY s.snapshot_ts ASC
        """,
        (DTE_MIN, DTE_MAX, UNDERLYING),
    ).fetchall()
    return [r[0] for r in rows]

def compute_metrics_for_snapshot(con: sqlite3.Connection, snapshot_ts: str) -> Optional[dict]:
    # Pull scoped strikes for this snapshot/horizon
    cols = column_names(con, "options_chain_snapshots")
    has_gamma = all(c in cols for c in ["call_gamma", "put_gamma"])
    # Greedy select only what we need
    select_cols = [
        "snapshot_ts", "session_date", "underlying", "expiry_date", "dte", "strike",
        "call_oi", "put_oi", "call_volume", "put_volume",
    ]
    if has_gamma:
        select_cols += ["call_gamma", "put_gamma"]

    q = f"""
    SELECT {", ".join(select_cols)}
    FROM options_chain_snapshots
    WHERE underlying = ?
      AND snapshot_ts = ?
      AND dte BETWEEN ? AND ?
    """

    rows = con.execute(q, (UNDERLYING, snapshot_ts, DTE_MIN, DTE_MAX)).fetchall()
    if not rows:
        return None

    # Unpack with indices
    # Build a lightweight list of dict-ish tuples
    # indices:
    base_len = 10  # up to put_volume
    idx = {name: i for i, name in enumerate(select_cols)}

    # session_date should be consistent within snapshot_ts; take first non-null
    session_date = None
    for r in rows:
        if r[idx["session_date"]]:
            session_date = str(r[idx["session_date"]])
            break
    if session_date is None:
        return None

    spot = get_spot_for_session(con, session_date, UNDERLYING)

    # Totals
    total_call_oi = sum(int(r[idx["call_oi"]] or 0) for r in rows)
    total_put_oi  = sum(int(r[idx["put_oi"]] or 0) for r in rows)
    total_call_vol = sum(int(r[idx["call_volume"]] or 0) for r in rows)
    total_put_vol  = sum(int(r[idx["put_volume"]] or 0) for r in rows)

    pcr_oi = (float(total_put_oi) / float(total_call_oi)) if total_call_oi else None
    pcr_vol = (float(total_put_vol) / float(total_call_vol)) if total_call_vol else None

    # Walls
    # tie-breaker: smaller strike first for determinism
    def strike_key(r):
        return float(r[idx["strike"]])

    max_total_row = max(rows, key=lambda r: (int(r[idx["call_oi"]] or 0) + int(r[idx["put_oi"]] or 0), -strike_key(r)))
    max_call_row  = max(rows, key=lambda r: (int(r[idx["call_oi"]] or 0), -strike_key(r)))
    max_put_row   = max(rows, key=lambda r: (int(r[idx["put_oi"]] or 0),  -strike_key(r)))

    max_total_oi_strike = float(max_total_row[idx["strike"]])
    max_call_oi_strike  = float(max_call_row[idx["strike"]])
    max_put_oi_strike   = float(max_put_row[idx["strike"]])

    # ATM strike (nearest strike to spot) — only if spot exists
    atm_strike = None
    if spot is not None:
        atm_row = min(rows, key=lambda r: abs(float(r[idx["strike"]]) - float(spot)))
        atm_strike = float(atm_row[idx["strike"]])

    # Gamma proxy (optional)
    gamma_proxy = None
    if has_gamma and spot is not None:
        gsum = 0.0
        for r in rows:
            co = float(r[idx["call_oi"]] or 0)
            po = float(r[idx["put_oi"]] or 0)
            cg = float(r[idx["call_gamma"]] or 0.0)
            pg = float(r[idx["put_gamma"]] or 0.0)
            gsum += (cg * co) - (pg * po)
        gamma_proxy = gsum * (float(spot) ** 2) * 0.01

    # Dealer state hint (simple heuristic, optional)
    dealer_state_hint = None
    if COMPUTE_STATE and spot is not None:
        dist_pct = abs(float(spot) - float(max_total_oi_strike)) / float(spot)
        if dist_pct <= PIN_THRESH_PCT:
            dealer_state_hint = "pin"
        elif gamma_proxy is None:
            dealer_state_hint = "unknown"
        else:
            # crude: negative net gamma => chase-y; positive => unwind-y
            dealer_state_hint = "chase" if gamma_proxy < 0 else "unwind"

    return {
        "snapshot_ts": snapshot_ts,
        "session_date": session_date,
        "underlying": UNDERLYING,
        "dte_min": DTE_MIN,
        "dte_max": DTE_MAX,
        "spot": spot,
        "atm_strike": atm_strike,
        "max_total_oi_strike": max_total_oi_strike,
        "max_call_oi_strike": max_call_oi_strike,
        "max_put_oi_strike": max_put_oi_strike,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr_oi": pcr_oi,
        "total_call_vol": total_call_vol,
        "total_put_vol": total_put_vol,
        "pcr_vol": pcr_vol,
        "gamma_proxy": gamma_proxy,
        "dealer_state_hint": dealer_state_hint,
    }

def upsert_metrics(con: sqlite3.Connection, m: dict):
    con.execute(
        """
        INSERT INTO options_positioning_metrics (
          snapshot_ts, session_date, underlying,
          dte_min, dte_max,
          spot, atm_strike,
          max_total_oi_strike, max_call_oi_strike, max_put_oi_strike,
          total_call_oi, total_put_oi, pcr_oi,
          total_call_vol, total_put_vol, pcr_vol,
          gamma_proxy, dealer_state_hint
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_ts, underlying, dte_min, dte_max) DO UPDATE SET
          session_date        = excluded.session_date,
          spot                = excluded.spot,
          atm_strike          = excluded.atm_strike,
          max_total_oi_strike = excluded.max_total_oi_strike,
          max_call_oi_strike  = excluded.max_call_oi_strike,
          max_put_oi_strike   = excluded.max_put_oi_strike,
          total_call_oi       = excluded.total_call_oi,
          total_put_oi        = excluded.total_put_oi,
          pcr_oi              = excluded.pcr_oi,
          total_call_vol      = excluded.total_call_vol,
          total_put_vol       = excluded.total_put_vol,
          pcr_vol             = excluded.pcr_vol,
          gamma_proxy         = excluded.gamma_proxy,
          dealer_state_hint   = excluded.dealer_state_hint
        """,
        (
            m["snapshot_ts"], m["session_date"], m["underlying"],
            m["dte_min"], m["dte_max"],
            m["spot"], m["atm_strike"],
            m["max_total_oi_strike"], m["max_call_oi_strike"], m["max_put_oi_strike"],
            m["total_call_oi"], m["total_put_oi"], m["pcr_oi"],
            m["total_call_vol"], m["total_put_vol"], m["pcr_vol"],
            m["gamma_proxy"], m["dealer_state_hint"],
        ),
    )

def main():
    con = connect(DB_PATH)
    try:
        if not table_exists(con, "options_chain_snapshots"):
            raise RuntimeError(
                "Missing table options_chain_snapshots. Create/ingest raw options snapshots first."
            )

        ensure_metrics_table(con)

        snapshot_list = pick_new_snapshot_ts(con)
        if not snapshot_list:
            print(f"OK: no new snapshots to aggregate for {UNDERLYING} DTE[{DTE_MIN},{DTE_MAX}]")
            return

        n_ok = 0
        n_skip = 0

        for snap in snapshot_list:
            m = compute_metrics_for_snapshot(con, snap)
            if not m:
                n_skip += 1
                continue
            upsert_metrics(con, m)
            n_ok += 1

        con.commit()
        print(f"OK: aggregated metrics rows={n_ok} skipped={n_skip} for {UNDERLYING} DTE[{DTE_MIN},{DTE_MAX}]")

    finally:
        con.close()

if __name__ == "__main__":
    main()
