#!/usr/bin/env python3
"""Build active options surface state for SharpEdge.

Reads the latest options snapshot and positioning row, then emits compact
surface-state metrics for the intraday execution card.
"""

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
DTE_MIN = int(os.getenv("SURFACE_DTE_MIN", "0"))
DTE_MAX = int(os.getenv("SURFACE_DTE_MAX", "3"))
NEAR_PCT = float(os.getenv("SURFACE_NEAR_PCT", "0.01"))
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con, name):
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def columns(con, table):
    if not table_exists(con, table):
        return set()
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def latest_positioning(con):
    if not table_exists(con, "options_positioning_metrics"):
        return None
    return con.execute(
        """
        SELECT *
        FROM options_positioning_metrics
        WHERE underlying=?
        ORDER BY session_date DESC, snapshot_ts DESC
        LIMIT 1
        """,
        (SYMBOL,),
    ).fetchone()


def latest_price_fallback(con):
    """Best-effort spot fallback so observability never kills the pipeline."""
    # Prefer latest intraday close if available.
    bars_table = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
    cols = columns(con, bars_table)
    if cols and "close" in cols:
        ts_col = next((c for c in ("ts", "timestamp", "bar_ts", "datetime", "time") if c in cols), None)
        symbol_filter = ""
        params = []
        if "symbol" in cols:
            symbol_filter = "WHERE symbol=?"
            params.append(SYMBOL)
        order = ts_col or "rowid"
        try:
            row = con.execute(
                f"SELECT close FROM {bars_table} {symbol_filter} ORDER BY {order} DESC LIMIT 1",
                params,
            ).fetchone()
            if row and row[0] is not None and float(row[0]) > 0:
                return float(row[0]), f"fallback:{bars_table}.close"
        except Exception as e:
            print(f"WARNING: intraday spot fallback failed: {e}")

    # Then latest daily close from common daily tables.
    for table in ("spy_daily", "spy_truth_daily", "daily_bars", "features_daily"):
        cols = columns(con, table)
        if not cols or "close" not in cols:
            continue
        date_col = next((c for c in ("date", "session_date", "bar_date") if c in cols), None)
        symbol_filter = ""
        params = []
        if "symbol" in cols:
            symbol_filter = "WHERE symbol=?"
            params.append(SYMBOL)
        order = date_col or "rowid"
        try:
            row = con.execute(
                f"SELECT close FROM {table} {symbol_filter} ORDER BY {order} DESC LIMIT 1",
                params,
            ).fetchone()
            if row and row[0] is not None and float(row[0]) > 0:
                return float(row[0]), f"fallback:{table}.close"
        except Exception as e:
            print(f"WARNING: daily spot fallback failed for {table}: {e}")

    return None, "missing"


def previous_surface(con, session_date, snapshot_ts):
    if not table_exists(con, "options_surface_state"):
        return None
    return con.execute(
        """
        SELECT *
        FROM options_surface_state
        WHERE underlying=? AND session_date=? AND snapshot_ts < ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (SYMBOL, session_date, snapshot_ts),
    ).fetchone()


def ensure_schema(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS options_surface_state (
          snapshot_ts TEXT NOT NULL,
          session_date TEXT NOT NULL,
          underlying TEXT NOT NULL,
          spot REAL,
          active_wall REAL,
          prior_active_wall REAL,
          wall_drift REAL,
          gamma_concentration REAL,
          surface_skew REAL,
          upper_pressure REAL,
          lower_pressure REAL,
          near_spot_density REAL,
          transition_state TEXT,
          row_count INTEGER,
          created_at TEXT NOT NULL,
          PRIMARY KEY (snapshot_ts, underlying)
        )
        """
    )


def load_snapshot_rows(con, snapshot_ts):
    if not table_exists(con, "options_chain_snapshots"):
        return []
    return con.execute(
        """
        SELECT strike, dte, call_oi, put_oi, call_volume, put_volume,
               call_gamma, put_gamma
        FROM options_chain_snapshots
        WHERE underlying=?
          AND snapshot_ts=?
          AND dte BETWEEN ? AND ?
        """,
        (SYMBOL, snapshot_ts, DTE_MIN, DTE_MAX),
    ).fetchall()


def row_pressure(row, spot):
    strike = float(row["strike"])
    dte = int(row["dte"] or 0)
    co = float(row["call_oi"] or 0)
    po = float(row["put_oi"] or 0)
    cv = float(row["call_volume"] or 0)
    pv = float(row["put_volume"] or 0)
    cg = abs(float(row["call_gamma"] or 0))
    pg = abs(float(row["put_gamma"] or 0))

    dist_pct = abs(strike - spot) / spot if spot else 1.0
    distance_weight = math.exp(-dist_pct * 12.0)
    expiry_weight = 1.0 / (dte + 1.0)

    call_pressure = (co + 1.0) * (1.0 + math.log1p(cv)) * (1.0 + cg * 100.0)
    put_pressure = (po + 1.0) * (1.0 + math.log1p(pv)) * (1.0 + pg * 100.0)
    total_pressure = (call_pressure + put_pressure) * distance_weight * expiry_weight
    return strike, total_pressure, dist_pct


def classify_transition(surface_skew, gamma_concentration, wall_drift, near_density, dealer_state):
    if wall_drift is not None and wall_drift > 0.75 and surface_skew > 0.15:
        return "UPSIDE_CHASE_TRANSITION"
    if wall_drift is not None and wall_drift < -0.75 and surface_skew < -0.15:
        return "DOWNSIDE_UNWIND_TRANSITION"
    if gamma_concentration >= 0.55 and near_density >= 0.45:
        return "PIN_COMPRESSION_TRANSITION"
    if dealer_state in ("DEFENSIVE", "UNWIND_RISK") and surface_skew < -0.10:
        return "DEFENSIVE_ROTATION_TRANSITION"
    if abs(surface_skew) < 0.10 and gamma_concentration < 0.35:
        return "BALANCED_AUCTION_TRANSITION"
    return "MIXED_SURFACE_TRANSITION"


def main():
    con = connect()
    ensure_schema(con)

    pos = latest_positioning(con)
    if not pos:
        raise SystemExit("No options_positioning_metrics rows found")

    snapshot_ts = pos["snapshot_ts"]
    session_date = pos["session_date"]
    raw_spot = pos["spot"] if "spot" in pos.keys() else None
    spot = float(raw_spot or 0)
    spot_source = "options_positioning_metrics.spot"
    if spot <= 0:
        fallback_spot, fallback_source = latest_price_fallback(con)
        if fallback_spot and fallback_spot > 0:
            spot = fallback_spot
            spot_source = fallback_source
            print(f"WARNING: latest positioning row missing spot; using {spot_source}={spot}")
        else:
            spot = 1.0
            spot_source = "synthetic:1.0"
            print("WARNING: no spot fallback found; using synthetic spot=1.0 and marking surface degraded")

    dealer_state = pos["dealer_state_hint"] if "dealer_state_hint" in pos.keys() and pos["dealer_state_hint"] else "UNKNOWN"

    rows = load_snapshot_rows(con, snapshot_ts)
    pressure_by = {}
    upper_pressure = 0.0
    lower_pressure = 0.0
    near_pressure = 0.0
    total_pressure = 0.0

    for row in rows:
        strike, pressure, dist_pct = row_pressure(row, spot)
        pressure_by[strike] = pressure_by.get(strike, 0.0) + pressure
        total_pressure += pressure
        if strike >= spot:
            upper_pressure += pressure
        else:
            lower_pressure += pressure
        if dist_pct <= NEAR_PCT:
            near_pressure += pressure

    active_wall = max(pressure_by, key=lambda k: pressure_by[k]) if pressure_by else None
    top_pressures = sorted(pressure_by.values(), reverse=True)[:3]
    gamma_concentration = sum(top_pressures) / total_pressure if total_pressure else 0.0
    surface_skew = (upper_pressure - lower_pressure) / total_pressure if total_pressure else 0.0
    near_density = near_pressure / total_pressure if total_pressure else 0.0

    prev = previous_surface(con, session_date, snapshot_ts)
    prior_wall = float(prev["active_wall"]) if prev and prev["active_wall"] is not None else None
    wall_drift = (float(active_wall) - prior_wall) if active_wall is not None and prior_wall is not None else None

    transition = classify_transition(
        surface_skew,
        gamma_concentration,
        wall_drift,
        near_density,
        dealer_state,
    )
    if not rows or spot_source.startswith("synthetic"):
        transition = "SURFACE_DATA_DEGRADED"

    created_at = datetime.now(timezone.utc).isoformat()

    con.execute(
        """
        INSERT INTO options_surface_state (
          snapshot_ts, session_date, underlying, spot, active_wall,
          prior_active_wall, wall_drift, gamma_concentration, surface_skew,
          upper_pressure, lower_pressure, near_spot_density,
          transition_state, row_count, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(snapshot_ts, underlying) DO UPDATE SET
          session_date=excluded.session_date,
          spot=excluded.spot,
          active_wall=excluded.active_wall,
          prior_active_wall=excluded.prior_active_wall,
          wall_drift=excluded.wall_drift,
          gamma_concentration=excluded.gamma_concentration,
          surface_skew=excluded.surface_skew,
          upper_pressure=excluded.upper_pressure,
          lower_pressure=excluded.lower_pressure,
          near_spot_density=excluded.near_spot_density,
          transition_state=excluded.transition_state,
          row_count=excluded.row_count,
          created_at=excluded.created_at
        """,
        (
            snapshot_ts,
            session_date,
            SYMBOL,
            spot,
            active_wall,
            prior_wall,
            wall_drift,
            gamma_concentration,
            surface_skew,
            upper_pressure,
            lower_pressure,
            near_density,
            transition,
            len(rows),
            created_at,
        ),
    )
    con.commit()
    con.close()

    payload = {
        "snapshot_ts": snapshot_ts,
        "session_date": session_date,
        "underlying": SYMBOL,
        "spot": spot,
        "spot_source": spot_source,
        "active_wall": active_wall,
        "prior_active_wall": prior_wall,
        "wall_drift": wall_drift,
        "gamma_concentration": gamma_concentration,
        "surface_skew": surface_skew,
        "upper_pressure": upper_pressure,
        "lower_pressure": lower_pressure,
        "near_spot_density": near_density,
        "transition_state": transition,
        "dealer_state": dealer_state,
        "row_count": len(rows),
        "created_at": created_at,
    }

    (OUTDIR / "options_surface_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTDIR / "options_surface_state.txt").write_text(
        "\n".join([
            "OPTIONS SURFACE STATE",
            "====================",
            f"session: {session_date}",
            f"spot: {spot}",
            f"spot_source: {spot_source}",
            f"active_wall: {active_wall}",
            f"wall_drift: {wall_drift}",
            f"gamma_concentration: {gamma_concentration:.2f}",
            f"surface_skew: {surface_skew:.2f}",
            f"near_spot_density: {near_density:.2f}",
            f"transition_state: {transition}",
            f"row_count: {len(rows)}",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
