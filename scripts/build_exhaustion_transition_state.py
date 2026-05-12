#!/usr/bin/env python3
"""Build exhaustion + auction transition state for SharpEdge.

This script turns intraday bars plus options surface context into a compact
state-machine layer. It is designed for readable execution cards, not raw
indicator overload.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")
BARS_TABLE = os.getenv("INTRADAY_BARS_TABLE", "spy_bars_15m")
LOOKBACK = int(os.getenv("EXHAUSTION_LOOKBACK", "12"))
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def ensure_schema(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS exhaustion_transition_state (
          session_date TEXT NOT NULL,
          asof_ts TEXT NOT NULL,
          underlying TEXT NOT NULL,
          exhaustion_score REAL,
          expansion_efficiency REAL,
          upper_wick_ratio REAL,
          lower_wick_ratio REAL,
          momentum_decay REAL,
          failed_acceptance_count INTEGER,
          exhaustion_state TEXT,
          transition_state TEXT,
          participation_quality TEXT,
          note TEXT,
          created_at TEXT NOT NULL,
          PRIMARY KEY (session_date, asof_ts, underlying)
        )
        """
    )


def load_bars(con):
    if not table_exists(con, BARS_TABLE):
        return []
    cols = columns(con, BARS_TABLE)
    ts_col = pick_col(cols, ["ts", "timestamp", "bar_ts", "datetime", "time"])
    if not ts_col:
        raise SystemExit(f"No timestamp column found in {BARS_TABLE}")
    symbol_filter = ""
    params = []
    if "symbol" in cols:
        symbol_filter = "WHERE symbol=?"
        params.append(SYMBOL)
    q = f"""
        SELECT {ts_col} AS ts, open, high, low, close, volume
        FROM {BARS_TABLE}
        {symbol_filter}
        ORDER BY {ts_col} DESC
        LIMIT ?
    """
    params.append(max(LOOKBACK, 6))
    rows = con.execute(q, params).fetchall()
    return list(reversed(rows))


def latest_surface(con):
    if not table_exists(con, "options_surface_state"):
        return None
    return con.execute(
        """
        SELECT * FROM options_surface_state
        WHERE underlying=?
        ORDER BY session_date DESC, snapshot_ts DESC
        LIMIT 1
        """,
        (SYMBOL,),
    ).fetchone()


def prior_state(con, session_date, asof_ts):
    if not table_exists(con, "exhaustion_transition_state"):
        return None
    return con.execute(
        """
        SELECT * FROM exhaustion_transition_state
        WHERE underlying=? AND session_date=? AND asof_ts < ?
        ORDER BY asof_ts DESC
        LIMIT 1
        """,
        (SYMBOL, session_date, asof_ts),
    ).fetchone()


def safe_range(bar):
    return max(float(bar["high"] or 0) - float(bar["low"] or 0), 1e-9)


def candle_metrics(bar):
    o = float(bar["open"] or 0)
    h = float(bar["high"] or 0)
    l = float(bar["low"] or 0)
    c = float(bar["close"] or 0)
    r = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = max(h - max(o, c), 0.0)
    lower_wick = max(min(o, c) - l, 0.0)
    return {
        "range": r,
        "body_ratio": body / r,
        "upper_wick_ratio": upper_wick / r,
        "lower_wick_ratio": lower_wick / r,
        "direction": 1 if c > o else (-1 if c < o else 0),
    }


def failed_acceptance_count(bars):
    if len(bars) < 4:
        return 0
    count = 0
    for i in range(2, len(bars)):
        prev_high = max(float(bars[j]["high"] or 0) for j in range(max(0, i - 3), i))
        h = float(bars[i]["high"] or 0)
        c = float(bars[i]["close"] or 0)
        if h > prev_high and c < prev_high:
            count += 1
    return count


def classify_exhaustion(score, fail_count, upper_wick, momentum_decay):
    if score >= 0.75 and fail_count >= 2:
        return "FAILED_CONTINUATION"
    if score >= 0.65 and upper_wick >= 0.45:
        return "BUYER_EXHAUSTION"
    if score >= 0.60 and momentum_decay < 0.80:
        return "MOMENTUM_DECAY"
    if score <= 0.35:
        return "HEALTHY_EXPANSION"
    return "BALANCED_COMPRESSION"


def classify_participation(exhaustion_state, volume_ratio, body_eff):
    if exhaustion_state == "HEALTHY_EXPANSION" and volume_ratio >= 1.0 and body_eff >= 0.55:
        return "HEALTHY_TREND"
    if exhaustion_state in ("BUYER_EXHAUSTION", "FAILED_CONTINUATION"):
        return "WEAK_CONTINUATION"
    if volume_ratio < 0.75 and body_eff < 0.45:
        return "EMPTY_BREAKOUT_RISK"
    return "ROTATIONAL"


def classify_transition(exhaustion_state, prior, surface):
    surface_transition = surface["transition_state"] if surface and "transition_state" in surface.keys() else None
    prior_ex = prior["exhaustion_state"] if prior else None

    if prior_ex == "HEALTHY_EXPANSION" and exhaustion_state in ("MOMENTUM_DECAY", "BUYER_EXHAUSTION"):
        return "EXPANSION_TO_EXHAUSTION"
    if exhaustion_state == "FAILED_CONTINUATION":
        return "FAILED_ACCEPTANCE_TO_RESPONSIVE_SELLING"
    if surface_transition == "PIN_COMPRESSION_TRANSITION" and exhaustion_state == "BALANCED_COMPRESSION":
        return "PIN_TO_WAIT"
    if surface_transition == "UPSIDE_CHASE_TRANSITION" and exhaustion_state == "HEALTHY_EXPANSION":
        return "CHASE_CONTINUATION"
    if surface_transition == "DOWNSIDE_UNWIND_TRANSITION":
        return "SURFACE_UNWIND_RISK"
    return surface_transition or "NO_CLEAR_TRANSITION"


def make_note(exhaustion_state, transition_state, participation):
    if transition_state == "EXPANSION_TO_EXHAUSTION":
        return "Expansion is losing efficiency. Do not chase highs without fresh acceptance."
    if transition_state == "FAILED_ACCEPTANCE_TO_RESPONSIVE_SELLING":
        return "Failed acceptance risk is active. Watch for trapped breakout buyers and responsive selling."
    if transition_state == "SURFACE_UNWIND_RISK":
        return "Dealer surface is leaning unwind. Size down and require clean reclaim proof."
    if exhaustion_state == "HEALTHY_EXPANSION":
        return "Continuation quality is healthy. Pullbacks can be considered if acceptance holds."
    if participation == "EMPTY_BREAKOUT_RISK":
        return "Move lacks participation quality. Treat breakouts as suspect until confirmed."
    return "Auction is balanced. Wait for acceptance/rejection before forcing direction."


def main():
    con = connect()
    ensure_schema(con)
    bars = load_bars(con)
    if len(bars) < 4:
        raise SystemExit("Not enough intraday bars for exhaustion state")

    metrics = [candle_metrics(b) for b in bars]
    recent = metrics[-4:]
    prior = metrics[:-4] or recent

    avg_body_eff = sum(m["body_ratio"] for m in recent) / len(recent)
    avg_upper = sum(m["upper_wick_ratio"] for m in recent) / len(recent)
    avg_lower = sum(m["lower_wick_ratio"] for m in recent) / len(recent)
    recent_range = sum(m["range"] for m in recent) / len(recent)
    prior_range = sum(m["range"] for m in prior) / len(prior)
    momentum_decay = recent_range / prior_range if prior_range else 1.0

    vols = [float(b["volume"] or 0) for b in bars]
    recent_vol = sum(vols[-4:]) / 4.0
    prior_vols = vols[:-4] or vols[-4:]
    prior_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 1.0
    volume_ratio = recent_vol / prior_vol if prior_vol else 1.0

    fail_count = failed_acceptance_count(bars)
    exhaustion_score = min(
        1.0,
        max(0.0,
            (avg_upper * 0.35)
            + ((1.0 - avg_body_eff) * 0.25)
            + (min(fail_count, 3) / 3.0 * 0.25)
            + ((1.0 - min(momentum_decay, 1.0)) * 0.15)
        ),
    )

    exhaustion_state = classify_exhaustion(exhaustion_score, fail_count, avg_upper, momentum_decay)
    participation = classify_participation(exhaustion_state, volume_ratio, avg_body_eff)

    latest_bar = bars[-1]
    asof_ts = str(latest_bar["ts"])
    session_date = asof_ts[:10]
    surface = latest_surface(con)
    prev = prior_state(con, session_date, asof_ts)
    transition_state = classify_transition(exhaustion_state, prev, surface)
    note = make_note(exhaustion_state, transition_state, participation)
    created_at = datetime.now(timezone.utc).isoformat()

    con.execute(
        """
        INSERT INTO exhaustion_transition_state (
          session_date, asof_ts, underlying, exhaustion_score,
          expansion_efficiency, upper_wick_ratio, lower_wick_ratio,
          momentum_decay, failed_acceptance_count, exhaustion_state,
          transition_state, participation_quality, note, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_date, asof_ts, underlying) DO UPDATE SET
          exhaustion_score=excluded.exhaustion_score,
          expansion_efficiency=excluded.expansion_efficiency,
          upper_wick_ratio=excluded.upper_wick_ratio,
          lower_wick_ratio=excluded.lower_wick_ratio,
          momentum_decay=excluded.momentum_decay,
          failed_acceptance_count=excluded.failed_acceptance_count,
          exhaustion_state=excluded.exhaustion_state,
          transition_state=excluded.transition_state,
          participation_quality=excluded.participation_quality,
          note=excluded.note,
          created_at=excluded.created_at
        """,
        (
            session_date,
            asof_ts,
            SYMBOL,
            exhaustion_score,
            avg_body_eff,
            avg_upper,
            avg_lower,
            momentum_decay,
            fail_count,
            exhaustion_state,
            transition_state,
            participation,
            note,
            created_at,
        ),
    )
    con.commit()
    con.close()

    payload = {
        "session_date": session_date,
        "asof_ts": asof_ts,
        "underlying": SYMBOL,
        "exhaustion_score": exhaustion_score,
        "expansion_efficiency": avg_body_eff,
        "upper_wick_ratio": avg_upper,
        "lower_wick_ratio": avg_lower,
        "momentum_decay": momentum_decay,
        "failed_acceptance_count": fail_count,
        "exhaustion_state": exhaustion_state,
        "transition_state": transition_state,
        "participation_quality": participation,
        "note": note,
        "created_at": created_at,
    }

    (OUTDIR / "exhaustion_transition_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTDIR / "exhaustion_transition_state.txt").write_text(
        "\n".join([
            "EXHAUSTION / TRANSITION STATE",
            "=============================",
            f"asof: {asof_ts}",
            f"exhaustion_state: {exhaustion_state}",
            f"transition_state: {transition_state}",
            f"participation_quality: {participation}",
            f"exhaustion_score: {exhaustion_score:.2f}",
            f"expansion_efficiency: {avg_body_eff:.2f}",
            f"upper_wick_ratio: {avg_upper:.2f}",
            f"momentum_decay: {momentum_decay:.2f}",
            f"failed_acceptance_count: {fail_count}",
            f"note: {note}",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
