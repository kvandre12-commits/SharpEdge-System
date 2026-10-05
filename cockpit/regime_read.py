"""Surface the daily regime (regime_daily) as a cockpit backdrop strip.

`build_regime_spy_daily.py` computes a slow, once-a-day "what regime are we in"
label (vol / vol-trend / dark-pool / macro / compression) plus a day-over-day
transition flag, and writes it to the `regime_daily` table. Historically that
never reached the live cockpit — this module reads the latest computed row and
turns it into a small, DATED structural-backdrop packet the Live Read can show.

Design notes:
  - This is deliberately a READ of the last batch-computed regime, not a live
    recompute. The regime is an end-of-day structural state, so showing the most
    recent row (clearly stamped with its date + staleness) is honest and cheap.
  - If the DB / table / row is missing, we return available=False and the
    cockpit renders nothing (same graceful pattern as the other context reads).
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from typing import Any, Optional


def _default_db_path() -> str:
    env = os.getenv("SPY_DB_PATH")
    if env:
        return env
    return os.path.expanduser("~/SharpEdge-System/data/spy_truth.db")


def _humanize_label(regime_label: str | None) -> str:
    """Turn 'mid_vol|rising_voltrend|low_dp|low_macro|0_comp' into prose."""
    if not regime_label:
        return ""
    parts = regime_label.split("|")
    nice: list[str] = []
    for p in parts:
        p = p.strip()
        if p.endswith("_comp"):
            digit = p[:-5]
            nice.append("compression ON" if digit not in ("0", "") else "no compression")
        else:
            nice.append(p.replace("_", " "))
    return ", ".join(nice)


def build_regime_read_live(
    db_path: str | None = None,
    *,
    symbol: str = "SPY",
    today: dt.date | None = None,
) -> dict[str, Any]:
    """Read the latest `regime_daily` row and package it for the cockpit."""
    db_path = db_path or _default_db_path()
    today = today or dt.date.today()

    if not os.path.exists(db_path):
        return {
            "schema": "sharpedge.regime_read.v1",
            "available": False,
            "reason": f"regime DB not found at {db_path}",
            "source": "db:regime_daily",
        }

    try:
        con = sqlite3.connect(db_path)
        try:
            # Guard: table may not exist yet.
            tbls = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "regime_daily" not in tbls:
                return {
                    "schema": "sharpedge.regime_read.v1",
                    "available": False,
                    "reason": "regime_daily table not built yet "
                    "(run scripts/build_regime_spy_daily.py)",
                    "source": "db:regime_daily",
                }
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT date, symbol, regime_id, regime_label,
                       vol_state, vol_trend_state, dp_state, macro_state,
                       compression_flag, transition_flag, transition_score,
                       transition_label, regime_ts
                FROM regime_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        finally:
            con.close()
    except Exception as exc:  # never let a bad DB break the cockpit
        return {
            "schema": "sharpedge.regime_read.v1",
            "available": False,
            "reason": f"regime read failed: {exc}",
            "source": "db:regime_daily",
        }

    if row is None:
        return {
            "schema": "sharpedge.regime_read.v1",
            "available": False,
            "reason": f"no regime_daily rows for {symbol}",
            "source": "db:regime_daily",
        }

    regime_date = row["date"]
    stale_days: int | None = None
    try:
        stale_days = (today - dt.date.fromisoformat(regime_date)).days
    except Exception:
        stale_days = None

    transition_flag = int(row["transition_flag"] or 0)
    human = _humanize_label(row["regime_label"])

    # Story: structural backdrop + freshness + transition callout.
    freshness = ""
    if stale_days is not None:
        if stale_days <= 0:
            freshness = "today's close"
        elif stale_days == 1:
            freshness = "1 trading day old"
        else:
            freshness = f"{stale_days} days old"

    if transition_flag:
        headline = f"REGIME SHIFT · {row['regime_id']}"
        story = (
            f"Regime just transitioned ({row['transition_label']}). "
            f"Transitions often precede a volatility expansion — respect the change. "
            f"Backdrop: {human}."
        )
    else:
        headline = f"REGIME {row['regime_id']}"
        story = f"Structural backdrop: {human}."
    if freshness:
        story += f" [{freshness}]"

    return {
        "schema": "sharpedge.regime_read.v1",
        "available": True,
        "date": regime_date,
        "symbol": symbol,
        "regime_id": row["regime_id"],
        "regime_label": row["regime_label"],
        "regime_label_human": human,
        "vol_state": row["vol_state"],
        "vol_trend_state": row["vol_trend_state"],
        "dp_state": row["dp_state"],
        "macro_state": row["macro_state"],
        "compression_flag": int(row["compression_flag"] or 0),
        "transition_flag": transition_flag,
        "transition_score": int(row["transition_score"] or 0),
        "transition_label": row["transition_label"],
        "stale_days": stale_days,
        "headline": headline,
        "story": story,
        "regime_ts": row["regime_ts"],
        "source": "db:regime_daily",
    }


__all__ = ["build_regime_read_live"]
