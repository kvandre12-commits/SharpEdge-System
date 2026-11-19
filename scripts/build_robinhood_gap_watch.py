#!/usr/bin/env python3
"""Build a read-only SPY/NVDA overnight-gap watch artifact.

The builder uses local daily bars only. It never calls Robinhood and never places
orders. A separate refresh step may populate ``bars_daily`` before this runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cockpit.historical_refill_surface import build_historical_refill_context
from scripts.build_robinhood_fvg_monitor import robinhood_bridge_status

DB_PATH = Path(os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
SYMBOLS = tuple(
    symbol.strip().upper()
    for symbol in os.getenv("GAP_WATCH_SYMBOLS", "SPY,NVDA").split(",")
    if symbol.strip()
)
OUT_JSON = Path(os.getenv("GAP_WATCH_OUT_JSON", "outputs/robinhood_gap_watch.json"))
OUT_TXT = Path(os.getenv("GAP_WATCH_OUT_TXT", "outputs/robinhood_gap_watch.txt"))
MIN_GAP_PCT = float(os.getenv("GAP_WATCH_MIN_GAP_PCT", "0.0025"))
MAX_STALE_DAYS = int(os.getenv("GAP_WATCH_MAX_STALE_DAYS", "4"))
REFILL_SURFACE_DIR = Path(os.getenv("REFILL_SURFACE_DIR", "outputs"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def load_daily_bars(
    con: sqlite3.Connection, symbol: str
) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT date, symbol, open, high, low, close, volume, source, ingest_ts
        FROM bars_daily
        WHERE symbol = ?
          AND open IS NOT NULL
          AND high IS NOT NULL
          AND low IS NOT NULL
          AND close IS NOT NULL
        ORDER BY date ASC
        """,
        (symbol,),
    ).fetchall()
    return [dict(row) for row in rows]


def refill_surface_paths(symbol: str) -> tuple[Path, Path]:
    prefix = "" if symbol == "SPY" else f"{symbol.lower()}_"
    return (
        REFILL_SURFACE_DIR / f"{prefix}historical_refill_surface.json",
        REFILL_SURFACE_DIR / f"{prefix}historical_refill_stack_surface.json",
    )


def refill_structure(
    symbol: str, bars: list[dict[str, Any]]
) -> dict[str, Any]:
    if not bars:
        return {"available": False, "reason": "daily_bars_missing"}
    surface_path, stack_path = refill_surface_paths(symbol)
    context = build_historical_refill_context(
        bars,
        spot=float(bars[-1]["close"]),
        surface_path=surface_path,
        stack_surface_path=stack_path,
    )
    if not context.get("available"):
        return {
            "available": False,
            "reason": context.get("reason", "refill_context_unavailable"),
            "surface_path": str(surface_path),
        }
    stack = context.get("active_refill_stack") or {}
    return {
        "available": True,
        "authority": "diagnostic_only",
        "surface_path": str(surface_path),
        "surface_generated_at": context.get("surface_generated_at"),
        "stack_label": stack.get("stack_label"),
        "active_count": stack.get("active_count", 0),
        "oldest_event_date": stack.get("oldest_event_date"),
        "latest_event_date": stack.get("latest_event_date"),
        "highest_target": stack.get("highest_target"),
        "nearest_target": stack.get("nearest_target"),
        "items": stack.get("items", []),
        "historical_stack_surface": stack.get("historical_stack_surface"),
        "caveat": context.get("caveat"),
    }


def apply_refill_structure(
    session_gap: dict[str, Any], structure: dict[str, Any]
) -> dict[str, Any]:
    result = {**session_gap, "session_gap_decision": session_gap["decision"]}
    result["refill_structure"] = structure
    active_count = int(structure.get("active_count") or 0)
    if structure.get("available") and active_count:
        label = str(structure.get("stack_label") or "active_refill_stack")
        result["decision"] = "watch"
        result["reasons"] = [
            f"{label}_active",
            "historical_refill_targets_require_quote_monitoring_only",
        ]
    return result


def classify_gap(
    symbol: str,
    bars: list[dict[str, Any]],
    reference_date: date,
    min_gap_pct: float = MIN_GAP_PCT,
    max_stale_days: int = MAX_STALE_DAYS,
) -> dict[str, Any]:
    if len(bars) < 2:
        return {
            "symbol": symbol,
            "decision": "no_trade",
            "reasons": ["insufficient_daily_bars"],
        }

    previous, current = bars[-2], bars[-1]
    prior_close = float(previous["close"])
    session_open = float(current["open"])
    session_high = float(current["high"])
    session_low = float(current["low"])
    gap_pct = (session_open - prior_close) / prior_close if prior_close else 0.0
    gap_direction = "UP" if gap_pct > 0 else "DOWN" if gap_pct < 0 else "FLAT"
    session_date = date.fromisoformat(str(current["date"]))
    age_days = max((reference_date - session_date).days, 0)

    if gap_direction == "UP":
        fill_completed = session_low <= prior_close
    elif gap_direction == "DOWN":
        fill_completed = session_high >= prior_close
    else:
        fill_completed = True

    reasons: list[str] = []
    if age_days > max_stale_days:
        reasons.append(f"stale_bar_age_days>{max_stale_days}")
    if abs(gap_pct) < min_gap_pct:
        reasons.append(f"gap_below_threshold<{min_gap_pct:.4f}")
    if fill_completed:
        reasons.append("gap_already_filled")

    decision = "no_trade" if reasons else "watch"
    if decision == "watch":
        reasons = ["unfilled_gap_eligible_for_quote_monitoring_only"]

    return {
        "symbol": symbol,
        "decision": decision,
        "reasons": reasons,
        "session_date": str(current["date"]),
        "previous_session_date": str(previous["date"]),
        "age_days": age_days,
        "gap_direction": gap_direction,
        "gap_pct": round(gap_pct, 8),
        "gap_percent": round(gap_pct * 100.0, 4),
        "gap_fill_level": prior_close,
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "session_close": float(current["close"]),
        "fill_completed": fill_completed,
        "source": current.get("source"),
        "ingest_ts": current.get("ingest_ts"),
    }


def latest_requested_date(
    bars_by_symbol: dict[str, list[dict[str, Any]]], fallback: date
) -> date:
    dates = [
        date.fromisoformat(str(bars[-1]["date"]))
        for bars in bars_by_symbol.values()
        if bars
    ]
    return max(dates, default=fallback)


def build_payload(
    db_path: Path = DB_PATH,
    symbols: Iterable[str] = SYMBOLS,
    now: datetime | None = None,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
    current_time = now or datetime.now(timezone.utc)
    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    database_reasons: list[str] = []

    if not db_path.exists():
        database_reasons.append("database_missing")
    else:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            if not table_exists(con, "bars_daily"):
                database_reasons.append("bars_daily_missing")
            else:
                bars_by_symbol = {
                    symbol: load_daily_bars(con, symbol) for symbol in requested
                }
        finally:
            con.close()

    reference_date = latest_requested_date(bars_by_symbol, current_time.date())
    watches = [
        apply_refill_structure(
            classify_gap(symbol, bars_by_symbol.get(symbol, []), reference_date),
            refill_structure(symbol, bars_by_symbol.get(symbol, [])),
        )
        for symbol in requested
    ]
    bridge = robinhood_bridge_status()

    return {
        "schema_version": "sharpedge.gap-watch.v1",
        "created_ts": current_time.isoformat(),
        "mode": "read_only_monitoring",
        "symbols": list(requested),
        "reference_date": reference_date.isoformat(),
        "database_reasons": database_reasons,
        "summary": {
            "watch_count": sum(item["decision"] == "watch" for item in watches),
            "no_trade_count": sum(item["decision"] == "no_trade" for item in watches),
        },
        "watches": watches,
        "robinhood_handoff": {
            "bridge_status": bridge,
            "manual_review_required": True,
            "permitted_actions": [
                "read_quote",
                "read_historicals",
                "read_option_chain_if_supported",
            ],
            "blocked_actions": ["place_order", "replace_order", "cancel_order"],
            "operator_confirmation_required_for_orders": True,
        },
    }


def render_text(payload: dict[str, Any]) -> str:
    lines = [
        "ROBINHOOD GAP WATCH — READ ONLY",
        f"Created: {payload['created_ts']}",
        f"Reference date: {payload['reference_date']}",
        "",
    ]
    for item in payload["watches"]:
        lines.extend(
            [
                f"{item['symbol']}: {item['decision'].upper()}",
                f"  Reasons: {', '.join(item['reasons'])}",
                (
                    f"  Gap: {item.get('gap_direction', 'NA')} "
                    f"{item.get('gap_percent', 'NA')}%"
                ),
                f"  Fill target: {item.get('gap_fill_level', 'NA')}",
                f"  Filled: {item.get('fill_completed', 'NA')}",
            ]
        )
    lines.extend(["", "Orders: BLOCKED. Operator confirmation remains mandatory."])
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
