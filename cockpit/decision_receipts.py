"""Decision receipts and permission-score trend helpers for Live Read."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from execution_hierarchy import ADVISORY_SURFACE_PART_NAMES, part_label
from gate_workflows import gate_metadata, primary_context_setup, primary_trade_setup
from setup_event_lifecycle import (
    build_setup_event_lifecycle,
    primary_actionable_setup_event,
    primary_setup_event,
    setup_dict_from_event,
)


def _feature_label(name: str) -> str:
    return part_label(name)


def _rank_features(permission: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scored = []
    ignored = set(ADVISORY_SURFACE_PART_NAMES)
    for name, item in (permission.get("scores") or {}).items():
        if name in ignored:
            continue
        scored.append(
            {
                "name": name,
                "label": _feature_label(name),
                "score": int(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            }
        )
    ranked = sorted(scored, key=lambda row: row["score"], reverse=True)
    return {"best": ranked[:3], "worst": list(reversed(ranked[-3:]))}


def _feature_scores(permission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ignored = set(ADVISORY_SURFACE_PART_NAMES)
    return {
        name: {
            "label": _feature_label(name),
            "score": int(item.get("score", 0)),
            "reason": str(item.get("reason", "")),
            "phase": item.get("phase"),
            "phase_reason": item.get("phase_reason"),
        }
        for name, item in (permission.get("scores") or {}).items()
        if name not in ignored
    }


def build_decision_receipt(
    signal_ts: str,
    symbol: str,
    spot: float | None,
    permission: dict[str, Any],
    target_plan: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
    previous_receipt: dict[str, Any] | None = None,
    session_date: str | None = None,
    session_date_source: str = "signal_ts",
) -> dict[str, Any]:
    setup = primary_trade_setup(setups)
    context_setup = primary_context_setup(setups)
    ranked = _rank_features(permission)
    setup_events, setup_event_transitions = build_setup_event_lifecycle(
        signal_ts, setups, previous_receipt
    )
    promoted_setup_event = primary_actionable_setup_event(setup_events)
    if promoted_setup_event and not gate_metadata(setup).get("actionable"):
        setup = setup_dict_from_event(promoted_setup_event)
    return {
        "schema": "sharpedge.decision_receipt.v1",
        "ts": signal_ts,
        "symbol": symbol,
        "spot": spot,
        "session_date": session_date or signal_ts[:10],
        "session_date_source": session_date_source,
        "wall_clock_date": signal_ts[:10],
        "permission": permission.get("trade_permission_score"),
        "execution_permission": permission.get(
            "execution_permission_score", permission.get("trade_permission_score")
        ),
        "gate": permission.get("trade_gate"),
        "bias": permission.get("bias"),
        "setup_conviction": permission.get("setup_conviction") or {},
        "setup": setup.get("tag"),
        "setup_bias": setup.get("bias"),
        "entry_gate": gate_metadata(setup),
        "context_gate": gate_metadata(context_setup),
        "setup_events": setup_events,
        "setup_event_transitions": setup_event_transitions,
        "primary_setup_event": primary_setup_event(setup_events, setup.get("tag")),
        "strategic_target": {
            "label": target_plan.get("label"),
            "price": target_plan.get("price"),
            "status": target_plan.get("status"),
            "distance": target_plan.get("distance"),
        },
        "reachable_today": target_plan.get("reachable_today") or {},
        "likely_travel": target_plan.get("likely_travel", ""),
        "top_trade": [item["label"] for item in ranked["best"]],
        "top_wait": [item["label"] for item in ranked["worst"]],
        "feature_scores": _feature_scores(permission),
        "outcome": None,
    }


def load_recent_receipts(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def append_decision_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt) + "\n")


def _setup_transition_label(transition: dict[str, Any]) -> str:
    event_type = str(transition.get("event_type") or "setup")
    status = str(transition.get("status") or "").upper()
    level_name = transition.get("level_name")
    suffix = f" @ {level_name}" if level_name else ""
    return f"{event_type} {status}{suffix}"


def build_permission_score_trend(
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
    max_points: int = 6,
) -> dict[str, Any]:
    history = [*prior_receipts[-(max_points - 1) :], current_receipt]
    points = [
        {
            "ts": row.get("ts"),
            "time": str(row.get("ts", ""))[11:16],
            "score": row.get("permission"),
            "event_markers": [
                _setup_transition_label(item)
                for item in (row.get("setup_event_transitions") or [])
                if isinstance(item, dict)
            ],
        }
        for row in history
    ]
    previous = prior_receipts[-1] if prior_receipts else None
    direction = "new"
    score_delta = None
    feature_deltas: list[dict[str, Any]] = []
    if previous:
        current_score = current_receipt.get("permission")
        previous_score = previous.get("permission")
        if isinstance(current_score, (int, float)) and isinstance(
            previous_score, (int, float)
        ):
            score_delta = int(current_score - previous_score)
            direction = (
                "strengthening"
                if score_delta > 0
                else "weakening"
                if score_delta < 0
                else "flat"
            )
        prev_features = previous.get("feature_scores") or {}
        cur_features = current_receipt.get("feature_scores") or {}
        deltas = []
        for name, item in cur_features.items():
            cur_score = item.get("score")
            prev_score = (prev_features.get(name) or {}).get("score", 0)
            if not isinstance(cur_score, (int, float)) or not isinstance(
                prev_score, (int, float)
            ):
                continue
            delta = int(cur_score - prev_score)
            if delta:
                deltas.append(
                    {
                        "feature": item.get("label", _feature_label(name)),
                        "delta": delta,
                    }
                )
        feature_deltas = sorted(
            deltas, key=lambda row: abs(row["delta"]), reverse=True
        )[:4]
    setup_transitions = [
        {
            **item,
            "label": _setup_transition_label(item),
        }
        for item in (current_receipt.get("setup_event_transitions") or [])
        if isinstance(item, dict)
    ]
    return {
        "schema": "sharpedge.permission_score_trend.v1",
        "points": points,
        "current": current_receipt.get("permission"),
        "previous": previous.get("permission") if previous else None,
        "delta": score_delta,
        "direction": direction,
        "largest_changes_since_last_update": feature_deltas,
        "setup_transitions_since_last_update": setup_transitions,
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_truth_rows(path: Path, symbol: str = "SPY") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            session_date = str(row.get("date", ""))
            if not session_date:
                continue
            rows[session_date] = {
                "high": _as_float(row.get("high")),
                "low": _as_float(row.get("low")),
                "close": _as_float(row.get("close")),
                "open": _as_float(row.get("open")),
            }
    return rows


def load_trade_outcomes(
    db_path: Path, symbol: str = "SPY"
) -> dict[str, dict[str, Any]]:
    if not db_path.exists():
        return {}
    outcomes: dict[str, dict[str, Any]] = {}
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "trades" in tables:
            rows = conn.execute(
                "SELECT trade_date, symbol, pnl_r, pnl FROM trades WHERE symbol = ?",
                (symbol,),
            ).fetchall()
            for session_date, _symbol, pnl_r, pnl in rows:
                bucket = outcomes.setdefault(
                    str(session_date),
                    {
                        "trade_taken": False,
                        "trade_count": 0,
                        "pnl_r": 0.0,
                        "realized_pnl": 0.0,
                    },
                )
                bucket["trade_taken"] = True
                bucket["trade_count"] += 1
                bucket["pnl_r"] += pnl_r or 0.0
                bucket["realized_pnl"] += pnl or 0.0
        if "trade_execution_log" in tables:
            rows = conn.execute(
                "SELECT session_date, underlying, realized_pnl, realized_return_pct FROM trade_execution_log WHERE underlying = ?",
                (symbol,),
            ).fetchall()
            for session_date, _underlying, realized_pnl, realized_return_pct in rows:
                bucket = outcomes.setdefault(
                    str(session_date),
                    {
                        "trade_taken": False,
                        "trade_count": 0,
                        "pnl_r": None,
                        "realized_pnl": 0.0,
                        "realized_return_pct": 0.0,
                    },
                )
                bucket["trade_taken"] = True
                bucket["trade_count"] += 1
                bucket["realized_pnl"] += realized_pnl or 0.0
                bucket["realized_return_pct"] += realized_return_pct or 0.0
    finally:
        conn.close()
    return outcomes


def enrich_receipt_outcome(
    receipt: dict[str, Any],
    truth_row: dict[str, Any] | None,
    trade_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trade_row = trade_row or {}
    spot = _as_float(receipt.get("spot"))
    strategic = receipt.get("strategic_target") or {}
    reachable = receipt.get("reachable_today") or {}
    target_price = _as_float(strategic.get("price"))
    reachable_price = _as_float(reachable.get("price"))
    high = _as_float((truth_row or {}).get("high"))
    low = _as_float((truth_row or {}).get("low"))
    close = _as_float((truth_row or {}).get("close"))
    direction = None
    if spot is not None and target_price is not None:
        direction = (
            "up" if target_price > spot else "down" if target_price < spot else "flat"
        )
    strategic_reached = None
    reachable_reached = None
    max_excursion = None
    if direction == "up" and high is not None and spot is not None:
        strategic_reached = bool(target_price is not None and high >= target_price)
        reachable_reached = bool(
            reachable_price is not None and high >= reachable_price
        )
        max_excursion = round(high - spot, 2)
    elif direction == "down" and low is not None and spot is not None:
        strategic_reached = bool(target_price is not None and low <= target_price)
        reachable_reached = bool(reachable_price is not None and low <= reachable_price)
        max_excursion = round(spot - low, 2)
    return {
        "target_reached": strategic_reached,
        "reachable_today_reached": reachable_reached,
        "max_excursion": max_excursion,
        "close": close,
        "trade_taken": bool(trade_row.get("trade_taken", False)),
        "trade_count": int(trade_row.get("trade_count", 0) or 0),
        "pnl_r": trade_row.get("pnl_r"),
        "realized_pnl": trade_row.get("realized_pnl"),
        "realized_return_pct": trade_row.get("realized_return_pct"),
        "session_high": high,
        "session_low": low,
    }


def update_receipt_outcomes(
    receipt_path: Path,
    truth_path: Path,
    db_path: Path | None = None,
    signal_path: Path | None = None,
    symbol: str = "SPY",
) -> dict[str, Any]:
    receipts = load_recent_receipts(receipt_path, limit=10_000)
    truth_rows = load_truth_rows(truth_path, symbol=symbol)
    trade_rows = load_trade_outcomes(db_path, symbol=symbol) if db_path else {}
    updated = []
    updated_count = 0
    for receipt in receipts:
        session_date = str(
            receipt.get("session_date") or str(receipt.get("ts", ""))[:10]
        )
        outcome = enrich_receipt_outcome(
            receipt,
            truth_rows.get(session_date),
            trade_rows.get(session_date),
        )
        merged = dict(receipt)
        if merged.get("outcome") != outcome:
            updated_count += 1
        merged["outcome"] = outcome
        updated.append(merged)
    receipt_path.write_text(
        "".join(json.dumps(row) + "\n" for row in updated), encoding="utf-8"
    )
    signal_updated = False
    if signal_path and signal_path.exists() and updated:
        signal = json.loads(signal_path.read_text(encoding="utf-8"))
        latest = updated[-1]
        signal_receipt = signal.get("decision_receipt") or {}
        if signal_receipt.get("ts") == latest.get("ts"):
            signal_receipt["outcome"] = latest.get("outcome")
            signal["decision_receipt"] = signal_receipt
            signal_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
            signal_updated = True
    return {
        "updated_count": updated_count,
        "receipt_count": len(updated),
        "signal_updated": signal_updated,
    }


__all__ = [
    "append_decision_receipt",
    "build_decision_receipt",
    "build_permission_score_trend",
    "enrich_receipt_outcome",
    "load_recent_receipts",
    "load_trade_outcomes",
    "load_truth_rows",
    "update_receipt_outcomes",
]
