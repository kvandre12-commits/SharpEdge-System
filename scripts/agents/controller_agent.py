#!/usr/bin/env python3
"""Deterministic publication controller for SharpEdge artifacts.

The old controller called Gemini. This version is intentionally local-only:
it reads finished pipeline artifacts, applies conservative rules, and writes the
same output contract without requiring any AI API access.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
SYMBOL = os.getenv("SYMBOL", "SPY").upper()
OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUTDIR / "agent_controller_decision.json"
OUT_TXT = OUTDIR / "agent_controller_prompt.txt"
LATEST_SIGNAL_CSV = OUTDIR / "latest_signal_strength.csv"
ROBINHOOD_MONITOR_JSON = OUTDIR / "robinhood_fvg_monitor.json"
HEALTH_WARNINGS = OUTDIR / "health" / "warnings.log"

MIN_CONTROLLER_CONFIDENCE = float(os.getenv("CONTROLLER_MIN_CONFIDENCE", "0.55"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def fetch_latest(con: sqlite3.Connection, table: str, query: str) -> dict[str, Any]:
    if not table_exists(con, table):
        return {}
    con.row_factory = sqlite3.Row
    row = con.execute(query, (SYMBOL,)).fetchone()
    return dict(row) if row else {}


def latest_db_context() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"db_present": False}

    con = sqlite3.connect(DB_PATH)
    try:
        return {
            "db_present": True,
            "latest_signal": fetch_latest(
                con,
                "signals_daily",
                """
                SELECT * FROM signals_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
                """,
            ),
            "latest_regime": fetch_latest(
                con,
                "regime_daily",
                """
                SELECT * FROM regime_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
                """,
            ),
            "latest_risk": fetch_latest(
                con,
                "risk_decision_layer",
                """
                SELECT * FROM risk_decision_layer
                WHERE symbol = ?
                ORDER BY date DESC, decision_ts DESC
                LIMIT 1
                """,
            ),
            "latest_options_positioning": fetch_latest(
                con,
                "options_positioning_metrics",
                """
                SELECT * FROM options_positioning_metrics
                WHERE underlying = ?
                ORDER BY session_date DESC, snapshot_ts DESC
                LIMIT 1
                """,
            ),
        }
    finally:
        con.close()


def health_warnings() -> list[str]:
    if not HEALTH_WARNINGS.exists():
        return []
    return [line.strip() for line in HEALTH_WARNINGS.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_evidence() -> dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "latest_signal_strength": read_first_csv_row(LATEST_SIGNAL_CSV),
        "robinhood_fvg_monitor": read_json(ROBINHOOD_MONITOR_JSON),
        "db_context": latest_db_context(),
        "health_warnings": health_warnings(),
    }


def decide(evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    risk_flags: list[str] = []
    score = 0.0

    signal = evidence.get("latest_signal_strength", {})
    monitor = evidence.get("robinhood_fvg_monitor", {})
    db_context = evidence.get("db_context", {})
    latest_risk = db_context.get("latest_risk", {}) or {}
    warnings = evidence.get("health_warnings", [])

    if not db_context.get("db_present"):
        reasons.append("SPY database missing")
        risk_flags.append("missing_database")
    else:
        score += 0.20

    if signal:
        score += 0.20
        reasons.append("latest signal artifact present")
    else:
        risk_flags.append("missing_latest_signal")

    if monitor:
        score += 0.20
        reasons.append(f"Robinhood FVG monitor decision={monitor.get('decision', 'unknown')}")
    else:
        risk_flags.append("missing_robinhood_fvg_monitor")

    deployment_state = str(latest_risk.get("deployment_state", "")).upper()
    deployment_confidence = as_float(latest_risk.get("deployment_confidence"))
    if deployment_state and deployment_state != "NO_TRADE":
        score += 0.20
        reasons.append(f"risk layer state={deployment_state}")
    elif deployment_state == "NO_TRADE":
        risk_flags.append("risk_layer_no_trade")

    if deployment_confidence > 0:
        score += min(0.20, deployment_confidence * 0.20)

    if warnings:
        risk_flags.append("pipeline_warnings_present")
        reasons.append(f"pipeline warnings={len(warnings)}")
        score = max(0.0, score - 0.20)

    monitor_decision = str(monitor.get("decision", "")).lower()
    if monitor_decision == "no_trade":
        risk_flags.append("monitor_no_trade")

    confidence = max(0.0, min(1.0, score))
    decision = "post" if confidence >= MIN_CONTROLLER_CONFIDENCE and not risk_flags else "hold"

    if not reasons:
        reasons.append("insufficient evidence")

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "summary": "Deterministic local controller; no external AI API used.",
        "reasons": reasons[:6],
        "risk_flags": risk_flags[:8],
        "symbol": SYMBOL,
        "ts_utc": utc_now(),
    }


def render_evidence(evidence: dict[str, Any]) -> str:
    return (
        "SharpEdge deterministic controller evidence\n"
        "External AI APIs: disabled\n\n"
        f"{json.dumps(evidence, indent=2, default=str, sort_keys=True)}\n"
    )


def main() -> None:
    evidence = build_evidence()
    result = decide(evidence)
    OUT_TXT.write_text(render_evidence(evidence), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"controller_decision={result['decision']}")


if __name__ == "__main__":
    main()
