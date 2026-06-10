#!/usr/bin/env python3
"""Build a Robinhood MCP handoff for fair-value-gap fill monitoring.

This script is intentionally conservative. It does not place trades and it does
not require Robinhood credentials. It reads the repository's existing SharpEdge
artifacts and emits a small JSON/TXT payload that an MCP-connected agent can use
for watchlist/quote monitoring.
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

GAP_EVENTS_CSV = OUTDIR / "gap_excursion_metrics.csv"
TOP_EDGES_CSV = OUTDIR / "top_gap_fill_edges.csv"
OUT_JSON = OUTDIR / "robinhood_fvg_monitor.json"
OUT_TXT = OUTDIR / "robinhood_fvg_monitor.txt"

MIN_SAMPLE_N = int(os.getenv("FVG_MONITOR_MIN_SAMPLE_N", "5"))
MIN_FILL_RATE = float(os.getenv("FVG_MONITOR_MIN_FILL_RATE", "0.55"))
MIN_TRADABILITY = float(os.getenv("FVG_MONITOR_MIN_TRADABILITY", "0.35"))
ROBINHOOD_MCP_SERVER = os.getenv("ROBINHOOD_MCP_SERVER", "robinhood-trading")
ROBINHOOD_MCP_AGENT = os.getenv("ROBINHOOD_MCP_AGENT", "code-puppy")
CODE_PUPPY_CONFIG_DIR = Path(
    os.getenv("CODE_PUPPY_CONFIG_DIR", "~/.code_puppy")
).expanduser()
MCP_REGISTRY_JSON = CODE_PUPPY_CONFIG_DIR / "mcp_registry.json"
MCP_BINDINGS_JSON = CODE_PUPPY_CONFIG_DIR / "mcp_agent_bindings.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_file(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def lookup_registered_server(registry: Any, server_name: str) -> dict[str, Any]:
    if not isinstance(registry, dict):
        return {}
    direct = registry.get(server_name)
    if isinstance(direct, dict) and str(direct.get("name", server_name)) == server_name:
        return direct
    for key, value in registry.items():
        if isinstance(value, dict) and (key == server_name or value.get("name") == server_name):
            return value
    return {}


def robinhood_bridge_status() -> dict[str, Any]:
    registry = read_json_file(MCP_REGISTRY_JSON)
    bindings = read_json_file(MCP_BINDINGS_JSON)
    server = lookup_registered_server(registry, ROBINHOOD_MCP_SERVER)
    bindings_map = bindings.get("bindings", {}) if isinstance(bindings, dict) else {}
    agent_bindings = (
        bindings_map.get(ROBINHOOD_MCP_AGENT, {}) if isinstance(bindings_map, dict) else {}
    )
    bound_config = (
        agent_bindings.get(ROBINHOOD_MCP_SERVER)
        if isinstance(agent_bindings, dict)
        else None
    )

    registered = bool(server)
    enabled = bool(server.get("enabled")) if registered else False
    agent_bound = isinstance(bound_config, dict)
    auto_start = bool(bound_config.get("auto_start")) if agent_bound else False

    status = "ready"
    reason = "bridge_ready"
    if not registered:
        status = "unconfigured"
        reason = "server_not_registered"
    elif not enabled:
        status = "disabled"
        reason = "server_disabled"
    elif not agent_bound:
        status = "unbound"
        reason = "server_not_bound_to_agent"

    available = status == "ready"
    return {
        "server": ROBINHOOD_MCP_SERVER,
        "agent": ROBINHOOD_MCP_AGENT,
        "registered": registered,
        "enabled": enabled,
        "agent_bound": agent_bound,
        "auto_start": auto_start,
        "available": available,
        "status": status,
        "reason": reason,
        "fallback_mode": "mcp_quote_monitoring" if available else "artifact_only_manual_review",
        "url": server.get("config", {}).get("url") if registered else None,
    }


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def fetch_one(con: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> dict[str, Any]:
    con.row_factory = sqlite3.Row
    row = con.execute(query, params).fetchone()
    return dict(row) if row else {}


def latest_gap_event(rows: list[dict[str, str]]) -> dict[str, str]:
    symbol_rows = [r for r in rows if r.get("symbol", "").upper() == SYMBOL]
    usable = [r for r in symbol_rows if r.get("gap_direction") not in {"", "FLAT"}]
    candidates = usable or symbol_rows
    if not candidates:
        return {}
    return sorted(candidates, key=lambda r: r.get("session_date", ""))[-1]


def edge_score(row: dict[str, str]) -> tuple[float, float, int]:
    return (
        as_float(row.get("tradability_score")),
        as_float(row.get("fill_rate")),
        as_int(row.get("n")),
    )


def matching_edges(
    latest_event: dict[str, str],
    edges: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not latest_event:
        return sorted(edges, key=edge_score, reverse=True)

    keys = [
        "event_type",
        "gap_direction",
        "vol_state",
        "macro_state",
        "dp_state",
        "open_regime_label",
        "setup_dir",
        "key_source",
    ]

    def match_count(edge: dict[str, str]) -> int:
        score = 0
        for key in keys:
            event_value = str(latest_event.get(key, "")).upper()
            edge_value = str(edge.get(key, "")).upper()
            if event_value and edge_value and event_value == edge_value:
                score += 1
        return score

    ranked = sorted(edges, key=lambda e: (match_count(e), *edge_score(e)), reverse=True)
    return ranked


def latest_options_context() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "options_positioning_metrics"):
            return {}
        return fetch_one(
            con,
            """
            SELECT *
            FROM options_positioning_metrics
            WHERE underlying = ?
            ORDER BY session_date DESC, snapshot_ts DESC
            LIMIT 1
            """,
            (SYMBOL,),
        )
    finally:
        con.close()


def latest_risk_context() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        if not table_exists(con, "risk_decision_layer"):
            return {}
        return fetch_one(
            con,
            """
            SELECT *
            FROM risk_decision_layer
            WHERE symbol = ?
            ORDER BY date DESC, decision_ts DESC
            LIMIT 1
            """,
            (SYMBOL,),
        )
    finally:
        con.close()


def directional_hypothesis(gap_direction: str) -> dict[str, str]:
    direction = gap_direction.upper()
    if direction == "UP":
        return {
            "fill_bias": "bearish_gap_fill_watch",
            "option_side_watch": "puts_or_put_spreads",
            "reason": "Up gaps fill downward toward prior close / gap fill level.",
        }
    if direction == "DOWN":
        return {
            "fill_bias": "bullish_gap_fill_watch",
            "option_side_watch": "calls_or_call_spreads",
            "reason": "Down gaps fill upward toward prior close / gap fill level.",
        }
    return {
        "fill_bias": "neutral_no_gap_watch",
        "option_side_watch": "none",
        "reason": "No directional gap detected.",
    }


def build_decision(
    latest_event: dict[str, Any],
    edge: dict[str, Any],
    risk: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if not latest_event:
        return "no_trade", ["missing_gap_event"]

    if str(latest_event.get("fill_completed", "0")) == "1":
        reasons.append("gap_already_filled")

    if latest_event.get("gap_direction") in {"", "FLAT", None}:
        reasons.append("no_directional_gap")

    sample_n = as_int(edge.get("n"))
    fill_rate = as_float(edge.get("fill_rate"))
    tradability = as_float(edge.get("tradability_score"))

    if sample_n < MIN_SAMPLE_N:
        reasons.append(f"low_sample_n<{MIN_SAMPLE_N}")
    if fill_rate < MIN_FILL_RATE:
        reasons.append(f"fill_rate<{MIN_FILL_RATE:.2f}")
    if tradability < MIN_TRADABILITY:
        reasons.append(f"tradability<{MIN_TRADABILITY:.2f}")

    risk_state = str(risk.get("deployment_state", "")).upper()
    if not risk:
        reasons.append("risk_context_missing")
    elif risk_state == "NO_TRADE":
        reasons.append("risk_layer_no_trade")

    if reasons:
        return "no_trade", reasons
    return "watch", ["eligible_for_robinhood_quote_monitoring_only"]


def compact_dict(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row}


def build_payload() -> dict[str, Any]:
    gap_rows = read_csv_rows(GAP_EVENTS_CSV)
    edge_rows = read_csv_rows(TOP_EDGES_CSV)
    event = latest_gap_event(gap_rows)
    ranked_edges = matching_edges(event, edge_rows)
    edge = ranked_edges[0] if ranked_edges else {}
    options = latest_options_context()
    risk = latest_risk_context()
    bridge_status = robinhood_bridge_status()
    hypothesis = directional_hypothesis(str(event.get("gap_direction", "")))
    decision, reasons = build_decision(event, edge, risk)

    return {
        "schema_version": "1.0",
        "created_ts": utc_now(),
        "symbol": SYMBOL,
        "decision": decision,
        "reasons": reasons,
        "mode": "dry_run_monitoring_only",
        "latest_gap_event": compact_dict(
            event,
            [
                "session_date",
                "event_type",
                "gap_direction",
                "gap_pct",
                "gap_fill_level",
                "fill_completed",
                "fill_ts",
                "time_to_fill_minutes",
                "session_open",
                "session_high",
                "session_low",
                "session_close",
                "vol_state",
                "macro_state",
                "dp_state",
                "open_regime_label",
                "fill_path_type",
            ],
        ),
        "matched_gap_edge": compact_dict(
            edge,
            [
                "event_type",
                "gap_direction",
                "fill_path_type",
                "n",
                "fill_rate",
                "direct_fill_rate",
                "failed_fill_rate",
                "avg_time_to_fill_minutes",
                "avg_MAE_pct",
                "avg_MFE_pct",
                "tradability_score",
                "sample_quality",
            ],
        ),
        "options_context": compact_dict(
            options,
            [
                "session_date",
                "dte_min",
                "dte_max",
                "spot",
                "atm_strike",
                "max_total_oi_strike",
                "max_call_oi_strike",
                "max_put_oi_strike",
                "total_call_oi",
                "total_put_oi",
                "pcr_oi",
                "gamma_proxy",
                "dealer_state_hint",
            ],
        ),
        "risk_context": compact_dict(
            risk,
            [
                "date",
                "deployment_state",
                "deployment_confidence",
                "capital_risk_pct",
                "tradability_score",
                "sample_n",
                "deployment_denied_reason",
            ],
        ),
        "directional_hypothesis": hypothesis,
        "robinhood_mcp_handoff": {
            "server": ROBINHOOD_MCP_SERVER,
            "intent": "monitor_fair_value_gap_fill_options_context",
            "mode": "dry_run",
            "bridge_status": bridge_status,
            "manual_review_required": not bridge_status["available"],
            "fallback_mode": bridge_status["fallback_mode"],
            "permitted_actions": [
                "read_account_status",
                "read_positions",
                "read_option_chain",
                "monitor_underlying_quote",
                "monitor_option_quotes",
            ],
            "blocked_actions": [
                "place_order",
                "replace_order",
                "cancel_order_without_operator_confirmation",
            ],
            "operator_confirmation_required_for_orders": True,
        },
    }


def render_text(payload: dict[str, Any]) -> str:
    gap = payload["latest_gap_event"]
    edge = payload["matched_gap_edge"]
    hypothesis = payload["directional_hypothesis"]
    options = payload["options_context"]

    handoff = payload["robinhood_mcp_handoff"]
    bridge = handoff["bridge_status"]

    return "\n".join(
        [
            "ROBINHOOD FVG MONITOR",
            f"Created: {payload['created_ts']}",
            f"Symbol: {payload['symbol']}",
            f"Decision: {payload['decision']}",
            f"Reasons: {', '.join(payload['reasons'])}",
            "",
            f"Gap date: {gap.get('session_date', 'NA')}",
            f"Gap direction: {gap.get('gap_direction', 'NA')}",
            f"Gap fill level: {gap.get('gap_fill_level', 'NA')}",
            f"Fill completed: {gap.get('fill_completed', 'NA')}",
            f"Edge fill rate: {edge.get('fill_rate', 'NA')} n={edge.get('n', 'NA')}",
            f"Avg time to fill: {edge.get('avg_time_to_fill_minutes', 'NA')} min",
            "",
            f"Hypothesis: {hypothesis.get('fill_bias')}",
            f"Option side watch: {hypothesis.get('option_side_watch')}",
            f"Spot: {options.get('spot', 'NA')}",
            f"ATM strike: {options.get('atm_strike', 'NA')}",
            f"Dealer state: {options.get('dealer_state_hint', 'NA')}",
            "",
            "Robinhood MCP mode: dry_run_monitoring_only",
            f"Bridge status: {bridge.get('status', 'unknown')} ({bridge.get('reason', 'na')})",
            f"Fallback mode: {handoff.get('fallback_mode', 'artifact_only_manual_review')}",
            "Orders: BLOCKED unless operator confirms manually.",
        ]
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
