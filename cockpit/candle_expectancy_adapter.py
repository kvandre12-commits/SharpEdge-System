"""Live adapter for candle-conditioned expectancy research rows.

This module attaches historical candle expectancy context to Candle Coach without
turning it into execution authority.  It is deliberately fail-soft: no table, no
match, or thin/research-only rows remain education-only evidence.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

CANDLE_EXPECTANCY_ADAPTER_SCHEMA = "sharpedge.candle_expectancy_adapter.v1"
DEFAULT_DB_PATH = Path(os.getenv("SPY_DB_PATH", "data/spy_truth.db"))
DEFAULT_TABLE = os.getenv("CANDLE_CONFIDENCE_TABLE", "candle_confidence_matrix")

DIMENSIONS = (
    "event_name",
    "event_direction",
    "nearest_reference_name",
    "nearest_reference_relation",
    "reference_distance_bucket",
    "acceptance_state",
    "volume_confirmation",
    "vol_state",
    "macro_state",
    "dp_state",
    "regime_label",
    "open_regime_label",
    "time_bucket",
)

TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tier_1_full",
        (
            "event_name",
            "event_direction",
            "nearest_reference_name",
            "nearest_reference_relation",
            "reference_distance_bucket",
            "acceptance_state",
            "volume_confirmation",
            "vol_state",
            "macro_state",
            "dp_state",
            "regime_label",
            "open_regime_label",
            "time_bucket",
        ),
    ),
    (
        "tier_2_execution",
        (
            "event_name",
            "event_direction",
            "nearest_reference_name",
            "reference_distance_bucket",
            "acceptance_state",
            "volume_confirmation",
            "time_bucket",
        ),
    ),
    (
        "tier_3_core",
        ("event_name", "event_direction", "acceptance_state", "volume_confirmation"),
    ),
    ("tier_4_event_only", ("event_name", "event_direction")),
)

PATTERN_EVENT_MAP = {
    "Bullish engulfing": ("bullish_engulfing", "CALLS"),
    "Bearish engulfing": ("bearish_engulfing", "PUTS"),
    "Inside bar": ("inside_bar", "NEUTRAL"),
    "Outside bar": ("outside_bar", "NEUTRAL"),
    "Dragonfly doji": ("dragonfly_doji", "CALLS"),
    "Gravestone doji": ("gravestone_doji", "PUTS"),
    "Hammer / demand tail": ("demand_tail", "CALLS"),
    "Inverted hammer": ("supply_tail", "PUTS"),
    "Shooting star / supply tail": ("supply_tail", "PUTS"),
    "Doji": ("doji", "NEUTRAL"),
    "Spinning top": ("spinning_top", "NEUTRAL"),
    "Bullish belt hold": ("bullish_conviction", "CALLS"),
    "Bearish belt hold": ("bearish_conviction", "PUTS"),
    "Bullish marubozu / conviction candle": ("bullish_conviction", "CALLS"),
    "Bearish marubozu / conviction candle": ("bearish_conviction", "PUTS"),
    "Strong bullish candle": ("bullish_conviction", "CALLS"),
    "Strong bearish candle": ("bearish_conviction", "PUTS"),
}

OUTPUT_COLUMNS = (
    "match_tier",
    *DIMENSIONS,
    "n",
    "target_before_stop_rate",
    "stop_before_target_rate",
    "same_bar_rate",
    "no_resolution_rate",
    "up_target_first_rate",
    "down_target_first_rate",
    "avg_realized_R",
    "avg_favorable_excursion_pct",
    "avg_adverse_excursion_pct",
    "sample_quality",
    "sample_bucket",
    "confidence_score",
    "confidence_label",
    "positive_edge",
    "deployment_tier",
    "deployment_ready",
    "confidence_notes",
    "confidence_ts",
    "confidence_version",
)

_SAFE_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _safe_identifier(value: str, label: str) -> str:
    if not value or any(ch not in _SAFE_IDENT for ch in value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _pattern_name(pattern: dict[str, Any] | None) -> str:
    return str((pattern or {}).get("name") or "")


def _event_from_patterns(patterns: dict[str, Any]) -> dict[str, Any]:
    # Match the historical event classifier: two-candle events take precedence
    # over single-candle anatomy when both are available.
    for key in ("latest_pair", "latest_single"):
        name = _pattern_name(patterns.get(key))
        if name in PATTERN_EVENT_MAP:
            event_name, direction = PATTERN_EVENT_MAP[name]
            return {
                "event_name": event_name,
                "event_direction": direction,
                "source_pattern": key,
                "source_pattern_name": name,
            }
    return {
        "event_name": None,
        "event_direction": None,
        "source_pattern": None,
        "source_pattern_name": None,
    }


def _state_packet(context: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = context.get(name)
        if isinstance(value, dict):
            return value
    permission = context.get("permission") or context.get("trade_permission") or {}
    if isinstance(permission, dict):
        for name in names:
            value = permission.get(name)
            if isinstance(value, dict):
                return value
    return {}


def _reference_name(raw: Any) -> str:
    name = str(raw or "UNKNOWN")
    if name == "ORH":
        return "ORH_30m"
    if name == "ORL":
        return "ORL_30m"
    if name in {"PDH", "PDL", "PDC", "ORH_30m", "ORL_30m"}:
        return name
    return "UNKNOWN"


def _reference_distance_bucket(distance_pct: Any) -> str:
    try:
        pct = abs(float(distance_pct))
    except (TypeError, ValueError):
        return "unknown_distance"
    # Live location_state distance_pct is percentage points. Historical matrix
    # buckets are equivalent to <=0.1% at, <=0.3% near.
    if pct <= 0.10:
        return "at_reference"
    if pct <= 0.30:
        return "near_reference"
    return "away_from_reference"


def _acceptance_state(raw: Any) -> str:
    value = str(raw or "unknown")
    if value == "accepted_above_level":
        return "accepted_above"
    if value == "accepted_below_level":
        return "accepted_below"
    if value in {
        "accepted_above",
        "accepted_below",
        "testing_reference",
        "insufficient",
    }:
        return value
    if value in {"near_reference", "observed", "required", "unknown"}:
        return "testing_reference"
    return value


def _volume_confirmation(raw: Any) -> str:
    value = str(raw or "unknown").lower()
    if value in {"confirmed", "participating", "mixed", "thin", "unknown"}:
        return value
    if value in {"weak", "missing", "not_evaluable"}:
        return "unknown"
    return value


def _vol_state(raw: Any) -> str:
    value = str(raw or "UNKNOWN").lower()
    if value in {"low", "high"}:
        return value
    if value in {"mid", "normal", "squeeze", "narrow_channel"}:
        return "mid"
    return "UNKNOWN"


def _time_bucket(patterns: dict[str, Any]) -> str:
    latest = patterns.get("latest_single") or {}
    minute = None
    candles = latest.get("candles") if isinstance(latest, dict) else None
    if candles and isinstance(candles[0], dict):
        minute = candles[0].get("minute")
    try:
        value = int(minute)
    except (TypeError, ValueError):
        return "unknown_time"
    if value < 60:
        return "opening_60m"
    if value >= 330:
        return "power_hour"
    return "midday"


def build_live_candle_context(
    patterns: dict[str, Any], sharpedge_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    context = sharpedge_context if isinstance(sharpedge_context, dict) else {}
    event = _event_from_patterns(patterns)
    location = _state_packet(context, "execution_location_state", "location_state")
    acceptance = _state_packet(
        context, "execution_acceptance_state", "acceptance_state"
    )
    volume = _state_packet(context, "execution_volume_state", "volume_state")
    nearest = location.get("nearest_reference") or {}
    volatility = (
        context.get("volatility_structure")
        if isinstance(context.get("volatility_structure"), dict)
        else {}
    )
    return {
        **event,
        "nearest_reference_name": _reference_name(nearest.get("reference_name")),
        "nearest_reference_relation": str(nearest.get("relation") or "UNKNOWN"),
        "reference_distance_bucket": _reference_distance_bucket(
            nearest.get("distance_pct")
        ),
        "acceptance_state": _acceptance_state(acceptance.get("state")),
        "volume_confirmation": _volume_confirmation(
            volume.get("confirmation") or volume.get("state")
        ),
        "vol_state": _vol_state(volatility.get("volatility_state")),
        "macro_state": "UNKNOWN",
        "dp_state": "UNKNOWN",
        "regime_label": "UNKNOWN",
        "open_regime_label": "UNKNOWN",
        "time_bucket": _time_bucket(patterns),
    }


def _connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _query_tier(
    con: sqlite3.Connection,
    table: str,
    live: dict[str, Any],
    match_tier: str,
    tier_dims: tuple[str, ...],
) -> dict[str, Any] | None:
    values = {
        dim: live.get(dim, "ANY") if dim in tier_dims else "ANY" for dim in DIMENSIONS
    }
    where = ["match_tier = ?", *[f"{dim} = ?" for dim in DIMENSIONS]]
    params = [match_tier, *[values[dim] for dim in DIMENSIONS]]
    sql = f"""
        SELECT {", ".join(OUTPUT_COLUMNS)}
        FROM {table}
        WHERE {" AND ".join(where)}
        ORDER BY deployment_ready DESC, confidence_score DESC, n DESC
        LIMIT 1
    """
    row = con.execute(sql, params).fetchone()
    return dict(row) if row else None


def lookup_candle_expectancy(
    patterns: dict[str, Any],
    sharpedge_context: dict[str, Any] | None = None,
    *,
    db_path: Path | str | None = None,
    table: str = DEFAULT_TABLE,
) -> dict[str, Any]:
    """Attach historical candle-confidence row to the live candle context."""
    context = sharpedge_context if isinstance(sharpedge_context, dict) else {}
    live = build_live_candle_context(patterns, context)
    if not live.get("event_name") or not live.get("event_direction"):
        return {
            "schema": CANDLE_EXPECTANCY_ADAPTER_SCHEMA,
            "available": False,
            "status": "unmapped_live_pattern",
            "live_context": live,
            "authority": "education_only_not_trade_permission",
        }
    db = db_path or context.get("candle_expectancy_db_path") or DEFAULT_DB_PATH
    try:
        safe_table = _safe_identifier(str(table), "table")
        with _connect(db) as con:
            if not _table_exists(con, safe_table):
                return {
                    "schema": CANDLE_EXPECTANCY_ADAPTER_SCHEMA,
                    "available": False,
                    "status": "missing_table",
                    "live_context": live,
                    "table": safe_table,
                    "authority": "education_only_not_trade_permission",
                }
            for match_tier, dims in TIERS:
                row = _query_tier(con, safe_table, live, match_tier, dims)
                if row:
                    return {
                        "schema": CANDLE_EXPECTANCY_ADAPTER_SCHEMA,
                        "available": True,
                        "status": "matched",
                        "authority": "education_only_not_trade_permission",
                        "live_context": live,
                        "match": row,
                        "match_tier": match_tier,
                        "interpretation": _interpret_match(row),
                        "lookup_order": [tier for tier, _ in TIERS],
                    }
    except (OSError, sqlite3.Error, ValueError) as exc:
        return {
            "schema": CANDLE_EXPECTANCY_ADAPTER_SCHEMA,
            "available": False,
            "status": "lookup_error",
            "error": str(exc),
            "live_context": live,
            "authority": "education_only_not_trade_permission",
        }
    return {
        "schema": CANDLE_EXPECTANCY_ADAPTER_SCHEMA,
        "available": False,
        "status": "no_matching_row",
        "authority": "education_only_not_trade_permission",
        "live_context": live,
        "lookup_order": [tier for tier, _ in TIERS],
    }


def _interpret_match(row: dict[str, Any]) -> str:
    label = str(row.get("confidence_label") or "NO_CONFIDENCE")
    tier = str(row.get("deployment_tier") or "RESEARCH_ONLY")
    ready = bool(row.get("deployment_ready"))
    n = int(row.get("n") or 0)
    avg_r = row.get("avg_realized_R")
    avg_r_text = f"{float(avg_r):+.2f}R" if isinstance(avg_r, (int, float)) else "n/a"
    if ready:
        posture = "deployment-ready research row"
    elif label in {"MEDIUM", "HIGH"}:
        posture = "supported research context, not permission"
    else:
        posture = "research-only / low-confidence context"
    return f"{posture}: {label}, {tier}, n={n}, avg R {avg_r_text}."


def expectancy_gate_from_lookup(
    permission: dict[str, Any] | None, expectancy: dict[str, Any]
) -> dict[str, str]:
    permission = permission if isinstance(permission, dict) else {}
    score = permission.get("execution_permission_score") or permission.get(
        "trade_permission_score"
    )
    gate = permission.get("trade_gate") or "UNKNOWN"
    if not expectancy.get("available"):
        status = str(expectancy.get("status") or "missing_empirical_ev")
        return {
            "label": "Net expectancy",
            "status": status if status != "matched" else "missing_empirical_ev",
            "message": (
                f"SharpEdge permission spine says {gate} {score}/100, but no live candle-confidence matrix row matched yet. "
                f"Adapter status={status}."
            ),
        }
    row = expectancy.get("match") or {}
    ready = bool(row.get("deployment_ready"))
    state = (
        "historical_context_attached" if not ready else "deployment_ready_research_row"
    )
    return {
        "label": "Net expectancy",
        "status": state,
        "message": (
            f"Historical candle matrix attached: {expectancy.get('interpretation')} "
            f"Permission spine remains {gate} {score}/100; this row is education/context, not execution authority."
        ),
    }


__all__ = [
    "CANDLE_EXPECTANCY_ADAPTER_SCHEMA",
    "build_live_candle_context",
    "expectancy_gate_from_lookup",
    "lookup_candle_expectancy",
]
