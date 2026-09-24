"""Read-only market context for Social information-value observations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .eligibility import POLICY_VERSION as ELIGIBILITY_POLICY_VERSION
from .evidence_store import connect

CONTEXT_POLICY_VERSION = 1
BAR_DURATION = timedelta(minutes=15)
MAX_PRIOR_REGIME_AGE = timedelta(days=7)
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ContextField:
    field_name: str
    status: str
    value: Any
    source_database: str
    source_table: str
    source_column: str | None
    source_record_time: str | None
    selection_method: str
    offset_seconds: float | None
    quality_status: str
    reason: str


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_context_snapshots (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            context_policy_version INTEGER NOT NULL,
            research_family TEXT NOT NULL
                CHECK (research_family = 'social_information_value'),
            anchor_at_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            source_database TEXT NOT NULL,
            source_db_sha256 TEXT NOT NULL,
            overall_quality TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            attached_at_utc TEXT NOT NULL,
            PRIMARY KEY (capture_id, post_id, context_policy_version),
            FOREIGN KEY (capture_id, post_id)
                REFERENCES post_observations(capture_id, post_id)
        );

        CREATE TABLE IF NOT EXISTS market_context_fields (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            context_policy_version INTEGER NOT NULL,
            field_name TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('observed', 'missing', 'unavailable', 'stale')),
            value_json TEXT,
            source_database TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_column TEXT,
            source_record_time TEXT,
            selection_method TEXT NOT NULL,
            offset_seconds REAL,
            quality_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (
                capture_id, post_id, context_policy_version, field_name
            ),
            FOREIGN KEY (capture_id, post_id, context_policy_version)
                REFERENCES market_context_snapshots(
                    capture_id, post_id, context_policy_version
                )
        );
        """
    )


def _utc_timestamp() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _database_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_market_truth(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.row_factory = sqlite3.Row
    return connection


def _session_clock_field(anchor: datetime) -> ContextField:
    local = anchor.astimezone(NY)
    local_time = local.time()
    if local.weekday() >= 5:
        state = "calendar_weekend"
    elif time(4) <= local_time < time(9, 30):
        state = "premarket_clock_window"
    elif time(9, 30) <= local_time < time(16):
        state = "regular_clock_window"
    elif time(16) <= local_time < time(20):
        state = "after_hours_clock_window"
    else:
        state = "closed_clock_window"
    return ContextField(
        field_name="session_clock_state",
        status="observed",
        value=state,
        source_database="robinhood_social_research.db",
        source_table="temporal_anchors",
        source_column="resolved_at_utc",
        source_record_time=anchor.isoformat().replace("+00:00", "Z"),
        selection_method="new_york_weekday_clock_v1",
        offset_seconds=0.0,
        quality_status="calendar_clock_only",
        reason="clock_window_is_not_exchange_open_confirmation",
    )


def _intraday_fields(
    market: sqlite3.Connection,
    symbol: str,
    anchor: datetime,
) -> list[ContextField]:
    session_date = anchor.astimezone(NY).date().isoformat()
    cutoff = anchor - BAR_DURATION
    cutoff_text = cutoff.isoformat().replace("+00:00", "Z")
    coverage = market.execute(
        """
        SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts
        FROM spy_bars_15m WHERE symbol = ?
        """,
        (symbol,),
    ).fetchone()
    min_ts = coverage["min_ts"] if coverage else None
    max_ts = coverage["max_ts"] if coverage else None
    coverage_quality = "covered"
    if not max_ts:
        coverage_quality = "symbol_unavailable"
    elif anchor > _parse_utc(max_ts) + BAR_DURATION:
        coverage_quality = "anchor_after_coverage"
    elif min_ts and anchor < _parse_utc(min_ts):
        coverage_quality = "anchor_before_coverage"

    fields = [
        ContextField(
            field_name="intraday_data_coverage",
            status="observed" if max_ts else "unavailable",
            value={"earliest_bar_start": min_ts, "latest_bar_start": max_ts}
            if max_ts
            else None,
            source_database="spy_truth.db",
            source_table="spy_bars_15m",
            source_column="ts",
            source_record_time=max_ts,
            selection_method="symbol_min_max_bar_start_v1",
            offset_seconds=(anchor - (_parse_utc(max_ts) + BAR_DURATION)).total_seconds()
            if max_ts
            else None,
            quality_status=coverage_quality,
            reason="intraday_table_coverage_relative_to_anchor",
        )
    ]
    bar = market.execute(
        """
        SELECT ts, close, vwap
        FROM spy_bars_15m
        WHERE symbol = ? AND session_date = ? AND ts <= ?
        ORDER BY ts DESC LIMIT 1
        """,
        (symbol, session_date, cutoff_text),
    ).fetchone()
    if bar is None:
        status = "stale" if coverage_quality == "anchor_after_coverage" else "unavailable"
        reason = (
            "anchor_after_intraday_coverage"
            if status == "stale"
            else "no_completed_same_session_bar_at_anchor"
        )
        for field_name, column in (
            ("underlying_price", "close"),
            ("completed_bar_vwap", "vwap"),
        ):
            fields.append(
                ContextField(
                    field_name=field_name,
                    status=status,
                    value=None,
                    source_database="spy_truth.db",
                    source_table="spy_bars_15m",
                    source_column=column,
                    source_record_time=max_ts,
                    selection_method="latest_completed_same_session_15m_bar_v1",
                    offset_seconds=None,
                    quality_status=coverage_quality,
                    reason=reason,
                )
            )
        return fields

    bar_complete = _parse_utc(bar["ts"]) + BAR_DURATION
    offset = (anchor - bar_complete).total_seconds()
    fields.append(
        ContextField(
            field_name="underlying_price",
            status="observed",
            value=bar["close"],
            source_database="spy_truth.db",
            source_table="spy_bars_15m",
            source_column="close",
            source_record_time=bar["ts"],
            selection_method="latest_completed_same_session_15m_bar_v1",
            offset_seconds=offset,
            quality_status="completed_bar",
            reason="bar_end_not_later_than_anchor",
        )
    )
    fields.append(
        ContextField(
            field_name="completed_bar_vwap",
            status="observed" if bar["vwap"] is not None else "missing",
            value=bar["vwap"],
            source_database="spy_truth.db",
            source_table="spy_bars_15m",
            source_column="vwap",
            source_record_time=bar["ts"],
            selection_method="latest_completed_same_session_15m_bar_v1",
            offset_seconds=offset,
            quality_status="completed_bar",
            reason="bar_local_vwap_not_session_cumulative_vwap",
        )
    )
    return fields


def _prior_regime_field(
    market: sqlite3.Connection,
    symbol: str,
    anchor: datetime,
) -> ContextField:
    session_date = anchor.astimezone(NY).date().isoformat()
    anchor_text = anchor.isoformat().replace("+00:00", "Z")
    row = market.execute(
        """
        SELECT date, regime_label, regime_ts
        FROM regime_daily
        WHERE symbol = ? AND date < ? AND regime_ts <= ?
        ORDER BY date DESC LIMIT 1
        """,
        (symbol, session_date, anchor_text),
    ).fetchone()
    if row is None:
        return ContextField(
            "prior_session_regime",
            "unavailable",
            None,
            "spy_truth.db",
            "regime_daily",
            "regime_label",
            None,
            "latest_prior_date_with_asof_regime_ts_v1",
            None,
            "no_asof_safe_row",
            "no_prior_regime_available_by_anchor",
        )
    regime_time = _parse_utc(row["regime_ts"])
    age = anchor - regime_time
    stale = age > MAX_PRIOR_REGIME_AGE
    return ContextField(
        "prior_session_regime",
        "stale" if stale else "observed",
        None if stale else row["regime_label"],
        "spy_truth.db",
        "regime_daily",
        "regime_label",
        row["regime_ts"],
        "latest_prior_date_with_asof_regime_ts_v1",
        age.total_seconds(),
        "prior_regime_too_old" if stale else "asof_safe_prior_session",
        "regime_value_withheld_when_older_than_seven_days" if stale else "prior_session_only",
    )


def _open_resolution_field(
    market: sqlite3.Connection,
    symbol: str,
    anchor: datetime,
) -> ContextField:
    session_date = anchor.astimezone(NY).date().isoformat()
    anchor_text = anchor.isoformat().replace("+00:00", "Z")
    row = market.execute(
        """
        SELECT snapshot_ts, open_regime_label
        FROM open_resolution_regime
        WHERE underlying = ? AND session_date = ? AND snapshot_ts <= ?
        ORDER BY snapshot_ts DESC LIMIT 1
        """,
        (symbol, session_date, anchor_text),
    ).fetchone()
    if row is None:
        return ContextField(
            "open_resolution_state",
            "unavailable",
            None,
            "spy_truth.db",
            "open_resolution_regime",
            "open_regime_label",
            None,
            "latest_same_session_snapshot_not_after_anchor_v1",
            None,
            "no_asof_safe_same_session_row",
            "open_resolution_not_available_at_anchor",
        )
    offset = (anchor - _parse_utc(row["snapshot_ts"])).total_seconds()
    return ContextField(
        "open_resolution_state",
        "observed",
        row["open_regime_label"],
        "spy_truth.db",
        "open_resolution_regime",
        "open_regime_label",
        row["snapshot_ts"],
        "latest_same_session_snapshot_not_after_anchor_v1",
        offset,
        "asof_safe_same_session_snapshot",
        "snapshot_existed_by_anchor",
    )


def _opening_range_field() -> ContextField:
    return ContextField(
        "opening_range_relationship",
        "unavailable",
        None,
        "spy_truth.db",
        "none",
        None,
        None,
        "no_canonical_asof_relationship_v1",
        None,
        "unsupported_without_lookahead",
        "no_canonical_anchor_time_opening_range_relationship_available",
    )


def _store_field(
    connection: sqlite3.Connection,
    capture_id: str,
    post_id: str,
    field: ContextField,
) -> None:
    connection.execute(
        """
        INSERT INTO market_context_fields (
            capture_id, post_id, context_policy_version, field_name,
            status, value_json, source_database, source_table, source_column,
            source_record_time, selection_method, offset_seconds,
            quality_status, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            capture_id,
            post_id,
            CONTEXT_POLICY_VERSION,
            field.field_name,
            field.status,
            json.dumps(field.value) if field.value is not None else None,
            field.source_database,
            field.source_table,
            field.source_column,
            field.source_record_time,
            field.selection_method,
            field.offset_seconds,
            field.quality_status,
            field.reason,
        ),
    )


def attach_pending_market_context(
    social_db_path: str | Path,
    market_db_path: str | Path,
) -> dict[str, Any]:
    """Attach as-of context without modifying or forward-reading market truth."""
    market_path = Path(market_db_path).expanduser()
    market_hash = _database_sha256(market_path)
    attached = 0
    field_status_counts = {status: 0 for status in ("observed", "missing", "unavailable", "stale")}
    with _open_market_truth(market_path) as market, connect(social_db_path) as social:
        ensure_schema(social)
        pending = social.execute(
            """
            SELECT eligibility.capture_id, eligibility.post_id,
                   eligibility.effective_anchor_at_utc,
                   claims.value_json AS symbol_json
            FROM evaluation_eligibility AS eligibility
            JOIN claim_fields AS claims
              ON claims.capture_id = eligibility.capture_id
             AND claims.post_id = eligibility.post_id
             AND claims.field_name = 'symbol'
             AND claims.status = 'observed'
            WHERE eligibility.evaluation_scope = 'market_context_join'
              AND eligibility.policy_version = ?
              AND eligibility.status = 'eligible'
              AND NOT EXISTS (
                  SELECT 1 FROM market_context_snapshots AS context
                  WHERE context.capture_id = eligibility.capture_id
                    AND context.post_id = eligibility.post_id
                    AND context.context_policy_version = ?
              )
            ORDER BY eligibility.effective_anchor_at_utc, eligibility.post_id
            """,
            (ELIGIBILITY_POLICY_VERSION, CONTEXT_POLICY_VERSION),
        ).fetchall()
        for observation in pending:
            anchor = _parse_utc(observation["effective_anchor_at_utc"])
            symbol = json.loads(observation["symbol_json"])
            fields = [
                _session_clock_field(anchor),
                *_intraday_fields(market, symbol, anchor),
                _prior_regime_field(market, symbol, anchor),
                _open_resolution_field(market, symbol, anchor),
                _opening_range_field(),
            ]
            reasons = sorted({field.quality_status for field in fields})
            overall_quality = (
                "context_available"
                if any(
                    field.field_name == "underlying_price" and field.status == "observed"
                    for field in fields
                )
                else "market_price_unavailable"
            )
            social.execute(
                """
                INSERT INTO market_context_snapshots (
                    capture_id, post_id, context_policy_version,
                    research_family, anchor_at_utc, symbol, source_database,
                    source_db_sha256, overall_quality, reasons_json, attached_at_utc
                ) VALUES (?, ?, ?, 'social_information_value', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation["capture_id"],
                    observation["post_id"],
                    CONTEXT_POLICY_VERSION,
                    observation["effective_anchor_at_utc"],
                    symbol,
                    "spy_truth.db",
                    market_hash,
                    overall_quality,
                    json.dumps(reasons),
                    _utc_timestamp(),
                ),
            )
            for field in fields:
                _store_field(social, observation["capture_id"], observation["post_id"], field)
                field_status_counts[field.status] += 1
            attached += 1
    return {
        "status": "attached",
        "attached_observations": attached,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "research_family": "social_information_value",
        "field_status_counts": field_status_counts,
        "market_truth_open_mode": "read_only_query_only",
    }
