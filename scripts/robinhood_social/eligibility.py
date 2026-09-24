"""Auditable temporal anchors and pre-market evaluation eligibility."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .claims import EXTRACTOR_VERSION
from .evidence_store import connect

POLICY_VERSION = 1
EVALUATION_SCOPES = (
    "market_context_join",
    "instrument_path_join",
    "directional_evaluation",
    "precommitment_evaluation",
)


def ensure_schema(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS temporal_anchors (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            anchor_kind TEXT NOT NULL,
            source_authority TEXT NOT NULL,
            source_capture_id TEXT NOT NULL,
            raw_value_json TEXT,
            resolved_at_utc TEXT,
            interval_start_utc TEXT,
            interval_end_utc TEXT,
            precision TEXT NOT NULL,
            source_field TEXT NOT NULL,
            evidence_start INTEGER,
            evidence_end INTEGER,
            evidence_text TEXT,
            extraction_method TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            can_establish_publication_time INTEGER NOT NULL
                CHECK (can_establish_publication_time IN (0, 1)),
            PRIMARY KEY (capture_id, post_id, anchor_kind),
            FOREIGN KEY (capture_id, post_id)
                REFERENCES post_observations(capture_id, post_id)
        );

        CREATE TABLE IF NOT EXISTS evaluation_eligibility (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            evaluation_scope TEXT NOT NULL,
            policy_version INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('eligible', 'ineligible')),
            effective_anchor_kind TEXT,
            effective_anchor_at_utc TEXT,
            reasons_json TEXT NOT NULL,
            assessed_at_utc TEXT NOT NULL,
            PRIMARY KEY (capture_id, post_id, evaluation_scope, policy_version),
            FOREIGN KEY (capture_id, post_id)
                REFERENCES post_observations(capture_id, post_id)
        );
        """
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _field_map(
    connection: Any, capture_id: str, post_id: str
) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT field_name, status, value_json, source_field, evidence_start,
               evidence_end, evidence_text, extraction_method, confidence
        FROM claim_fields
        WHERE capture_id = ? AND post_id = ?
        """,
        (capture_id, post_id),
    ).fetchall()
    return {
        row["field_name"]: {
            "status": row["status"],
            "value": json.loads(row["value_json"]) if row["value_json"] else None,
            "source_field": row["source_field"],
            "evidence_start": row["evidence_start"],
            "evidence_end": row["evidence_end"],
            "evidence_text": row["evidence_text"],
            "extraction_method": row["extraction_method"],
            "confidence": row["confidence"],
        }
        for row in rows
    }


def _is_observed(fields: dict[str, dict[str, Any]], name: str) -> bool:
    return fields.get(name, {}).get("status") == "observed"


def _insert_anchor(
    connection: Any,
    *,
    capture_id: str,
    post_id: str,
    anchor_kind: str,
    source_authority: str,
    source_capture_id: str,
    raw_value: Any,
    resolved_at_utc: str | None,
    precision: str,
    source_field: str,
    evidence_start: int | None,
    evidence_end: int | None,
    evidence_text: str | None,
    extraction_method: str,
    confidence: float,
    can_establish_publication_time: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO temporal_anchors (
            capture_id, post_id, anchor_kind, source_authority,
            source_capture_id, raw_value_json, resolved_at_utc,
            interval_start_utc, interval_end_utc, precision, source_field,
            evidence_start, evidence_end, evidence_text, extraction_method,
            confidence, can_establish_publication_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(capture_id, post_id, anchor_kind) DO UPDATE SET
            source_authority = excluded.source_authority,
            source_capture_id = excluded.source_capture_id,
            raw_value_json = excluded.raw_value_json,
            resolved_at_utc = excluded.resolved_at_utc,
            interval_start_utc = NULL,
            interval_end_utc = NULL,
            precision = excluded.precision,
            source_field = excluded.source_field,
            evidence_start = excluded.evidence_start,
            evidence_end = excluded.evidence_end,
            evidence_text = excluded.evidence_text,
            extraction_method = excluded.extraction_method,
            confidence = excluded.confidence,
            can_establish_publication_time = excluded.can_establish_publication_time
        """,
        (
            capture_id,
            post_id,
            anchor_kind,
            source_authority,
            source_capture_id,
            json.dumps(raw_value) if raw_value is not None else None,
            resolved_at_utc,
            precision,
            source_field,
            evidence_start,
            evidence_end,
            evidence_text,
            extraction_method,
            confidence,
            int(can_establish_publication_time),
        ),
    )


def _store_anchors(
    connection: Any,
    observation: Any,
    fields: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    first_revision = connection.execute(
        """
        SELECT capture_id, observed_at_utc
        FROM post_observations
        WHERE post_id = ? AND content_fingerprint = ?
        ORDER BY observed_at_utc, capture_id
        LIMIT 1
        """,
        (observation["post_id"], observation["content_fingerprint"]),
    ).fetchone()
    first_capture_id = first_revision["capture_id"]
    first_observed_at = first_revision["observed_at_utc"]
    _insert_anchor(
        connection,
        capture_id=observation["capture_id"],
        post_id=observation["post_id"],
        anchor_kind="independent_revision_observation",
        source_authority="sharpedge_capture_clock",
        source_capture_id=first_capture_id,
        raw_value=first_observed_at,
        resolved_at_utc=first_observed_at,
        precision="capture_second",
        source_field="post_observations.observed_at_utc",
        evidence_start=None,
        evidence_end=None,
        evidence_text=None,
        extraction_method="first_observed_content_fingerprint_v1",
        confidence=1.0,
        can_establish_publication_time=False,
    )

    payload = json.loads(observation["normalized_payload_json"])
    displayed_age = payload.get("observed_age")
    if displayed_age:
        _insert_anchor(
            connection,
            capture_id=observation["capture_id"],
            post_id=observation["post_id"],
            anchor_kind="platform_displayed_relative_age",
            source_authority="robinhood_visible_ui",
            source_capture_id=observation["capture_id"],
            raw_value=displayed_age,
            resolved_at_utc=None,
            precision="unresolved_platform_relative_label",
            source_field="normalized_payload.observed_age",
            evidence_start=None,
            evidence_end=None,
            evidence_text=displayed_age,
            extraction_method="normalized_header_age_v1",
            confidence=1.0,
            can_establish_publication_time=False,
        )

    temporal = fields.get("temporal_language")
    if temporal and temporal["status"] == "observed":
        _insert_anchor(
            connection,
            capture_id=observation["capture_id"],
            post_id=observation["post_id"],
            anchor_kind="creator_authored_temporal_language",
            source_authority="creator_authored_text",
            source_capture_id=observation["capture_id"],
            raw_value=temporal["value"],
            resolved_at_utc=None,
            precision="unresolved_authored_language",
            source_field=temporal["source_field"],
            evidence_start=temporal["evidence_start"],
            evidence_end=temporal["evidence_end"],
            evidence_text=temporal["evidence_text"],
            extraction_method=temporal["extraction_method"],
            confidence=temporal["confidence"],
            can_establish_publication_time=False,
        )
    return first_capture_id, first_observed_at


def _assess_scopes(
    fields: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, list[str]]]:
    has_symbol = _is_observed(fields, "symbol")
    has_asset_type = _is_observed(fields, "asset_type")
    asset_type = fields.get("asset_type", {}).get("value")

    market_reasons = (
        ["symbol_observed", "independent_revision_anchor_available"]
        if has_symbol
        else ["symbol_missing"]
    )
    market_status = "eligible" if has_symbol else "ineligible"

    if not has_symbol:
        instrument_status = "ineligible"
        instrument_reasons = ["symbol_missing"]
    elif not has_asset_type:
        instrument_status = "ineligible"
        instrument_reasons = ["asset_type_missing"]
    elif asset_type == "equity":
        instrument_status = "eligible"
        instrument_reasons = ["explicit_equity_symbol_observed"]
    elif asset_type == "option":
        required = ("option_type", "strike", "expiration", "dte")
        missing = [name for name in required if not _is_observed(fields, name)]
        instrument_status = "ineligible" if missing else "eligible"
        instrument_reasons = (
            [f"option_field_missing:{name}" for name in missing]
            if missing
            else ["complete_explicit_option_contract_observed"]
        )
    else:
        instrument_status = "ineligible"
        instrument_reasons = ["unsupported_asset_type"]

    has_action = _is_observed(fields, "action")
    has_entry = _is_observed(fields, "claimed_entry")
    directional_reasons = []
    if instrument_status != "eligible":
        directional_reasons.append("instrument_path_ineligible")
    if not has_action:
        directional_reasons.append("action_missing")
    if not has_entry:
        directional_reasons.append("explicit_claimed_entry_missing")
    directional_status = "eligible" if not directional_reasons else "ineligible"
    if directional_status == "eligible":
        directional_reasons = [
            "instrument_path_eligible",
            "action_observed",
            "explicit_claimed_entry_observed",
            "independent_revision_anchor_available",
        ]

    precommitment_reasons = ["exact_platform_publication_time_unavailable"]
    if directional_status != "eligible":
        precommitment_reasons.append("directional_evaluation_ineligible")

    return {
        "market_context_join": (market_status, market_reasons),
        "instrument_path_join": (instrument_status, instrument_reasons),
        "directional_evaluation": (directional_status, directional_reasons),
        "precommitment_evaluation": ("ineligible", precommitment_reasons),
    }


def assess_pending_eligibility(db_path: str) -> dict[str, Any]:
    """Assess methodological eligibility without reading or joining market truth."""
    processed = 0
    eligible = 0
    ineligible = 0
    with connect(db_path) as connection:
        ensure_schema(connection)
        pending = connection.execute(
            """
            SELECT observation.capture_id, observation.post_id,
                   observation.observed_at_utc, observation.content_fingerprint,
                   observation.normalized_payload_json
            FROM post_observations AS observation
            JOIN claim_extractions AS claims
              ON claims.capture_id = observation.capture_id
             AND claims.post_id = observation.post_id
             AND claims.extractor_version = ?
            WHERE NOT EXISTS (
                SELECT 1 FROM evaluation_eligibility AS eligibility
                WHERE eligibility.capture_id = observation.capture_id
                  AND eligibility.post_id = observation.post_id
                  AND eligibility.policy_version = ?
            )
            ORDER BY observation.observed_at_utc, observation.post_id
            """,
            (EXTRACTOR_VERSION, POLICY_VERSION),
        ).fetchall()
        for observation in pending:
            fields = _field_map(
                connection, observation["capture_id"], observation["post_id"]
            )
            _, first_observed_at = _store_anchors(connection, observation, fields)
            scopes = _assess_scopes(fields)
            assessed_at = _utc_timestamp()
            connection.executemany(
                """
                INSERT INTO evaluation_eligibility (
                    capture_id, post_id, evaluation_scope, policy_version,
                    status, effective_anchor_kind, effective_anchor_at_utc,
                    reasons_json, assessed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        observation["capture_id"],
                        observation["post_id"],
                        scope,
                        POLICY_VERSION,
                        status,
                        "independent_revision_observation"
                        if status == "eligible"
                        else None,
                        first_observed_at if status == "eligible" else None,
                        json.dumps(reasons, sort_keys=True),
                        assessed_at,
                    )
                    for scope, (status, reasons) in scopes.items()
                ],
            )
            processed += 1
            eligible += sum(status == "eligible" for status, _ in scopes.values())
            ineligible += sum(status == "ineligible" for status, _ in scopes.values())
    return {
        "status": "assessed",
        "processed_observations": processed,
        "eligible_scope_decisions": eligible,
        "ineligible_scope_decisions": ineligible,
        "policy_version": POLICY_VERSION,
        "scope_count": len(EVALUATION_SCOPES),
    }
