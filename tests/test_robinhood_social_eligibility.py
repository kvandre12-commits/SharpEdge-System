from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from robinhood_social.claims import EXTRACTOR_VERSION, FIELD_NAMES
from robinhood_social.claims import ensure_schema as ensure_claim_schema
from robinhood_social.eligibility import assess_pending_eligibility


def _create_observation_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE post_observations (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            normalized_payload_json TEXT NOT NULL,
            PRIMARY KEY (capture_id, post_id)
        )
        """
    )
    ensure_claim_schema(connection)


def _field_values(**overrides):
    fields = {name: None for name in FIELD_NAMES}
    fields.update(overrides)
    return fields


def _seed_observation(
    connection: sqlite3.Connection,
    *,
    capture_id: str,
    post_id: str,
    observed_at: str,
    content_fingerprint: str,
    observed_age: str,
    fields: dict,
) -> None:
    payload = {
        "observed_age": observed_age,
        "commentary": "Entered at $2.00 today.",
    }
    connection.execute(
        """
        INSERT INTO post_observations (
            capture_id, post_id, observed_at_utc, content_fingerprint,
            normalized_payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            capture_id,
            post_id,
            observed_at,
            content_fingerprint,
            json.dumps(payload),
        ),
    )
    connection.execute(
        """
        INSERT INTO claim_extractions (
            capture_id, post_id, extractor_version, extracted_at_utc
        ) VALUES (?, ?, ?, ?)
        """,
        (capture_id, post_id, EXTRACTOR_VERSION, observed_at),
    )
    for field_name, value in fields.items():
        observed = value is not None
        source_field = (
            "commentary"
            if field_name
            in {
                "claimed_entry",
                "temporal_language",
            }
            else "instrument"
        )
        evidence = str(value) if observed else None
        connection.execute(
            """
            INSERT INTO claim_fields (
                capture_id, post_id, field_name, status, value_json,
                source_field, evidence_start, evidence_end, evidence_text,
                extraction_method, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capture_id,
                post_id,
                field_name,
                "observed" if observed else "missing",
                json.dumps(value) if observed else None,
                source_field if observed else None,
                0 if observed else None,
                len(evidence) if evidence else None,
                evidence,
                "test_explicit_v1" if observed else "no_supported_evidence_v1",
                1.0 if observed else 0.0,
            ),
        )


def test_content_revisions_receive_independent_first_observation_anchors(tmp_path):
    db = tmp_path / "research.db"
    fields = _field_values(
        symbol="SPY",
        asset_type="equity",
        action="buy",
        claimed_entry=2.0,
        temporal_language="today",
    )
    with sqlite3.connect(db) as connection:
        _create_observation_table(connection)
        _seed_observation(
            connection,
            capture_id="capture-1",
            post_id="post-1",
            observed_at="2026-09-10T14:00:00Z",
            content_fingerprint="content-a",
            observed_age="2h",
            fields=fields,
        )
        _seed_observation(
            connection,
            capture_id="capture-2",
            post_id="post-1",
            observed_at="2026-09-10T14:05:00Z",
            content_fingerprint="content-a",
            observed_age="2h",
            fields=fields,
        )
        _seed_observation(
            connection,
            capture_id="capture-3",
            post_id="post-1",
            observed_at="2026-09-10T14:10:00Z",
            content_fingerprint="content-b",
            observed_age="2h",
            fields=fields,
        )

    result = assess_pending_eligibility(str(db))

    with sqlite3.connect(db) as connection:
        repeated_anchor = connection.execute(
            """
            SELECT source_capture_id, resolved_at_utc,
                   can_establish_publication_time
            FROM temporal_anchors
            WHERE capture_id = 'capture-2'
              AND anchor_kind = 'independent_revision_observation'
            """
        ).fetchone()
        edited_anchor = connection.execute(
            """
            SELECT source_capture_id, resolved_at_utc
            FROM temporal_anchors
            WHERE capture_id = 'capture-3'
              AND anchor_kind = 'independent_revision_observation'
            """
        ).fetchone()
        relative_anchor = connection.execute(
            """
            SELECT resolved_at_utc, precision, can_establish_publication_time
            FROM temporal_anchors
            WHERE capture_id = 'capture-2'
              AND anchor_kind = 'platform_displayed_relative_age'
            """
        ).fetchone()
        authored_anchor = connection.execute(
            """
            SELECT source_authority, resolved_at_utc,
                   can_establish_publication_time, evidence_text
            FROM temporal_anchors
            WHERE capture_id = 'capture-2'
              AND anchor_kind = 'creator_authored_temporal_language'
            """
        ).fetchone()

    assert result["processed_observations"] == 3
    assert repeated_anchor == ("capture-1", "2026-09-10T14:00:00Z", 0)
    assert edited_anchor == ("capture-3", "2026-09-10T14:10:00Z")
    assert relative_anchor == (None, "unresolved_platform_relative_label", 0)
    assert authored_anchor == ("creator_authored_text", None, 0, "today")


def test_scoped_eligibility_is_conservative_and_idempotent(tmp_path):
    db = tmp_path / "research.db"
    yearless_option = _field_values(
        symbol="SPY",
        asset_type="option",
        action="buy",
        option_type="call",
        strike=650.0,
        expiration="9/18",
        claimed_entry=2.0,
    )
    with sqlite3.connect(db) as connection:
        _create_observation_table(connection)
        _seed_observation(
            connection,
            capture_id="capture-1",
            post_id="post-1",
            observed_at="2026-09-10T14:00:00Z",
            content_fingerprint="content-a",
            observed_age="2d",
            fields=yearless_option,
        )

    first = assess_pending_eligibility(str(db))
    second = assess_pending_eligibility(str(db))

    with sqlite3.connect(db) as connection:
        decisions = {
            scope: (status, json.loads(reasons), anchor)
            for scope, status, reasons, anchor in connection.execute(
                """
                SELECT evaluation_scope, status, reasons_json,
                       effective_anchor_at_utc
                FROM evaluation_eligibility
                """
            )
        }

    assert first["eligible_scope_decisions"] == 1
    assert first["ineligible_scope_decisions"] == 3
    assert second["processed_observations"] == 0
    assert decisions["market_context_join"] == (
        "eligible",
        ["symbol_observed", "independent_revision_anchor_available"],
        "2026-09-10T14:00:00Z",
    )
    assert decisions["instrument_path_join"][0] == "ineligible"
    assert "option_field_missing:dte" in decisions["instrument_path_join"][1]
    assert decisions["directional_evaluation"][0] == "ineligible"
    assert decisions["precommitment_evaluation"][0] == "ineligible"
    assert (
        "exact_platform_publication_time_unavailable"
        in decisions["precommitment_evaluation"][1]
    )
