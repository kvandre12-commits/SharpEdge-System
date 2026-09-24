from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from robinhood_social.claims import (
    FIELD_NAMES,
    extract_claim_fields,
    extract_pending_claims,
)


def by_name(fields):
    return {field.field_name: field for field in fields}


def payload(**overrides):
    base = {
        "creator_handle": "OptionDog",
        "normalized_handle": "optiondog",
        "observed_age": "2d",
        "commentary": "Watching this because momentum is strong today.",
        "instrument": "SPY $650 Call 9/18",
        "trade_action": "Sell 2 @ $3.25",
        "displayed_pnl": "+$500.00",
        "identity_basis": "creator_trade_signature",
        "identity_confidence": 0.7,
    }
    base.update(overrides)
    return base


def test_option_card_extracts_literal_fields_without_lifecycle_inference():
    fields = by_name(extract_claim_fields(payload(), "2026-09-10T14:00:00Z"))

    assert fields["symbol"].value == "SPY"
    assert fields["asset_type"].value == "option"
    assert fields["option_type"].value == "call"
    assert fields["strike"].value == 650.0
    assert fields["expiration"].value == "9/18"
    assert fields["action"].value == "sell"
    assert fields["quantity"].value == 2
    assert fields["dte"].status == "missing"
    assert fields["claimed_entry"].status == "missing"
    assert fields["claimed_exit"].status == "missing"
    assert fields["thesis_text"].status == "observed"
    assert fields["temporal_language"].value.casefold() == "today"


def test_full_visible_expiration_supports_dte_with_provenance():
    instrument = "SPY $8,000 Put 9/18/2026"
    fields = by_name(
        extract_claim_fields(
            payload(instrument=instrument, trade_action="Buy 1 @ $2.00"),
            "2026-09-10T14:00:00Z",
        )
    )

    assert fields["strike"].value == 8000.0
    assert fields["dte"].value == 8
    assert fields["dte"].evidence_text == "9/18/2026"
    assert fields["dte"].extraction_method == "explicit_expiration_ny_calendar_v1"
    assert (
        instrument[fields["strike"].evidence_start : fields["strike"].evidence_end]
        == "8,000"
    )


def test_generic_calls_do_not_manufacture_symbol_or_option_contract():
    fields = by_name(
        extract_claim_fields(
            payload(
                commentary="Calls look interesting tomorrow.",
                instrument=None,
                trade_action=None,
            ),
            "2026-09-10T14:00:00Z",
        )
    )

    for field_name in (
        "symbol",
        "asset_type",
        "option_type",
        "strike",
        "expiration",
        "dte",
        "action",
        "quantity",
    ):
        assert fields[field_name].status == "missing"
    assert fields["temporal_language"].value.casefold() == "tomorrow"


def test_explicit_entry_exit_risk_and_conviction_retain_exact_spans():
    commentary = (
        "High conviction because breadth improved today. "
        "Entered at $2.10. Stop at 1.60. Closed at $3.40."
    )
    fields = by_name(
        extract_claim_fields(payload(commentary=commentary), "2026-09-10T14:00:00Z")
    )

    assert fields["claimed_entry"].value == 2.1
    assert fields["claimed_exit"].value == 3.4
    assert fields["risk_invalidation_text"].value == "Stop at 1.60."
    assert fields["conviction_language"].value.casefold() == "high conviction"
    for field_name in (
        "claimed_entry",
        "claimed_exit",
        "thesis_text",
        "risk_invalidation_text",
        "conviction_language",
        "temporal_language",
    ):
        field = fields[field_name]
        assert (
            commentary[field.evidence_start : field.evidence_end] == field.evidence_text
        )


def _seed_observations(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE post_observations (
                capture_id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                observed_at_utc TEXT NOT NULL,
                normalized_payload_json TEXT NOT NULL,
                PRIMARY KEY (capture_id, post_id)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO post_observations (
                capture_id, post_id, observed_at_utc, normalized_payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "capture-1",
                    "post-1",
                    "2026-09-10T14:00:00Z",
                    json.dumps(payload()),
                ),
                (
                    "capture-2",
                    "post-2",
                    "2026-09-10T14:05:00Z",
                    json.dumps(
                        payload(
                            commentary="Calls look interesting.",
                            instrument=None,
                            trade_action=None,
                        )
                    ),
                ),
            ],
        )


def test_pending_extraction_stores_present_and_explicit_missing_rows(tmp_path):
    db_path = tmp_path / "research.db"
    _seed_observations(db_path)

    first = extract_pending_claims(str(db_path))
    second = extract_pending_claims(str(db_path))

    assert first["processed_observations"] == 2
    assert first["observed_fields"] + first["missing_fields"] == 2 * len(FIELD_NAMES)
    assert second["processed_observations"] == 0

    with sqlite3.connect(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM claim_fields").fetchone()[0]
        missing_symbol = connection.execute(
            """
            SELECT status, value_json, source_field, evidence_text
            FROM claim_fields
            WHERE capture_id = 'capture-2' AND field_name = 'symbol'
            """
        ).fetchone()
        action = connection.execute(
            """
            SELECT value_json, source_field, evidence_start, evidence_end,
                   evidence_text, extraction_method, confidence
            FROM claim_fields
            WHERE capture_id = 'capture-1' AND field_name = 'action'
            """
        ).fetchone()

    assert total == 2 * len(FIELD_NAMES)
    assert missing_symbol == ("missing", None, None, None)
    assert json.loads(action[0]) == "sell"
    assert action[1:5] == ("trade_action", 0, 4, "Sell")
    assert action[5] == "explicit_trade_card_action_v1"
    assert action[6] == 1.0
