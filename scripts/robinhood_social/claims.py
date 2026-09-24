"""Field-provenanced semantic claim extraction from normalized observations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from .evidence_store import connect

EXTRACTOR_VERSION = 2
FIELD_NAMES = (
    "symbol",
    "asset_type",
    "action",
    "option_type",
    "strike",
    "expiration",
    "dte",
    "quantity",
    "claimed_entry",
    "claimed_exit",
    "thesis_text",
    "risk_invalidation_text",
    "conviction_language",
    "temporal_language",
)
OPTION_INSTRUMENT = re.compile(
    r"^(?P<symbol>[A-Z][A-Z0-9.]{0,9})\s+\$(?P<strike>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<option_type>Call|Put)\s+"
    r"(?P<expiration>\d{1,2}/\d{1,2}(?:/(?:\d{2}|\d{4}))?)$",
    re.IGNORECASE,
)
EQUITY_INSTRUMENT = re.compile(r"^(?P<symbol>[A-Z][A-Z0-9.]{0,9})$")
TRADE_ACTION = re.compile(
    r"^(?P<action>Buy|Sell)\s+(?P<quantity>[\d,]+)\s+@\s+\$(?P<price>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
ENTRY_PRICE = re.compile(
    r"\b(?:entered|entry)\s+(?:at|@)\s+\$?(?P<price>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
EXIT_PRICE = re.compile(
    r"\b(?:exited|closed)\s+(?:at|@)\s+\$?(?P<price>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
THESIS_CUE = re.compile(
    r"\b(?:because|expect(?:ing)?|thesis|target|I think|should move)\b", re.IGNORECASE
)
RISK_CUE = re.compile(
    r"\b(?:invalidat(?:e|ed|ion)|stop(?:ped)?(?:\s+at)?|cut if|risking|I'm out if)\b",
    re.IGNORECASE,
)
CONVICTION_CUE = re.compile(
    r"\b(?:high conviction|strong conviction|very confident|confident|lotto)\b",
    re.IGNORECASE,
)
TEMPORAL_CUE = re.compile(
    r"\b(?:today|tomorrow|this week|next week|before close|after open|"
    r"\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FieldClaim:
    field_name: str
    status: str
    value: Any = None
    source_field: str | None = None
    evidence_start: int | None = None
    evidence_end: int | None = None
    evidence_text: str | None = None
    extraction_method: str = "no_supported_evidence_v1"
    confidence: float = 0.0


def _missing(field_name: str) -> FieldClaim:
    return FieldClaim(field_name=field_name, status="missing")


def _from_match(
    field_name: str,
    value: Any,
    source_field: str,
    source_text: str,
    match: re.Match[str],
    *,
    group: str | int = 0,
    method: str,
    confidence: float,
) -> FieldClaim:
    start, end = match.span(group)
    return FieldClaim(
        field_name=field_name,
        status="observed",
        value=value,
        source_field=source_field,
        evidence_start=start,
        evidence_end=end,
        evidence_text=source_text[start:end],
        extraction_method=method,
        confidence=confidence,
    )


def _sentence_claim(
    field_name: str,
    commentary: str,
    cue: re.Pattern[str],
    method: str,
    confidence: float,
) -> FieldClaim:
    for sentence_match in SENTENCE.finditer(commentary):
        sentence = sentence_match.group(0)
        if cue.search(sentence):
            start, end = sentence_match.span()
            return FieldClaim(
                field_name=field_name,
                status="observed",
                value=sentence.strip(),
                source_field="commentary",
                evidence_start=start,
                evidence_end=end,
                evidence_text=commentary[start:end],
                extraction_method=method,
                confidence=confidence,
            )
    return _missing(field_name)


def _expiration_date(raw: str) -> date | None:
    if raw.count("/") != 2:
        return None
    try:
        month, day, year = (int(part) for part in raw.split("/"))
        if year < 100:
            year += 2000
        return date(year, month, day)
    except ValueError:
        return None


def extract_claim_fields(
    normalized_payload: dict[str, Any], observed_at_utc: str
) -> list[FieldClaim]:
    """Extract auditable claims without evaluating correctness or outcomes."""
    claims = {field_name: _missing(field_name) for field_name in FIELD_NAMES}
    instrument = normalized_payload.get("instrument") or ""
    commentary = normalized_payload.get("commentary") or ""
    trade_action = normalized_payload.get("trade_action") or ""

    option_match = OPTION_INSTRUMENT.fullmatch(instrument)
    equity_match = EQUITY_INSTRUMENT.fullmatch(instrument)
    if option_match:
        claims["symbol"] = _from_match(
            "symbol",
            option_match.group("symbol").upper(),
            "instrument",
            instrument,
            option_match,
            group="symbol",
            method="explicit_option_instrument_v1",
            confidence=1.0,
        )
        claims["asset_type"] = _from_match(
            "asset_type",
            "option",
            "instrument",
            instrument,
            option_match,
            group="option_type",
            method="explicit_option_marker_v1",
            confidence=1.0,
        )
        claims["option_type"] = _from_match(
            "option_type",
            option_match.group("option_type").casefold(),
            "instrument",
            instrument,
            option_match,
            group="option_type",
            method="explicit_option_marker_v1",
            confidence=1.0,
        )
        claims["strike"] = _from_match(
            "strike",
            float(option_match.group("strike").replace(",", "")),
            "instrument",
            instrument,
            option_match,
            group="strike",
            method="explicit_option_instrument_v1",
            confidence=1.0,
        )
        expiration_raw = option_match.group("expiration")
        claims["expiration"] = _from_match(
            "expiration",
            expiration_raw,
            "instrument",
            instrument,
            option_match,
            group="expiration",
            method="explicit_visible_expiration_v1",
            confidence=1.0,
        )
        expiration = _expiration_date(expiration_raw)
        if expiration is not None:
            observed_date = (
                datetime.fromisoformat(observed_at_utc.replace("Z", "+00:00"))
                .astimezone(NY)
                .date()
            )
            dte = (expiration - observed_date).days
            claims["dte"] = _from_match(
                "dte",
                dte,
                "instrument",
                instrument,
                option_match,
                group="expiration",
                method="explicit_expiration_ny_calendar_v1",
                confidence=0.95,
            )
    elif equity_match:
        claims["symbol"] = _from_match(
            "symbol",
            equity_match.group("symbol").upper(),
            "instrument",
            instrument,
            equity_match,
            group="symbol",
            method="explicit_equity_instrument_v1",
            confidence=0.95,
        )
        claims["asset_type"] = _from_match(
            "asset_type",
            "equity",
            "instrument",
            instrument,
            equity_match,
            group="symbol",
            method="explicit_equity_instrument_v1",
            confidence=0.9,
        )

    action_match = TRADE_ACTION.fullmatch(trade_action)
    if action_match:
        claims["action"] = _from_match(
            "action",
            action_match.group("action").casefold(),
            "trade_action",
            trade_action,
            action_match,
            group="action",
            method="explicit_trade_card_action_v1",
            confidence=1.0,
        )
        claims["quantity"] = _from_match(
            "quantity",
            int(action_match.group("quantity").replace(",", "")),
            "trade_action",
            trade_action,
            action_match,
            group="quantity",
            method="explicit_trade_card_quantity_v1",
            confidence=1.0,
        )

    entry_match = ENTRY_PRICE.search(commentary)
    if entry_match:
        claims["claimed_entry"] = _from_match(
            "claimed_entry",
            float(entry_match.group("price")),
            "commentary",
            commentary,
            entry_match,
            group="price",
            method="explicit_authored_entry_price_v1",
            confidence=0.95,
        )
    exit_match = EXIT_PRICE.search(commentary)
    if exit_match:
        claims["claimed_exit"] = _from_match(
            "claimed_exit",
            float(exit_match.group("price")),
            "commentary",
            commentary,
            exit_match,
            group="price",
            method="explicit_authored_exit_price_v1",
            confidence=0.95,
        )

    claims["thesis_text"] = _sentence_claim(
        "thesis_text",
        commentary,
        THESIS_CUE,
        "authored_thesis_sentence_v1",
        0.8,
    )
    claims["risk_invalidation_text"] = _sentence_claim(
        "risk_invalidation_text",
        commentary,
        RISK_CUE,
        "authored_risk_sentence_v1",
        0.8,
    )
    conviction_match = CONVICTION_CUE.search(commentary)
    if conviction_match:
        claims["conviction_language"] = _from_match(
            "conviction_language",
            conviction_match.group(0),
            "commentary",
            commentary,
            conviction_match,
            method="explicit_conviction_phrase_v1",
            confidence=0.9,
        )
    temporal_match = TEMPORAL_CUE.search(commentary)
    if temporal_match:
        claims["temporal_language"] = _from_match(
            "temporal_language",
            temporal_match.group(0),
            "commentary",
            commentary,
            temporal_match,
            method="explicit_temporal_phrase_v1",
            confidence=0.9,
        )
    return [claims[field_name] for field_name in FIELD_NAMES]


def ensure_schema(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS claim_extractions (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            extractor_version INTEGER NOT NULL,
            extracted_at_utc TEXT NOT NULL,
            PRIMARY KEY (capture_id, post_id),
            FOREIGN KEY (capture_id, post_id)
                REFERENCES post_observations(capture_id, post_id)
        );

        CREATE TABLE IF NOT EXISTS claim_fields (
            capture_id TEXT NOT NULL,
            post_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('observed', 'missing', 'ambiguous')),
            value_json TEXT,
            source_field TEXT,
            evidence_start INTEGER,
            evidence_end INTEGER,
            evidence_text TEXT,
            extraction_method TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            PRIMARY KEY (capture_id, post_id, field_name),
            FOREIGN KEY (capture_id, post_id)
                REFERENCES post_observations(capture_id, post_id),
            CHECK (
                status != 'missing' OR (
                    value_json IS NULL AND source_field IS NULL
                    AND evidence_start IS NULL AND evidence_end IS NULL
                    AND evidence_text IS NULL
                )
            )
        );
        """
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def extract_pending_claims(db_path: str) -> dict[str, Any]:
    """Extract claims for every normalized observation not already processed."""
    processed = 0
    observed_fields = 0
    missing_fields = 0
    with connect(db_path) as connection:
        ensure_schema(connection)
        pending = connection.execute(
            """
            SELECT observation.capture_id, observation.post_id,
                   observation.observed_at_utc, observation.normalized_payload_json
            FROM post_observations AS observation
            LEFT JOIN claim_extractions AS extraction
              ON extraction.capture_id = observation.capture_id
             AND extraction.post_id = observation.post_id
            WHERE extraction.capture_id IS NULL
               OR extraction.extractor_version < ?
            ORDER BY observation.observed_at_utc, observation.post_id
            """,
            (EXTRACTOR_VERSION,),
        ).fetchall()
        for observation in pending:
            fields = extract_claim_fields(
                json.loads(observation["normalized_payload_json"]),
                observation["observed_at_utc"],
            )
            connection.execute(
                "DELETE FROM claim_fields WHERE capture_id = ? AND post_id = ?",
                (observation["capture_id"], observation["post_id"]),
            )
            connection.execute(
                """
                INSERT INTO claim_extractions (
                    capture_id, post_id, extractor_version, extracted_at_utc
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(capture_id, post_id) DO UPDATE SET
                    extractor_version = excluded.extractor_version,
                    extracted_at_utc = excluded.extracted_at_utc
                """,
                (
                    observation["capture_id"],
                    observation["post_id"],
                    EXTRACTOR_VERSION,
                    _utc_timestamp(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO claim_fields (
                    capture_id, post_id, field_name, status, value_json,
                    source_field, evidence_start, evidence_end, evidence_text,
                    extraction_method, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        observation["capture_id"],
                        observation["post_id"],
                        field.field_name,
                        field.status,
                        json.dumps(field.value) if field.status != "missing" else None,
                        field.source_field,
                        field.evidence_start,
                        field.evidence_end,
                        field.evidence_text,
                        field.extraction_method,
                        field.confidence,
                    )
                    for field in fields
                ],
            )
            processed += 1
            observed_fields += sum(field.status == "observed" for field in fields)
            missing_fields += sum(field.status == "missing" for field in fields)
    return {
        "status": "extracted",
        "processed_observations": processed,
        "observed_fields": observed_fields,
        "missing_fields": missing_fields,
        "supported_field_count": len(FIELD_NAMES),
    }
