"""Conservative identity and observation history for Robinhood Social evidence."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .evidence_store import connect

NORMALIZER_VERSION = 1
AGE_PATTERN = re.compile(r"^\d+[smhdwy]$", re.IGNORECASE)
TRADE_PATTERN = re.compile(
    r"^(Buy|Sell)\s+[\d,.]+\s+@\s+\$?[\d,.]+(?:\.\d+)?$", re.IGNORECASE
)
PNL_PATTERN = re.compile(r"^[+-]\$[\d,]+(?:\.\d+)?$")

TYPE_RULES: dict[str, tuple[re.Pattern[str], ...]] = {
    "education": (
        re.compile(r"\blesson\b", re.IGNORECASE),
        re.compile(r"\bhow to\b", re.IGNORECASE),
        re.compile(r"\bexplainer\b", re.IGNORECASE),
        re.compile(r"\bguide\b", re.IGNORECASE),
    ),
    "entry": (
        re.compile(r"\bentered\b", re.IGNORECASE),
        re.compile(r"\bopen(?:ed|ing)\b", re.IGNORECASE),
        re.compile(r"\binitiated\b", re.IGNORECASE),
        re.compile(r"\bjust bought\b", re.IGNORECASE),
    ),
    "update": (
        re.compile(r"\bholding\b", re.IGNORECASE),
        re.compile(r"\bstill hold(?:ing)?\b", re.IGNORECASE),
        re.compile(r"\badded\b", re.IGNORECASE),
        re.compile(r"\btrimmed\b", re.IGNORECASE),
        re.compile(r"\bupdate\b", re.IGNORECASE),
    ),
    "exit": (
        re.compile(r"\bclosed\b", re.IGNORECASE),
        re.compile(r"\bexited\b", re.IGNORECASE),
        re.compile(r"\btook profit\b", re.IGNORECASE),
        re.compile(r"\bstopped out\b", re.IGNORECASE),
        re.compile(r"\bsold my\b", re.IGNORECASE),
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def normalize_handle(handle: str) -> str:
    normalized = _canonical_text(handle).removeprefix("@").casefold()
    if not normalized:
        raise ValueError("creator handle is empty after normalization")
    return normalized


def classify_post_type(commentary: str) -> dict[str, Any]:
    """Classify only explicit authored-text cues; trade cards are not lifecycle proof."""
    matches: dict[str, list[str]] = {}
    for post_type, patterns in TYPE_RULES.items():
        cues = [
            match.group(0)
            for pattern in patterns
            if (match := pattern.search(commentary))
        ]
        if cues:
            matches[post_type] = cues

    if len(matches) == 1:
        post_type = next(iter(matches))
        return {
            "post_type": post_type,
            "method": "authored_text_rule_v1",
            "confidence": 0.8,
            "cues": matches,
        }
    return {
        "post_type": "unknown",
        "method": "authored_text_rule_v1",
        "confidence": 0.2 if matches else 0.0,
        "cues": matches,
    }


def _marker_count(element: ElementTree.Element, description: str) -> int:
    return sum(
        node.attrib.get("content-desc") == description for node in element.iter("node")
    )


def _post_containers(root: ElementTree.Element) -> list[ElementTree.Element]:
    containers = []
    for element in root.iter("node"):
        if element.attrib.get("clickable") != "true":
            continue
        if _marker_count(element, "view more actions") != 1:
            continue
        if _marker_count(element, "view profile") != 1:
            continue
        containers.append(element)
    return containers


def _text_elements(
    container: ElementTree.Element,
) -> list[tuple[int, ElementTree.Element, str]]:
    values = []
    for index, element in enumerate(container.iter("node")):
        text = _canonical_text(element.attrib.get("text", ""))
        if text:
            values.append((index, element, text))
    return values


def _is_noise(value: str) -> bool:
    return value == "·" or value.isdigit() or all(not char.isalnum() for char in value)


def _extract_post(container: ElementTree.Element) -> dict[str, Any] | None:
    all_elements = list(container.iter("node"))
    texts = _text_elements(container)
    if not texts:
        return None

    handle_item = next(
        (
            item
            for item in texts
            if item[1].attrib.get("clickable") == "true"
            and not PNL_PATTERN.fullmatch(item[2])
            and not item[2].isdigit()
        ),
        None,
    )
    if handle_item is None:
        return None
    handle_index, _, handle = handle_item
    normalized_handle = normalize_handle(handle)

    age_item = next(
        (
            item
            for item in texts
            if item[0] > handle_index and AGE_PATTERN.fullmatch(item[2])
        ),
        None,
    )
    content_start = age_item[0] + 1 if age_item else handle_index + 1
    observed_age = age_item[2].casefold() if age_item else None

    trade_item = next(
        (
            item
            for item in texts
            if item[0] >= content_start and TRADE_PATTERN.fullmatch(item[2])
        ),
        None,
    )
    trade_action = trade_item[2] if trade_item else None
    instrument = None
    content_end = None
    if trade_item:
        preceding = [item for item in texts if content_start <= item[0] < trade_item[0]]
        meaningful = [item for item in preceding if not _is_noise(item[2])]
        if meaningful:
            instrument = meaningful[-1][2]
            content_end = meaningful[-1][0]

    comment_index = next(
        (
            index
            for index, element in enumerate(all_elements)
            if element.attrib.get("content-desc") == "comment"
        ),
        len(all_elements),
    )
    content_end = content_end if content_end is not None else comment_index
    commentary_parts = [
        value
        for index, _, value in texts
        if content_start <= index < content_end and not _is_noise(value)
    ]
    commentary = "\n".join(commentary_parts)

    displayed_pnl = next(
        (
            value
            for index, _, value in texts
            if trade_item and index > trade_item[0] and PNL_PATTERN.fullmatch(value)
        ),
        None,
    )
    authored_payload = {
        "creator_handle": normalized_handle,
        "commentary": commentary,
        "instrument": instrument,
        "trade_action": trade_action,
    }
    content_fingerprint = _sha256_text(
        json.dumps(authored_payload, sort_keys=True, separators=(",", ":"))
    )
    display_payload = {**authored_payload, "displayed_pnl": displayed_pnl}
    display_fingerprint = _sha256_text(
        json.dumps(display_payload, sort_keys=True, separators=(",", ":"))
    )

    if instrument and trade_action:
        identity_basis = "creator_trade_signature"
        identity_confidence = 0.7
        identity_material = {
            "creator_handle": normalized_handle,
            "instrument": instrument,
            "trade_action": trade_action,
        }
    else:
        identity_basis = "creator_content_anchor"
        identity_confidence = 0.4
        identity_material = {
            "creator_handle": normalized_handle,
            "content_fingerprint": content_fingerprint,
        }
    identity_fingerprint = _sha256_text(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":"))
    )
    classification = classify_post_type(commentary)
    return {
        "creator_handle": handle,
        "normalized_handle": normalized_handle,
        "observed_age": observed_age,
        "commentary": commentary,
        "instrument": instrument,
        "trade_action": trade_action,
        "displayed_pnl": displayed_pnl,
        "identity_fingerprint": identity_fingerprint,
        "identity_basis": identity_basis,
        "identity_confidence": identity_confidence,
        "content_fingerprint": content_fingerprint,
        "display_fingerprint": display_fingerprint,
        "classification": classification,
    }


def extract_visible_posts(hierarchy_xml: bytes | str) -> list[dict[str, Any]]:
    raw = (
        hierarchy_xml.decode("utf-8")
        if isinstance(hierarchy_xml, bytes)
        else hierarchy_xml
    )
    root = ElementTree.fromstring(raw)
    return [
        post
        for container in _post_containers(root)
        if (post := _extract_post(container)) is not None
    ]


def ensure_schema(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS normalized_captures (
            capture_id TEXT PRIMARY KEY REFERENCES capture_sessions(capture_id),
            normalized_at_utc TEXT NOT NULL,
            normalizer_version INTEGER NOT NULL,
            post_count INTEGER NOT NULL CHECK (post_count >= 0)
        );

        CREATE TABLE IF NOT EXISTS creators (
            creator_id TEXT PRIMARY KEY,
            identity_basis TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS creator_aliases (
            normalized_handle TEXT PRIMARY KEY,
            creator_id TEXT NOT NULL REFERENCES creators(creator_id),
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            observation_count INTEGER NOT NULL CHECK (observation_count > 0)
        );

        CREATE TABLE IF NOT EXISTS normalized_posts (
            post_id TEXT PRIMARY KEY,
            creator_id TEXT NOT NULL REFERENCES creators(creator_id),
            identity_fingerprint TEXT NOT NULL UNIQUE,
            identity_basis TEXT NOT NULL,
            identity_confidence REAL NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            current_content_fingerprint TEXT NOT NULL,
            current_post_type TEXT NOT NULL,
            observation_count INTEGER NOT NULL CHECK (observation_count > 0)
        );

        CREATE TABLE IF NOT EXISTS post_observations (
            capture_id TEXT NOT NULL REFERENCES capture_sessions(capture_id),
            post_id TEXT NOT NULL REFERENCES normalized_posts(post_id),
            observed_at_utc TEXT NOT NULL,
            presence_status TEXT NOT NULL CHECK (presence_status = 'observed'),
            content_fingerprint TEXT NOT NULL,
            display_fingerprint TEXT NOT NULL,
            post_type TEXT NOT NULL,
            classification_method TEXT NOT NULL,
            classification_confidence REAL NOT NULL,
            classification_cues_json TEXT NOT NULL,
            normalized_payload_json TEXT NOT NULL,
            PRIMARY KEY (capture_id, post_id)
        );
        """
    )


def _creator_id(normalized_handle: str) -> str:
    return "creator_" + _sha256_text(f"visible_handle_v1:{normalized_handle}")[:24]


def _post_id(identity_fingerprint: str) -> str:
    return "post_" + identity_fingerprint[:24]


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_capture(
    db_path: Path | str,
    corpus_root: Path | str,
    capture_id: str,
) -> dict[str, Any]:
    """Normalize one retained capture, preserving append-only observation history."""
    root = Path(corpus_root).expanduser()
    with connect(db_path) as connection:
        ensure_schema(connection)
        already = connection.execute(
            "SELECT post_count FROM normalized_captures WHERE capture_id = ?",
            (capture_id,),
        ).fetchone()
        if already:
            return {
                "status": "already_normalized",
                "capture_id": capture_id,
                "post_count": already[0],
            }

        session = connection.execute(
            "SELECT observed_at_utc FROM capture_sessions WHERE capture_id = ?",
            (capture_id,),
        ).fetchone()
        if session is None:
            raise ValueError(f"unknown capture: {capture_id}")
        artifact = connection.execute(
            """
            SELECT relative_path, sha256 FROM capture_artifacts
            WHERE capture_id = ? AND artifact_kind = 'ui_hierarchy'
            """,
            (capture_id,),
        ).fetchone()
        if artifact is None:
            raise ValueError(f"capture has no UI hierarchy: {capture_id}")
        hierarchy = (root / artifact[0]).read_bytes()
        if hashlib.sha256(hierarchy).hexdigest() != artifact[1]:
            raise ValueError(f"UI hierarchy hash mismatch: {capture_id}")

        posts = extract_visible_posts(hierarchy)
        observed_at = session[0]
        for post in posts:
            creator_id = _creator_id(post["normalized_handle"])
            connection.execute(
                """
                INSERT INTO creators (
                    creator_id, identity_basis, first_seen_utc, last_seen_utc
                ) VALUES (?, 'provisional_normalized_handle', ?, ?)
                ON CONFLICT(creator_id) DO UPDATE SET last_seen_utc = excluded.last_seen_utc
                """,
                (creator_id, observed_at, observed_at),
            )
            connection.execute(
                """
                INSERT INTO creator_aliases (
                    normalized_handle, creator_id, first_seen_utc,
                    last_seen_utc, observation_count
                ) VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(normalized_handle) DO UPDATE SET
                    last_seen_utc = excluded.last_seen_utc,
                    observation_count = creator_aliases.observation_count + 1
                """,
                (post["normalized_handle"], creator_id, observed_at, observed_at),
            )

            post_id = _post_id(post["identity_fingerprint"])
            classification = post["classification"]
            connection.execute(
                """
                INSERT INTO normalized_posts (
                    post_id, creator_id, identity_fingerprint, identity_basis,
                    identity_confidence, first_seen_utc, last_seen_utc,
                    current_content_fingerprint, current_post_type, observation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(identity_fingerprint) DO UPDATE SET
                    last_seen_utc = excluded.last_seen_utc,
                    current_content_fingerprint = excluded.current_content_fingerprint,
                    current_post_type = excluded.current_post_type,
                    observation_count = normalized_posts.observation_count + 1
                """,
                (
                    post_id,
                    creator_id,
                    post["identity_fingerprint"],
                    post["identity_basis"],
                    post["identity_confidence"],
                    observed_at,
                    observed_at,
                    post["content_fingerprint"],
                    classification["post_type"],
                ),
            )
            private_payload = {
                key: post[key]
                for key in (
                    "creator_handle",
                    "normalized_handle",
                    "observed_age",
                    "commentary",
                    "instrument",
                    "trade_action",
                    "displayed_pnl",
                    "identity_basis",
                    "identity_confidence",
                )
            }
            connection.execute(
                """
                INSERT INTO post_observations (
                    capture_id, post_id, observed_at_utc, presence_status,
                    content_fingerprint, display_fingerprint, post_type,
                    classification_method, classification_confidence,
                    classification_cues_json, normalized_payload_json
                ) VALUES (?, ?, ?, 'observed', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    post_id,
                    observed_at,
                    post["content_fingerprint"],
                    post["display_fingerprint"],
                    classification["post_type"],
                    classification["method"],
                    classification["confidence"],
                    json.dumps(classification["cues"], sort_keys=True),
                    json.dumps(private_payload, sort_keys=True),
                ),
            )

        connection.execute(
            """
            INSERT INTO normalized_captures (
                capture_id, normalized_at_utc, normalizer_version, post_count
            ) VALUES (?, ?, ?, ?)
            """,
            (capture_id, _utc_timestamp(), NORMALIZER_VERSION, len(posts)),
        )
    return {"status": "normalized", "capture_id": capture_id, "post_count": len(posts)}
