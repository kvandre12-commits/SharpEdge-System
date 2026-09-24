"""SQLite persistence for private Robinhood Social UI observations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

CAPTURE_SCHEMA = "sharpedge.robinhood_social_capture.v1"
SOURCE_TIER = "visible_ui_observation"


def parse_visible_nodes(hierarchy_xml: bytes | str) -> list[dict[str, Any]]:
    """Return visible text-bearing accessibility nodes in document order."""
    raw = (
        hierarchy_xml.decode("utf-8")
        if isinstance(hierarchy_xml, bytes)
        else hierarchy_xml
    )
    root = ElementTree.fromstring(raw)
    nodes: list[dict[str, Any]] = []
    for element in root.iter("node"):
        text = element.attrib.get("text", "").strip()
        description = element.attrib.get("content-desc", "").strip()
        if not text and not description:
            continue
        nodes.append(
            {
                "ordinal": len(nodes),
                "text": text,
                "content_description": description,
                "class_name": element.attrib.get("class", ""),
                "bounds": element.attrib.get("bounds", ""),
                "clickable": element.attrib.get("clickable") == "true",
            }
        )
    return nodes


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.row_factory = sqlite3.Row
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS capture_sessions (
            capture_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            source_tier TEXT NOT NULL,
            capture_mode TEXT NOT NULL,
            package_name TEXT NOT NULL,
            activity_name TEXT NOT NULL,
            app_version_name TEXT,
            app_version_code TEXT,
            device_model TEXT,
            device_transport_hash TEXT NOT NULL,
            social_surface TEXT NOT NULL,
            artifact_root TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            CHECK (source_tier = 'visible_ui_observation'),
            CHECK (capture_mode = 'manual_curated')
        );

        CREATE TABLE IF NOT EXISTS capture_artifacts (
            capture_id TEXT NOT NULL REFERENCES capture_sessions(capture_id),
            artifact_kind TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
            PRIMARY KEY (capture_id, artifact_kind)
        );

        CREATE TABLE IF NOT EXISTS visible_nodes (
            capture_id TEXT NOT NULL REFERENCES capture_sessions(capture_id),
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_description TEXT NOT NULL,
            class_name TEXT NOT NULL,
            bounds TEXT NOT NULL,
            clickable INTEGER NOT NULL CHECK (clickable IN (0, 1)),
            PRIMARY KEY (capture_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS visible_nodes_text_idx
            ON visible_nodes(text);
        """
    )


def store_capture(
    db_path: Path | str,
    manifest: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    manifest_path: str,
    manifest_sha256: str,
) -> None:
    """Atomically index one immutable capture and its raw visible nodes."""
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported capture schema: {manifest.get('schema')!r}")
    if manifest.get("source_tier") != SOURCE_TIER:
        raise ValueError(f"unsupported source tier: {manifest.get('source_tier')!r}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("capture manifest must contain artifacts")

    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    provenance = manifest["provenance"]
    with connect(db_path) as connection:
        ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO capture_sessions (
                capture_id, schema_version, observed_at_utc, source_tier,
                capture_mode, package_name, activity_name, app_version_name,
                app_version_code, device_model, device_transport_hash,
                social_surface, artifact_root, manifest_path, manifest_sha256,
                manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest["capture_id"],
                manifest["schema"],
                manifest["observed_at_utc"],
                manifest["source_tier"],
                manifest["capture_mode"],
                provenance["package_name"],
                provenance["activity_name"],
                provenance.get("app_version_name"),
                provenance.get("app_version_code"),
                provenance.get("device_model"),
                provenance["device_transport_hash"],
                provenance["social_surface"],
                manifest["artifact_root"],
                manifest_path,
                manifest_sha256,
                canonical_manifest,
            ),
        )
        connection.executemany(
            """
            INSERT INTO capture_artifacts (
                capture_id, artifact_kind, relative_path, sha256, byte_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    manifest["capture_id"],
                    artifact["kind"],
                    artifact["relative_path"],
                    artifact["sha256"],
                    artifact["byte_count"],
                )
                for artifact in artifacts
            ],
        )
        connection.executemany(
            """
            INSERT INTO visible_nodes (
                capture_id, ordinal, text, content_description,
                class_name, bounds, clickable
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    manifest["capture_id"],
                    node["ordinal"],
                    node["text"],
                    node["content_description"],
                    node["class_name"],
                    node["bounds"],
                    int(node["clickable"]),
                )
                for node in nodes
            ],
        )
