from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from robinhood_social.capture_visible_observation import (
    _extract_activity,
    capture_from_adb,
    record_observation,
)
from robinhood_social.evidence_store import parse_visible_nodes

SOCIAL_XML = b"""<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node text="For you" content-desc="" class="android.widget.TextView"
        clickable="false" bounds="[48,322][212,394]" />
  <node text="Following" content-desc="" class="android.widget.TextView"
        clickable="false" bounds="[248,322][463,394]" />
  <node text="example_creator" content-desc="" class="android.widget.TextView"
        clickable="true" bounds="[204,430][420,538]" />
  <node text="SPY $600 Call 9/18" content-desc="" class="android.widget.TextView"
        clickable="false" bounds="[252,1018][616,1078]" />
  <node text="" content-desc="Social" class="android.view.View"
        clickable="false" bounds="[708,2163][804,2259]" />
</hierarchy>
"""


def provenance() -> dict:
    return {
        "collector": "sharpedge_robinhood_social_manual_capture",
        "collector_version": 1,
        "package_name": "com.robinhood.android",
        "activity_name": "com.robinhood.android/.ui.MainTabActivity",
        "app_version_name": "2026.34.1",
        "app_version_code": "123456",
        "device_model": "test-device",
        "device_transport_hash": "a" * 64,
        "adb_device_metadata_hash": "b" * 64,
        "social_surface": "feed",
    }


def test_parse_visible_nodes_preserves_order_and_accessibility_fields():
    nodes = parse_visible_nodes(SOCIAL_XML)

    assert [node["ordinal"] for node in nodes] == list(range(5))
    assert nodes[2]["text"] == "example_creator"
    assert nodes[2]["clickable"] is True
    assert nodes[4]["content_description"] == "Social"


def test_record_observation_writes_private_artifacts_and_provenance(tmp_path):
    corpus_root = tmp_path / "private-corpus"
    db_path = corpus_root / "robinhood_social_research.db"
    observed_at = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
    screenshot = b"\x89PNG\r\nprivate-test-image"

    result = record_observation(
        corpus_root=corpus_root,
        db_path=db_path,
        screenshot=screenshot,
        hierarchy_xml=SOCIAL_XML,
        provenance=provenance(),
        observed_at=observed_at,
    )

    assert result["status"] == "captured"
    assert result["source_tier"] == "visible_ui_observation"
    assert result["capture_mode"] == "manual_curated"
    assert result["visible_node_count"] == 5
    capture_root = Path(result["artifact_root"])
    assert (capture_root / "screenshot.png").read_bytes() == screenshot
    assert (capture_root / "hierarchy.xml").read_bytes() == SOCIAL_XML

    manifest = json.loads((capture_root / "manifest.json").read_text())
    assert manifest["provenance"]["social_surface"] == "feed"
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(screenshot).hexdigest()

    with sqlite3.connect(db_path) as connection:
        session = connection.execute(
            "SELECT source_tier, capture_mode, social_surface, manifest_json "
            "FROM capture_sessions"
        ).fetchone()
        artifact_count = connection.execute(
            "SELECT COUNT(*) FROM capture_artifacts"
        ).fetchone()[0]
        visible_text = connection.execute(
            "SELECT text FROM visible_nodes WHERE text <> '' ORDER BY ordinal"
        ).fetchall()

    assert session[:3] == ("visible_ui_observation", "manual_curated", "feed")
    assert json.loads(session[3])["capture_id"] == result["capture_id"]
    assert artifact_count == 2
    assert [row[0] for row in visible_text] == [
        "For you",
        "Following",
        "example_creator",
        "SPY $600 Call 9/18",
    ]


def test_record_observation_refuses_non_social_or_sensitive_screen(tmp_path):
    non_social = SOCIAL_XML.replace(
        b'content-desc="Social"', b'content-desc="Portfolio"'
    )
    with pytest.raises(RuntimeError, match="not the Social feed"):
        record_observation(
            corpus_root=tmp_path / "corpus",
            db_path=tmp_path / "corpus/research.db",
            screenshot=b"png",
            hierarchy_xml=non_social,
            provenance=provenance(),
            observed_at=datetime(2026, 8, 29, 18, 0, tzinfo=UTC),
        )

    sensitive = SOCIAL_XML.replace(b"Following", b"Buying power")
    with pytest.raises(RuntimeError, match="account-sensitive marker"):
        record_observation(
            corpus_root=tmp_path / "corpus",
            db_path=tmp_path / "corpus/research.db",
            screenshot=b"png",
            hierarchy_xml=sensitive,
            provenance=provenance(),
            observed_at=datetime(2026, 8, 29, 18, 1, tzinfo=UTC),
        )

    assert not (tmp_path / "corpus").exists()


def test_duplicate_observation_does_not_overwrite_immutable_artifacts(tmp_path):
    kwargs = {
        "corpus_root": tmp_path / "corpus",
        "db_path": tmp_path / "corpus/research.db",
        "screenshot": b"png",
        "hierarchy_xml": SOCIAL_XML,
        "provenance": provenance(),
        "observed_at": datetime(2026, 8, 29, 18, 0, tzinfo=UTC),
    }
    first = record_observation(**kwargs)

    with pytest.raises(FileExistsError, match="immutable capture already exists"):
        record_observation(**kwargs)

    assert Path(first["artifact_root"], "screenshot.png").read_bytes() == b"png"


class FakeAdb:
    def connected_device(self) -> tuple[str, str]:
        return "192.0.2.4:4321", "192.0.2.4:4321 device model:test transport_id:7"

    def shell_text(self, *args: str) -> str:
        if args[:2] == ("dumpsys", "activity"):
            return "mResumedActivity: com.robinhood.android/.ui.MainTabActivity"
        if args[:2] == ("dumpsys", "package"):
            return "versionCode=777 minSdk=28\nversionName=2026.34.1\n"
        if args[:2] == ("getprop", "ro.product.model"):
            return "Test Phone\n"
        raise AssertionError(args)

    def screenshot(self) -> bytes:
        return b"fake-png"

    def hierarchy(self) -> bytes:
        return SOCIAL_XML


def test_foreground_detection_rejects_stale_background_robinhood_task():
    stale_task_dump = (
        "mResumedActivity: com.sec.android.app.launcher/.Launcher\n"
        "ActivityRecord{abc u0 com.robinhood.android/.ui.MainTabActivity t42}\n"
    )

    with pytest.raises(RuntimeError, match="not the foreground"):
        _extract_activity(stale_task_dump)


def test_capture_from_adb_hashes_transport_identifier(tmp_path):
    result = capture_from_adb(
        corpus_root=tmp_path / "corpus",
        adb=FakeAdb(),
        observed_at=datetime(2026, 8, 29, 18, 0, tzinfo=UTC),
    )
    manifest = json.loads(Path(result["artifact_root"], "manifest.json").read_text())
    stored_hash = manifest["provenance"]["device_transport_hash"]

    assert stored_hash == hashlib.sha256(b"192.0.2.4:4321").hexdigest()
    assert "192.0.2.4" not in json.dumps(manifest)
