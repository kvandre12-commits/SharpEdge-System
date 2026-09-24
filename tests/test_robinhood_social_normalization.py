from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from robinhood_social.capture_visible_observation import record_observation
from robinhood_social.normalization import classify_post_type, normalize_capture


def feed_xml(
    *,
    handle: str | None = "OptionDog",
    commentary: str = "Lesson: defined risk matters.",
    instrument: str = "SPY $600 Call 9/18",
    trade_action: str = "Buy 1 @ $2.00",
    displayed_pnl: str = "+$100.00",
) -> bytes:
    post = ""
    if handle is not None:
        post = f"""
        <node text="" content-desc="" class="android.view.View"
              clickable="true" bounds="[0,418][1080,1348]">
          <node text="" content-desc="view profile" class="android.widget.ImageView"
                clickable="true" bounds="[36,454][180,598]" />
          <node text="{handle}" content-desc="" class="android.widget.TextView"
                clickable="true" bounds="[204,430][400,538]" />
          <node text="&#183;" content-desc="" class="android.widget.TextView"
                clickable="false" bounds="[402,478][413,526]" />
          <node text="2d" content-desc="" class="android.widget.TextView"
                clickable="false" bounds="[425,478][465,526]" />
          <node text="" content-desc="view more actions" class="android.widget.ImageView"
                clickable="true" bounds="[924,430][1068,574]" />
          <node text="{commentary}" content-desc="" class="android.widget.TextView"
                clickable="false" bounds="[204,538][1032,958]" />
          <node text="{instrument}" content-desc="" class="android.widget.TextView"
                clickable="false" bounds="[252,1018][616,1078]" />
          <node text="{trade_action}" content-desc="" class="android.widget.TextView"
                clickable="false" bounds="[252,1078][505,1126]" />
          <node text="{displayed_pnl}" content-desc="" class="android.widget.TextView"
                clickable="true" bounds="[687,1006][984,1150]" />
          <node text="" content-desc="comment" class="android.widget.ImageView"
                clickable="true" bounds="[444,1180][514,1324]" />
          <node text="" content-desc="quick trade" class="android.widget.ImageView"
                clickable="true" bounds="[924,1180][1068,1324]" />
        </node>
        """
    return f"""<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
    <hierarchy rotation="0">
      <node text="For you" content-desc="" class="android.widget.TextView"
            clickable="false" bounds="[48,322][212,394]" />
      <node text="Following" content-desc="" class="android.widget.TextView"
            clickable="false" bounds="[248,322][463,394]" />
      {post}
      <node text="" content-desc="Social" class="android.view.View"
            clickable="false" bounds="[708,2163][804,2259]" />
    </hierarchy>
    """.encode()


def provenance() -> dict:
    return {
        "collector": "test",
        "collector_version": 1,
        "package_name": "com.robinhood.android",
        "activity_name": "com.robinhood.android/.ui.MainTabActivity",
        "app_version_name": "test",
        "app_version_code": "1",
        "device_model": "test",
        "device_transport_hash": "a" * 64,
        "adb_device_metadata_hash": "b" * 64,
        "social_surface": "feed",
    }


def capture(
    root: Path,
    *,
    observed_at: datetime,
    hierarchy: bytes,
    screenshot: bytes,
) -> dict:
    return record_observation(
        corpus_root=root,
        db_path=root / "robinhood_social_research.db",
        screenshot=screenshot,
        hierarchy_xml=hierarchy,
        provenance=provenance(),
        observed_at=observed_at,
    )


@pytest.mark.parametrize(
    ("commentary", "expected"),
    [
        ("I entered calls at the open.", "entry"),
        ("Still holding through tomorrow.", "update"),
        ("I closed the position for a gain.", "exit"),
        ("Lesson: defined risk matters.", "education"),
        ("Watching SPY here.", "unknown"),
        ("Lesson update: still holding.", "unknown"),
    ],
)
def test_post_types_require_unambiguous_authored_text(commentary, expected):
    result = classify_post_type(commentary)

    assert result["post_type"] == expected
    assert result["method"] == "authored_text_rule_v1"


def test_creator_post_identity_and_revision_history_are_stable(tmp_path):
    root = tmp_path / "corpus"
    db = root / "robinhood_social_research.db"
    first_time = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
    first = capture(
        root,
        observed_at=first_time,
        hierarchy=feed_xml(handle="OptionDog", displayed_pnl="+$100.00"),
        screenshot=b"first-png",
    )
    second = capture(
        root,
        observed_at=first_time + timedelta(minutes=5),
        hierarchy=feed_xml(
            handle="@optiondog",
            commentary="Still holding; defined risk matters.",
            displayed_pnl="+$175.00",
        ),
        screenshot=b"second-png",
    )

    assert normalize_capture(db, root, first["capture_id"])["post_count"] == 1
    assert normalize_capture(db, root, second["capture_id"])["post_count"] == 1

    with sqlite3.connect(db) as connection:
        creator_count = connection.execute("SELECT COUNT(*) FROM creators").fetchone()[
            0
        ]
        alias = connection.execute(
            "SELECT normalized_handle, observation_count FROM creator_aliases"
        ).fetchone()
        post = connection.execute(
            "SELECT identity_basis, observation_count, current_post_type "
            "FROM normalized_posts"
        ).fetchone()
        history = connection.execute(
            """
            SELECT content_fingerprint, display_fingerprint, post_type
            FROM post_observations ORDER BY observed_at_utc
            """
        ).fetchall()

    assert creator_count == 1
    assert alias == ("optiondog", 2)
    assert post == ("creator_trade_signature", 2, "update")
    assert len({row[0] for row in history}) == 2
    assert len({row[1] for row in history}) == 2
    assert [row[2] for row in history] == ["education", "update"]


def test_pnl_change_does_not_create_authored_content_revision(tmp_path):
    root = tmp_path / "corpus"
    db = root / "robinhood_social_research.db"
    start = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
    captures = [
        capture(
            root,
            observed_at=start + timedelta(minutes=index),
            hierarchy=feed_xml(displayed_pnl=pnl),
            screenshot=f"png-{index}".encode(),
        )
        for index, pnl in enumerate(("+$100.00", "+$150.00"))
    ]
    for item in captures:
        normalize_capture(db, root, item["capture_id"])

    with sqlite3.connect(db) as connection:
        content_revisions, display_revisions = connection.execute(
            """
            SELECT COUNT(DISTINCT content_fingerprint),
                   COUNT(DISTINCT display_fingerprint)
            FROM post_observations
            """
        ).fetchone()

    assert content_revisions == 1
    assert display_revisions == 2


def test_feed_absence_does_not_become_deletion(tmp_path):
    root = tmp_path / "corpus"
    db = root / "robinhood_social_research.db"
    start = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
    present = capture(
        root,
        observed_at=start,
        hierarchy=feed_xml(),
        screenshot=b"present",
    )
    absent = capture(
        root,
        observed_at=start + timedelta(minutes=5),
        hierarchy=feed_xml(handle=None),
        screenshot=b"absent",
    )
    normalize_capture(db, root, present["capture_id"])
    result = normalize_capture(db, root, absent["capture_id"])

    with sqlite3.connect(db) as connection:
        statuses = connection.execute(
            "SELECT DISTINCT presence_status FROM post_observations"
        ).fetchall()
        post_count = connection.execute(
            "SELECT COUNT(*) FROM normalized_posts"
        ).fetchone()[0]

    assert result["post_count"] == 0
    assert statuses == [("observed",)]
    assert post_count == 1


def test_normalization_is_idempotent_per_capture(tmp_path):
    root = tmp_path / "corpus"
    db = root / "robinhood_social_research.db"
    item = capture(
        root,
        observed_at=datetime(2026, 8, 29, 18, 0, tzinfo=UTC),
        hierarchy=feed_xml(),
        screenshot=b"png",
    )

    assert normalize_capture(db, root, item["capture_id"])["status"] == "normalized"
    assert (
        normalize_capture(db, root, item["capture_id"])["status"]
        == "already_normalized"
    )

    with sqlite3.connect(db) as connection:
        observations = connection.execute(
            "SELECT COUNT(*) FROM post_observations"
        ).fetchone()[0]
        alias_observations = connection.execute(
            "SELECT observation_count FROM creator_aliases"
        ).fetchone()[0]

    assert observations == 1
    assert alias_observations == 1
