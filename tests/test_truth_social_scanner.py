from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from event_calendar import build_event_radar_live  # noqa: E402
from live_read_view import render_event_radar_block  # noqa: E402
from truth_social_scanner import build_truth_social_event_scan  # noqa: E402


def test_truth_social_scanner_classifies_manual_market_event(tmp_path):
    manual = tmp_path / "manual.json"
    cache = tmp_path / "scan.json"
    text = tmp_path / "scan.txt"
    manual.write_text(
        """
        {
          "statuses": [
            {
              "id": "1",
              "created_at": "2026-07-28T14:00:00+00:00",
              "url": "https://truthsocial.com/@realDonaldTrump/posts/1",
              "content": "<p>Powell and the Fed should cut rates. Tariffs on China are coming.</p>"
            },
            {
              "id": "2",
              "created_at": "2026-07-28T13:00:00+00:00",
              "content": "<p>Have a nice day!</p>"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    scan = build_truth_social_event_scan(
        now=dt.datetime(2026, 7, 28, 14, 30, tzinfo=dt.timezone.utc),
        cache_path=cache,
        manual_path=manual,
        text_path=text,
        enable_network=False,
    )

    latest = scan["latest_relevant"]
    assert scan["available"] is True
    assert scan["source_status"]["status"] == "manual_override"
    assert latest["impact"] == "high"
    assert {"fed", "powell", "rates", "tariffs", "china"} <= set(
        latest["matched_terms"]
    )
    assert "<p>" not in latest["text"]
    assert cache.exists()
    assert text.exists()


def test_truth_social_scanner_does_not_match_keyword_substrings(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(
        """
        [
          {
            "id": "1",
            "created_at": "2026-07-28T14:00:00+00:00",
            "content": "Software claim praises a media primary story."
          }
        ]
        """.strip(),
        encoding="utf-8",
    )

    scan = build_truth_social_event_scan(
        now=dt.datetime(2026, 7, 28, 14, 30, tzinfo=dt.timezone.utc),
        cache_path=tmp_path / "scan.json",
        manual_path=manual,
        text_path=tmp_path / "scan.txt",
        enable_network=False,
    )

    assert scan["latest_any"]["matched_terms"] == []
    assert scan["latest_relevant"] is None


def test_truth_social_scanner_reports_network_disabled_without_manual(tmp_path):
    scan = build_truth_social_event_scan(
        now=dt.datetime(2026, 7, 28, 14, 30, tzinfo=dt.timezone.utc),
        cache_path=tmp_path / "scan.json",
        manual_path=tmp_path / "missing.json",
        text_path=tmp_path / "scan.txt",
        enable_network=False,
    )

    assert scan["available"] is False
    assert scan["source_status"]["status"] == "network_disabled"
    assert scan["latest_relevant"] is None


def test_event_radar_can_embed_social_catalyst(monkeypatch):
    def fake_scan():
        return {
            "schema": "sharpedge.truth_social_event_scan.v1",
            "available": True,
            "headline": "Trump Truth event watch: HIGH fed_rates",
            "story": "Fed/rates social catalyst.",
            "source_status": {"ok": True, "status": "manual_override"},
            "latest_relevant": {
                "text": "Powell and the Fed should cut rates.",
                "impact": "high",
                "matched_terms": ["powell", "fed", "rates"],
                "created_at": "2026-07-28T14:00:00+00:00",
            },
        }

    import truth_social_scanner

    monkeypatch.setattr(
        truth_social_scanner, "build_truth_social_event_scan", fake_scan
    )

    radar = build_event_radar_live(
        today=dt.date(2026, 7, 28),
        lookahead_days=1,
        include_social=True,
    )
    html = render_event_radar_block(radar)

    assert radar["risk_window"] is True
    assert radar["social_catalyst"]["latest_relevant"]["impact"] == "high"
    assert "TRUTH SOCIAL EVENT WATCH" in html
    assert "Powell and the Fed should cut rates" in html
