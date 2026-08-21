from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_chart_svg import chart_svg
from make_cockpit import _markers_for_price_source
from setup_markers import update_setup_markers


def _trap_confirmation(score: int = 78, bias: str = "CALLS") -> dict:
    return {
        "setup_conviction": {
            "live_trap_corroboration": {
                "trap_score": score,
                "trap_bias": bias,
                "trap_reason": "execution scoring confirmed failed-break trap",
            }
        }
    }


def _receipt(status: str) -> dict:
    return {
        "ts": "2026-07-16T09:31:00-04:00",
        "session_date": "2026-07-16",
        "spot": 625.25,
        "setup_conviction": {
            "setup_tag": "DOWNSIDE EXHAUSTION",
            "bias": "CALLS",
            "reason": "selling pressure is tiring",
            "setup_conviction_score": 72,
            "event_lifecycle": {
                "status": status,
                "last_seen_ts": "2026-07-16T09:31:00-04:00",
            },
        },
    }


def test_downside_exhaustion_conviction_requires_explicit_candidate_or_confirmed(
    tmp_path,
):
    marker_path = tmp_path / "setup_markers_spy.json"
    blank_status_receipt = _receipt("")

    markers = update_setup_markers(
        marker_path,
        decision_receipt=blank_status_receipt,
        prior_receipts=[],
    )

    assert markers == []

    candidate_markers = update_setup_markers(
        marker_path,
        decision_receipt=_receipt("candidate"),
        prior_receipts=[],
    )

    assert len(candidate_markers) == 1
    assert candidate_markers[0]["event_type"] == "DOWNSIDE EXHAUSTION"
    assert candidate_markers[0]["status"] == "candidate"
    assert candidate_markers[0]["color"] == "#d29922"


def test_confirmed_marker_uses_receipt_session_not_event_timestamp_date(tmp_path):
    marker_path = tmp_path / "setup_markers_spy.json"
    receipt = {
        "ts": "2026-07-17T00:05:00",
        "session_date": "2026-07-16",
        "session_date_source": "price_source",
        "spot": 750.72,
        **_trap_confirmation(),
        "setup_events": [
            {
                "event_id": "failed_breakdown:orl:750.67",
                "event_type": "FAILED BREAKDOWN",
                "status": "confirmed",
                "last_confirmed_ts": "2026-07-17T00:05:00",
                "bias": "CALLS",
                "detail": "reclaimed ORL $750.67 after midnight",
                "confidence": 70,
                "level": {"name": "ORL", "price": 750.67},
            }
        ],
    }

    markers = update_setup_markers(
        marker_path,
        decision_receipt=receipt,
        prior_receipts=[],
    )

    assert len(markers) == 1
    assert markers[0]["session_date"] == "2026-07-16"
    assert markers[0]["session_date_source"] == "price_source"
    assert markers[0]["marker_id"].startswith("2026-07-16:")


def test_failed_breakdown_marker_records_observation_before_execution_confirmation(
    tmp_path,
):
    marker_path = tmp_path / "setup_markers_spy.json"
    base_receipt = {
        "ts": "2026-07-16T10:46:00-04:00",
        "session_date": "2026-07-16",
        "session_date_source": "price_source",
        "spot": 753.11,
        "setup_events": [
            {
                "event_id": "failed_breakdown:orl:753.18",
                "event_type": "FAILED BREAKDOWN",
                "status": "confirmed",
                "last_confirmed_ts": "2026-07-16T10:46:00-04:00",
                "bias": "CALLS",
                "detail": "raw setup lifecycle confirmed, but execution did not",
                "confidence": 70,
                "level": {"name": "ORL", "price": 753.18},
            }
        ],
    }

    markers = update_setup_markers(
        marker_path,
        decision_receipt={
            **base_receipt,
            **_trap_confirmation(score=35, bias="NEUTRAL"),
        },
        prior_receipts=[],
    )

    assert len(markers) == 1
    assert markers[0]["event_type"] == "FAILED BREAKDOWN"
    assert markers[0]["status"] == "observed"
    assert markers[0]["setup_confirmation_status"] == "observed"
    assert markers[0]["execution_confirmation"]["confirmed"] is False

    confirmed_markers = update_setup_markers(
        marker_path,
        decision_receipt={**base_receipt, **_trap_confirmation(score=78, bias="CALLS")},
        prior_receipts=[],
    )

    confirmed_marker = next(
        marker for marker in confirmed_markers if marker["status"] == "confirmed"
    )
    assert confirmed_marker["event_type"] == "FAILED BREAKDOWN"
    assert confirmed_marker["execution_confirmation"]["confirmed"] is True
    assert confirmed_marker["execution_confirmation"]["bias"] == "CALLS"
    assert confirmed_marker["setup_confirmation_status"] == "execution_confirmed"


def test_current_session_failed_break_marker_without_execution_confirmation_is_kept_as_observed(
    tmp_path,
):
    marker_path = tmp_path / "setup_markers_spy.json"
    marker_path.write_text(
        """
{
  "schema": "sharpedge.setup_markers.v1",
  "markers": [
    {
      "marker_id": "2026-07-16:failed_breakdown:orl:753.18:confirmed",
      "session_date": "2026-07-16",
      "session_date_source": "price_source",
      "event_type": "FAILED BREAKDOWN",
      "status": "confirmed",
      "ts": "2026-07-16T10:46:00-04:00",
      "price": 753.18
    }
  ]
}
""",
        encoding="utf-8",
    )
    receipt = {
        "ts": "2026-07-16T10:47:00-04:00",
        "session_date": "2026-07-16",
        "session_date_source": "price_source",
        "spot": 753.11,
        "setup_events": [],
    }

    markers = update_setup_markers(
        marker_path,
        decision_receipt=receipt,
        prior_receipts=[],
    )

    assert len(markers) == 1
    assert markers[0]["event_type"] == "FAILED BREAKDOWN"
    assert markers[0]["status"] == "observed"
    assert markers[0]["setup_confirmation_status"] == "observed"
    assert markers[0]["color"] == "#26a641"
    assert "failed_breakdown" in marker_path.read_text(encoding="utf-8")


def test_verified_session_run_does_not_backfill_legacy_prior_receipts(tmp_path):
    marker_path = tmp_path / "setup_markers_spy.json"
    legacy_prior_receipt = {
        "ts": "2026-07-17T00:05:00",
        "session_date": "2026-07-17",
        "spot": 750.72,
        "setup_events": [
            {
                "event_id": "failed_breakdown:orl:750.67",
                "event_type": "FAILED BREAKDOWN",
                "status": "confirmed",
                "last_confirmed_ts": "2026-07-17T00:05:00",
                "bias": "CALLS",
                "detail": "legacy calendar-anchored marker",
                "confidence": 70,
                "level": {"name": "ORL", "price": 750.67},
            }
        ],
    }
    current_receipt = {
        "ts": "2026-07-17T09:35:00",
        "session_date": "2026-07-17",
        "session_date_source": "price_source",
        "spot": 743.80,
        "setup_events": [],
    }

    markers = update_setup_markers(
        marker_path,
        decision_receipt=current_receipt,
        prior_receipts=[legacy_prior_receipt],
    )

    assert markers == []


def test_price_source_filter_rejects_legacy_calendar_anchored_marker():
    markers = [
        {
            "marker_id": "bad-calendar-goblin",
            "session_date": "2026-07-17",
            "ts": "2026-07-17T00:05:00",
            "event_type": "FAILED BREAKDOWN",
        },
        {
            "marker_id": "verified-market-session",
            "session_date": "2026-07-17",
            "session_date_source": "price_source",
            "ts": "2026-07-17T09:35:00",
            "event_type": "FAILED BREAKDOWN",
        },
    ]

    filtered = _markers_for_price_source(
        markers,
        {"session_date": "2026-07-17", "last_bar_utc": "2026-07-17T13:35:00Z"},
    )

    assert [marker["marker_id"] for marker in filtered] == ["verified-market-session"]


def test_setup_marker_indicator_is_hidden_by_default_but_available_for_debug():
    rows = [
        (0, 625.0, 625.2, 624.8, 625.0, 1000),
        (1, 625.0, 625.35, 624.9, 625.25, 1200),
    ]
    svg = chart_svg(
        rows,
        {"vwap": 625.1, "vs_vwap": 0.15},
        level_states={
            "ORL": {"event_state": "testing_support"},
        },
        setup_markers=[
            {
                "event_type": "DOWNSIDE EXHAUSTION",
                "status": "candidate",
                "ts": "2026-07-16T09:31:00-04:00",
                "price": 625.25,
                "color": "#d29922",
                "detail": "candidate marker should stay visible",
            }
        ],
    )

    assert "LEVEL STATES" not in svg
    assert "DOWNSIDE EXHAUSTION CANDIDATE" not in svg

    debug_svg = chart_svg(
        rows,
        {"vwap": 625.1, "vs_vwap": 0.15},
        level_states={
            "ORL": {"event_state": "testing_support"},
        },
        setup_markers=[
            {
                "event_type": "DOWNSIDE EXHAUSTION",
                "status": "candidate",
                "ts": "2026-07-16T09:31:00-04:00",
                "price": 625.25,
                "color": "#d29922",
                "detail": "candidate marker should stay visible",
            }
        ],
        show_signal_overlays=True,
    )
    assert "LEVEL STATES" in debug_svg
    assert "DOWNSIDE EXHAUSTION CANDIDATE" in debug_svg
    assert debug_svg.rfind("DOWNSIDE EXHAUSTION CANDIDATE") > debug_svg.rfind(
        "LEVEL STATES"
    )
