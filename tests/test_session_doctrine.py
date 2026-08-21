from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from session_doctrine import (
    classify_session_window,
    clock_label,
    minutes_since_open,
    opening_auction_decay_profile,
    session_datetime_from_minute,
)


def test_minutes_since_open_and_session_datetime_round_trip():
    dt = session_datetime_from_minute(75, date(2026, 1, 1))

    assert clock_label(dt) == "10:45"
    assert minutes_since_open(dt) == 75.0


def test_session_timing_preserves_timezone_awareness():
    current_time = datetime(2026, 1, 1, 10, 45, tzinfo=UTC)

    assert minutes_since_open(current_time) == 75.0
    assert session_datetime_from_minute(75, current_time).tzinfo is UTC


def test_classify_session_window_uses_shared_boundaries():
    assert classify_session_window(5, clock="9:35")["state"] == "opening"
    assert classify_session_window(45, clock="10:15")["state"] == "morning"
    assert classify_session_window(180, clock="12:30")["state"] == "midday"
    assert classify_session_window(270, clock="14:00")["state"] == "afternoon"
    assert classify_session_window(340, clock="15:10")["state"] == "power_hour"
    outside = classify_session_window(410, clock="16:20")
    assert outside["state"] == "closed_or_unknown"
    assert outside["reason"] == "outside_regular_hours"


def test_opening_auction_decay_profile_uses_shared_phase_weights():
    assert opening_auction_decay_profile(45) == {"weight": 1.0, "label": "opening"}
    assert opening_auction_decay_profile(90) == {"weight": 0.5, "label": "midday"}
    assert opening_auction_decay_profile(330) == {
        "weight": 0.2,
        "label": "late session",
    }
