from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from session_doctrine import (  # noqa: E402
    classify_session_window,
    clock_label,
    minutes_since_open,
    opening_auction_decay_profile,
    session_datetime_from_minute,
)


def test_minutes_since_open_and_session_datetime_round_trip():
    dt = session_datetime_from_minute(75, datetime(2026, 1, 1, 0, 0))

    assert clock_label(dt) == "10:45"
    assert minutes_since_open(dt) == 75.0


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
