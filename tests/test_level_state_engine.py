from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from level_state_engine import build_level_state_map
from level_state_view import render_level_state_block


def test_support_level_can_be_classified_as_failed_break_reclaimed():
    bars = [
        (0, 100.30, 100.42, 100.18, 100.34, 900),
        (1, 100.34, 100.36, 99.72, 99.84, 1200),
        (2, 99.84, 100.08, 99.80, 100.02, 1300),
        (3, 100.02, 100.18, 99.98, 100.12, 1100),
    ]
    states = build_level_state_map(bars, {"ORL": 100.0})

    orl = states["ORL"]
    assert orl["role"] == "support"
    assert orl["event_state"] == "failed_break_reclaimed"
    assert orl["failed_break_candidate"] == "FAILED BREAKDOWN"
    assert orl["close_relation"] == "above"


def test_reference_level_can_be_classified_as_accepted_above_reference():
    bars = [
        (0, 99.90, 100.05, 99.88, 99.98, 800),
        (1, 99.98, 100.18, 99.95, 100.10, 850),
        (2, 100.10, 100.22, 100.02, 100.16, 900),
        (3, 100.16, 100.26, 100.10, 100.20, 950),
    ]
    states = build_level_state_map(bars, {"PDC": 100.0})

    pdc = states["PDC"]
    assert pdc["role"] == "reference"
    assert pdc["event_state"] == "accepted_above_reference"
    assert pdc["acceptance"]["state"] == "accepted_above"
    assert pdc["failed_break_candidate"] is None


def test_level_state_view_renders_engine_block_and_failed_break_candidate():
    html = render_level_state_block(
        {
            "ORL": {
                "level_name": "ORL",
                "role": "support",
                "close_relation": "above",
                "event_state": "failed_break_reclaimed",
                "summary": "ORL $100.00 broke down and was reclaimed.",
                "failed_break_candidate": "FAILED BREAKDOWN",
                "acceptance": {"state": "accepted_above"},
            }
        }
    )

    assert "LEVEL STATE ENGINE" in html
    assert "FAILED BREAKDOWN" in html
    assert "ORL SUPPORT" in html


def test_resistance_level_can_be_classified_as_accepted_above_resistance():
    bars = [
        (0, 99.82, 99.95, 99.78, 99.90, 900),
        (1, 99.90, 100.18, 99.88, 100.12, 1100),
        (2, 100.12, 100.26, 100.10, 100.20, 1200),
        (3, 100.20, 100.30, 100.16, 100.24, 1300),
    ]
    states = build_level_state_map(bars, {"ORH": 100.0})

    orh = states["ORH"]
    assert orh["role"] == "resistance"
    assert orh["event_state"] == "accepted_above_resistance"
    assert orh["acceptance"]["state"] == "accepted_above"
    assert orh["failed_break_candidate"] is None


def test_support_level_can_be_classified_as_lost_support_without_reclaim():
    bars = [
        (0, 100.14, 100.18, 100.02, 100.10, 900),
        (1, 100.10, 100.12, 99.92, 99.98, 1100),
        (2, 99.98, 100.00, 99.80, 99.86, 1200),
        (3, 99.86, 99.92, 99.74, 99.82, 1300),
    ]
    states = build_level_state_map(bars, {"ORL": 100.0})

    orl = states["ORL"]
    assert orl["event_state"] == "lost_support"
    assert orl["acceptance"]["state"] == "accepted_below"
    assert orl["failed_break_candidate"] is None
    assert orl["actionable"] is False


def test_support_level_stays_testing_when_close_is_inside_buffer():
    bars = [
        (0, 100.20, 100.24, 100.08, 100.16, 900),
        (1, 100.16, 100.18, 100.00, 100.08, 950),
        (2, 100.08, 100.12, 99.98, 100.04, 980),
    ]
    states = build_level_state_map(bars, {"ORL": 100.0})

    orl = states["ORL"]
    assert orl["close_relation"] == "at_level"
    assert orl["event_state"] == "testing_support"
    assert orl["actionable"] is True


def test_level_state_acceptance_packet_tracks_shared_acceptance_window_counts():
    bars = [
        (0, 99.90, 100.05, 99.88, 99.98, 800),
        (1, 99.98, 100.18, 99.95, 100.10, 850),
        (2, 100.10, 100.22, 100.02, 100.16, 900),
        (3, 100.16, 100.26, 100.10, 100.20, 950),
    ]
    states = build_level_state_map(bars, {"PDC": 100.0}, acceptance_window=4)

    pdc = states["PDC"]
    assert pdc["acceptance"] == {
        "window": 4,
        "above_count": 2,
        "below_count": 0,
        "state": "accepted_above",
    }
