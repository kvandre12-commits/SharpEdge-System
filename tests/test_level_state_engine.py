from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from level_state_engine import build_level_state_map  # noqa: E402
from level_state_view import render_level_state_block  # noqa: E402


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
