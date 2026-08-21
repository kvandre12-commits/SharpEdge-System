from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from level_interaction_facts import (
    level_interaction_facts,
    level_interaction_facts_for_levels,
)


def _failed_breakdown_bars() -> list[tuple]:
    return [
        (0, 100.02, 100.08, 99.96, 100.04, 1000),
        (1, 100.04, 100.06, 99.86, 99.94, 1100),
        (2, 99.94, 99.98, 99.74, 99.82, 1200),
        (3, 99.82, 100.08, 99.92, 100.04, 1300),
        (4, 100.04, 100.09, 99.98, 100.06, 1250),
    ]


def test_level_interaction_facts_expose_mechanical_relations_and_close_counts():
    facts = level_interaction_facts(
        _failed_breakdown_bars(),
        "ORL",
        100.0,
        recent_window=4,
        acceptance_window=3,
    )

    assert facts["schema"] == "sharpedge.level_interaction_facts.v1"
    assert facts["role"] == "support"
    assert facts["current_close_relation"] == "at_level"
    assert facts["current_high_relation"] == "at_level"
    assert facts["current_low_relation"] == "at_level"
    assert facts["acceptance_window"] == 3
    assert facts["acceptance_window_used"] == 3
    assert facts["closes_above_count"] == 0
    assert facts["closes_below_count"] == 1
    assert facts["closes_at_level_count"] == 2
    assert facts["first_close_above_index"] is None
    assert facts["first_close_below_index"] == 2
    assert facts["hold_above_count"] == 0
    assert facts["hold_below_count"] == 0
    assert facts["reclaim_above_level_index"] == 3
    assert facts["recent_breach_below"] is True


def test_level_interaction_facts_for_levels_support_reference_and_resistance_roles():
    bars = [
        (0, 99.82, 99.95, 99.78, 99.90, 900),
        (1, 99.90, 100.18, 99.88, 100.12, 1100),
        (2, 100.12, 100.26, 100.10, 100.20, 1200),
        (3, 100.20, 100.30, 100.16, 100.24, 1300),
    ]

    facts_by_level = level_interaction_facts_for_levels(
        bars,
        {"PDC": 100.0, "ORH": 100.0},
    )

    assert set(facts_by_level) == {"PDC", "ORH"}
    assert facts_by_level["PDC"]["role"] == "reference"
    assert facts_by_level["ORH"]["role"] == "resistance"
    assert facts_by_level["PDC"]["current_close_relation"] == "above"
    assert facts_by_level["ORH"]["current_close_relation"] == "above"
    assert facts_by_level["PDC"]["closes_above_count"] == 3
    assert facts_by_level["ORH"]["closes_above_count"] == 3
