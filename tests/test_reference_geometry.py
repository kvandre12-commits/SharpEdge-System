from __future__ import annotations

import sys
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from reference_geometry import (
    distance_pct,
    numeric_reference_map,
    reference_row,
    relation_to_reference,
)


def test_relation_to_reference_supports_multiple_at_labels():
    assert (
        relation_to_reference(100.04, 100.0, at_label="at_reference") == "at_reference"
    )
    assert relation_to_reference(100.04, 100.0, at_label="at_level") == "at_level"
    assert relation_to_reference(100.18, 100.0, at_label="at_reference") == "above"
    assert relation_to_reference(99.82, 100.0, at_label="at_reference") == "below"


def test_reference_row_and_numeric_reference_map_stay_pure_geometry():
    references = numeric_reference_map({"PDC": 100.0, "junk": "nope", "VWAP": 99.8})
    row = reference_row(100.0, "VWAP", 99.8)

    assert references == {"PDC": 100.0, "VWAP": 99.8}
    assert row["reference_name"] == "VWAP"
    assert row["reference_price"] == 99.8
    assert row["relation"] == "above"
    assert isclose(row["distance"], 0.2, rel_tol=0.0, abs_tol=1e-12)
    assert isclose(
        row["distance_pct"], distance_pct(100.0, 99.8), rel_tol=0.0, abs_tol=1e-12
    )
    assert isclose(distance_pct(100.0, 99.8), 0.2, rel_tol=0.0, abs_tol=1e-12)
