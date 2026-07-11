from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_hierarchy import (  # noqa: E402
    ADVISORY_SURFACE_PART_NAMES,
    CONTEXT_GOVERNOR_PART_NAMES,
    CORE_EXECUTION_SPINE_PART_NAMES,
    SECONDARY_CONFIRMATION_PART_NAMES,
    SUSPECT_DRIFT_VOICE_PART_NAMES,
)
from execution_vector_taxonomy import (  # noqa: E402
    ADVISORY_SURFACE,
    CONTEXT_GOVERNOR,
    CORE_STRUCTURAL,
    SUSPECT_DRIFT_VOICE,
    TACTICAL_CONFIRMATION,
    VECTOR_PART_TAXONOMY,
)
from execution_vector_weights import (  # noqa: E402
    DEFAULT_BASE_BIAS_WEIGHTS,
    DEFAULT_BASE_WEIGHTS,
)


def test_every_weighted_and_hierarchical_vector_part_has_taxonomy():
    referenced_parts = set(DEFAULT_BASE_WEIGHTS)
    referenced_parts.update(DEFAULT_BASE_BIAS_WEIGHTS)
    referenced_parts.update(CORE_EXECUTION_SPINE_PART_NAMES)
    referenced_parts.update(SECONDARY_CONFIRMATION_PART_NAMES)
    referenced_parts.update(CONTEXT_GOVERNOR_PART_NAMES)
    referenced_parts.update(SUSPECT_DRIFT_VOICE_PART_NAMES)
    referenced_parts.update(ADVISORY_SURFACE_PART_NAMES)

    assert referenced_parts == set(VECTOR_PART_TAXONOMY)


def test_vector_taxonomy_declares_role_and_correlation_family():
    for name, taxonomy in VECTOR_PART_TAXONOMY.items():
        assert taxonomy["category"] in {
            CORE_STRUCTURAL,
            TACTICAL_CONFIRMATION,
            CONTEXT_GOVERNOR,
            SUSPECT_DRIFT_VOICE,
            ADVISORY_SURFACE,
        }
        assert taxonomy["correlation_family"]
        assert isinstance(taxonomy["overlap_families"], tuple)
        assert taxonomy["note"], name


def test_momentum_family_makes_double_counting_risk_visible():
    momentum_parts = {
        name
        for name, taxonomy in VECTOR_PART_TAXONOMY.items()
        if taxonomy["correlation_family"] == "momentum"
    }

    assert {"trend_score", "pressure_score", "regime_score"} <= momentum_parts
    assert VECTOR_PART_TAXONOMY["pressure_score"]["category"] == SUSPECT_DRIFT_VOICE
    assert VECTOR_PART_TAXONOMY["regime_score"]["category"] == SUSPECT_DRIFT_VOICE
    assert "participation" in VECTOR_PART_TAXONOMY["pressure_score"]["overlap_families"]
    assert "balance" in VECTOR_PART_TAXONOMY["regime_score"]["overlap_families"]


def test_taxonomy_captures_formula_overlap_seams_not_just_primary_family():
    assert VECTOR_PART_TAXONOMY["rejection_score"]["correlation_family"] == "auction"
    assert (
        "tactical_candle" in VECTOR_PART_TAXONOMY["rejection_score"]["overlap_families"]
    )
    assert "candle_score" not in VECTOR_PART_TAXONOMY
    assert VECTOR_PART_TAXONOMY["location_score"]["correlation_family"] == "location"
    assert {"balance", "stretch"} <= set(
        VECTOR_PART_TAXONOMY["location_score"]["overlap_families"]
    )
    assert (
        "tactical_candle" in VECTOR_PART_TAXONOMY["pressure_score"]["overlap_families"]
    )
    assert VECTOR_PART_TAXONOMY["time_of_day_score"]["category"] == CONTEXT_GOVERNOR
