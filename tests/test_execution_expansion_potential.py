from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_expansion_potential import (  # noqa: E402
    build_execution_expansion_potential,
    build_expansion_fuel_surface,
    has_expansion_fuel_without_participation,
)


def test_expansion_potential_can_split_low_confirmation_from_high_fuel():
    scores = {
        "volume_score": {
            "score": 25,
            "bias": "NEUTRAL",
            "reason": "participation missing: local 0.6x",
        },
        "dealer_gamma_score": {
            "score": 72,
            "bias": "NEUTRAL",
            "reason": "negative gamma/OI proxy may support expansion",
        },
        "pressure_score": {
            "score": 64,
            "bias": "CALLS",
            "reason": "buying pressure persists",
        },
        "acceptance_score": {
            "score": 78,
            "bias": "CALLS",
            "reason": "accepted above ORH",
        },
        "location_score": {
            "score": 72,
            "bias": "CALLS",
            "reason": "good edge",
        },
        "time_of_day_score": {
            "score": 74,
            "bias": "NEUTRAL",
            "reason": "morning continuation window",
        },
        "compression_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "rejection_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
    }

    packet = build_execution_expansion_potential(
        scores,
        pa={"vs_vwap": 0.18},
        gp={"regime": "negative"},
    )

    assert packet["schema"] == "sharpedge.execution_expansion_potential.v1"
    assert packet["summary"]["participation_confirmation"] == "low"
    assert packet["summary"]["expansion_fuel"] == "high"
    assert packet["summary"]["state"] == "low_confirmation_high_fuel"
    assert packet["surface"]["score"] >= 80
    assert packet["surface"]["bias"] == "CALLS"
    mechanism_ids = {item["mechanism_id"] for item in packet["mechanisms"]}
    assert "dealer_gamma_feedback" in mechanism_ids
    assert "thin_liquidity_vacuum_proxy" in mechanism_ids
    assert has_expansion_fuel_without_participation(
        scores,
        pa={"vs_vwap": 0.18},
        gp={"regime": "negative"},
    )


def test_expansion_potential_marks_high_participation_without_extra_fuel():
    scores = {
        "volume_score": {
            "score": 85,
            "bias": "CALLS",
            "reason": "participation confirms move",
        },
        "dealer_gamma_score": {
            "score": 38,
            "bias": "NEUTRAL",
            "reason": "positive gamma gravity",
        },
        "pressure_score": {
            "score": 40,
            "bias": "NEUTRAL",
            "reason": "mixed",
        },
        "acceptance_score": {
            "score": 42,
            "bias": "NEUTRAL",
            "reason": "no clean acceptance",
        },
        "location_score": {
            "score": 48,
            "bias": "NEUTRAL",
            "reason": "mid-range",
        },
        "time_of_day_score": {
            "score": 42,
            "bias": "NEUTRAL",
            "reason": "midday chop window",
        },
        "compression_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "rejection_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
    }

    packet = build_execution_expansion_potential(
        scores,
        pa={"vs_vwap": 0.03},
        gp={"regime": "positive"},
    )

    assert packet["summary"]["participation_confirmation"] == "high"
    assert packet["summary"]["expansion_fuel"] == "low"
    assert packet["summary"]["state"] == "high_confirmation_low_fuel"


def test_expansion_fuel_surface_is_advisory_directional_lane():
    scores = {
        "volume_score": {
            "score": 35,
            "bias": "NEUTRAL",
            "reason": "thin participation",
        },
        "dealer_gamma_score": {
            "score": 72,
            "bias": "NEUTRAL",
            "reason": "negative gamma",
        },
        "pressure_score": {"score": 64, "bias": "PUTS", "reason": "selling pressure"},
        "acceptance_score": {
            "score": 74,
            "bias": "PUTS",
            "reason": "accepted below ORL",
        },
        "location_score": {
            "score": 72,
            "bias": "PUTS",
            "reason": "clean downside edge",
        },
        "time_of_day_score": {
            "score": 74,
            "bias": "NEUTRAL",
            "reason": "morning continuation",
        },
        "compression_score": {"score": 68, "bias": "NEUTRAL", "reason": "coil"},
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "rejection_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "trend_score": {"score": 70, "bias": "PUTS", "reason": "below VWAP"},
    }

    surface = build_expansion_fuel_surface(
        scores,
        pa={"vs_vwap": -0.18},
        gp={"regime": "negative"},
    )

    assert surface["score"] >= 80
    assert surface["bias"] == "PUTS"
    assert surface["dominant_mechanism"] in {
        "dealer_gamma_feedback",
        "thin_liquidity_vacuum_proxy",
        "structural_acceptance",
        "stored_energy_release",
    }
