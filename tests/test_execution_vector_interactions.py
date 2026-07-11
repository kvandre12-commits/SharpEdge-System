from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from execution_vector_interactions import (  # noqa: E402
    STRONGLY_BAD,
    STRONGLY_GOOD,
    WEAKLY_BAD,
    build_execution_vector_interactions,
)
from trade_permission import score_trade_permission  # noqa: E402


def _bull_bars():
    bars = []
    price = 100.0
    for minute in range(45):
        open_ = price
        close = price + 0.08
        high = close + 0.04
        low = open_ - 0.02
        volume = 1000 + minute * 12
        bars.append((minute, open_, high, low, close, volume))
        price = close
    return bars


def test_vector_interactions_surface_strong_alignment_and_conflict():
    scores = {
        "trend_score": {"score": 82, "bias": "CALLS", "reason": "above VWAP"},
        "acceptance_score": {
            "score": 78,
            "bias": "CALLS",
            "reason": "accepted above ORH",
        },
        "volume_score": {
            "score": 85,
            "bias": "CALLS",
            "reason": "participation confirms move",
        },
        "location_score": {"score": 72, "bias": "CALLS", "reason": "good edge"},
        "rejection_score": {
            "score": 70,
            "bias": "CALLS",
            "reason": "rejected lower prices",
        },
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "exhaustion_score": {"score": 72, "bias": "PUTS", "reason": "stretched high"},
        "dealer_gamma_score": {
            "score": 74,
            "bias": "NEUTRAL",
            "reason": "negative gamma",
        },
        "time_of_day_score": {"score": 74, "bias": "NEUTRAL", "reason": "morning"},
        "compression_score": {"score": 65, "bias": "NEUTRAL", "reason": "coil"},
        "pressure_score": {"score": 64, "bias": "CALLS", "reason": "buying pressure"},
        "regime_score": {"score": 82, "bias": "CALLS", "reason": "trend day"},
    }

    packet = build_execution_vector_interactions(
        scores,
        pa={"vs_vwap": 0.21},
        gp={"regime": "negative"},
    )

    assert packet["summary"]["interaction_balance"] in {"mixed", "favorable"}
    best_ids = {item["interaction_id"] for item in packet["best"]}
    warning_ids = {item["interaction_id"] for item in packet["warnings"]}

    assert "trend_acceptance_alignment" in best_ids
    assert "trend_volume_alignment" in best_ids
    assert "negative_gamma_expansion_window" in best_ids
    assert "location_exhaustion_conflict" in warning_ids
    assert any(item["classification"] == STRONGLY_GOOD for item in packet["best"])
    assert any(item["classification"] == STRONGLY_BAD for item in packet["warnings"])


def test_trade_permission_card_exposes_vector_interactions_packet():
    bars = _bull_bars()
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
        "rng_pos": 92.0,
    }
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    card = score_trade_permission(
        bars,
        pa,
        levels,
        [],
        {"atm_iv": 0.18},
        {"regime": "negative", "pin": closes[-1] + 0.6},
        {"premium_read": "cheap"},
    )

    packet = card["execution_vector_interactions"]

    assert packet["schema"] == "sharpedge.execution_vector_interactions.v1"
    assert "summary" in packet
    assert "best" in packet
    assert "warnings" in packet


def test_vector_interactions_downgrade_low_participation_conflict_when_fuel_exists():
    scores = {
        "trend_score": {"score": 82, "bias": "CALLS", "reason": "above VWAP"},
        "acceptance_score": {
            "score": 78,
            "bias": "CALLS",
            "reason": "accepted above ORH",
        },
        "volume_score": {
            "score": 25,
            "bias": "NEUTRAL",
            "reason": "participation missing",
        },
        "location_score": {"score": 72, "bias": "CALLS", "reason": "good edge"},
        "rejection_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "exhaustion_score": {"score": 35, "bias": "NEUTRAL", "reason": "not exhausted"},
        "dealer_gamma_score": {
            "score": 72,
            "bias": "NEUTRAL",
            "reason": "negative gamma",
        },
        "time_of_day_score": {"score": 74, "bias": "NEUTRAL", "reason": "morning"},
        "compression_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "pressure_score": {"score": 64, "bias": "CALLS", "reason": "buying pressure"},
        "regime_score": {"score": 82, "bias": "CALLS", "reason": "trend day"},
    }

    packet = build_execution_vector_interactions(
        scores,
        pa={"vs_vwap": 0.18},
        gp={"regime": "negative"},
    )

    conflict = next(
        item
        for item in packet["warnings"]
        if item["interaction_id"] == "trend_volume_conflict"
    )

    assert conflict["classification"] == WEAKLY_BAD
    assert conflict["label"] == "Thin participation, but fuel exists"


def test_vector_interactions_warn_when_momentum_chorus_lacks_outside_proof():
    scores = {
        "trend_score": {"score": 68, "bias": "PUTS", "reason": "below VWAP"},
        "acceptance_score": {
            "score": 42,
            "bias": "NEUTRAL",
            "reason": "no clean level acceptance",
        },
        "volume_score": {
            "score": 35,
            "bias": "NEUTRAL",
            "reason": "thin participation",
        },
        "location_score": {"score": 48, "bias": "PUTS", "reason": "mid-range"},
        "rejection_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "trap_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "exhaustion_score": {"score": 40, "bias": "NEUTRAL", "reason": "not exhausted"},
        "dealer_gamma_score": {"score": 38, "bias": "NEUTRAL", "reason": "pinning"},
        "time_of_day_score": {"score": 42, "bias": "NEUTRAL", "reason": "midday"},
        "compression_score": {"score": 35, "bias": "NEUTRAL", "reason": "none"},
        "pressure_score": {"score": 63, "bias": "PUTS", "reason": "selling pressure"},
        "regime_score": {"score": 61, "bias": "PUTS", "reason": "drift day"},
    }

    packet = build_execution_vector_interactions(
        scores,
        pa={"vs_vwap": -0.03},
        gp={"regime": "positive"},
    )

    warning_ids = {item["interaction_id"] for item in packet["warnings"]}
    classifications = {
        item["interaction_id"]: item["classification"] for item in packet["warnings"]
    }

    assert "momentum_chorus_without_support" in warning_ids
    assert classifications["momentum_chorus_without_support"] == WEAKLY_BAD
