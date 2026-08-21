from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from regime_refinement import annotate_market_behavior


def _permission(**score_overrides):
    scores = {
        "acceptance_score": {
            "score": 78,
            "bias": "CALLS",
            "reason": "3 closes accepted above VWAP 744.74",
        },
        "trend_score": {
            "score": 82,
            "bias": "CALLS",
            "reason": "above VWAP with positive short-term momentum",
        },
        "volume_score": {
            "score": 62,
            "bias": "NEUTRAL",
            "reason": "volume acceptable at 1.3x normal",
        },
        "regime_score": {
            "score": 82,
            "bias": "CALLS",
            "reason": "trend day regime: VWAP control + directional drift",
        },
        "pressure_score": {
            "score": 40,
            "bias": "NEUTRAL",
            "reason": "pressure mixed; closes are not one-sided",
        },
        "rejection_score": {
            "score": 35,
            "bias": "NEUTRAL",
            "reason": "no obvious rejection/trap",
        },
        "trap_score": {
            "score": 35,
            "bias": "NEUTRAL",
            "reason": "no failed-break trap detected",
        },
    }
    for name, value in score_overrides.items():
        scores[name]["score"] = value
    return {
        "trade_gate": "CAUTION",
        "trade_permission_score": 64,
        "bias": "NEUTRAL",
        "scores": scores,
    }


def _pa():
    return {
        "spot": 747.31,
        "vs_vwap": 0.34,
        "rng_pos": 98.1,
        "balance_confluence": {
            "state": "disagreement",
            "score": 28,
            "bias": "NEUTRAL",
            "reason": "balance lenses disagree: bulls=opening_balance bears=recent_balance",
        },
        "balance_disagreement": {
            "has_disagreement": True,
            "reason": "disagreement: bulls=opening_balance bears=recent_balance",
        },
    }


def test_regime_refinement_is_pure_annotation_and_preserves_og_buckets():
    permission = _permission()
    before = permission.copy()

    result = annotate_market_behavior(
        pa=_pa(),
        gp={"regime": "positive", "pin": 745.0},
        permission=permission,
        target_plan={
            "label": "Magnet $745.00",
            "distance": 2.31,
            "reachable_today": {"remaining_expected_move": 0.44},
        },
        magnitude={"exp_move_realized_usd": 0.44},
        setups=[{"tag": "STICKY DAY (calm/chop)", "bias": "FADE"}],
        edge_token_position={
            "position_state": "flat",
            "suggested_action": "stand_down",
        },
    )

    assert permission == before
    assert result["schema"] == "sharpedge.regime_refinement.v1"
    assert result["mode"] == "pure_annotation_no_permission_change"
    assert set(result["buckets"]) == {
        "core_execution_spine",
        "secondary_confirmations",
        "context_governors",
        "suspect_drift_voices",
    }


def test_sticky_trend_conflict_names_denied_magnet_fade_and_upper_rail_watch():
    result = annotate_market_behavior(
        pa=_pa(),
        gp={"regime": "positive", "pin": 745.0},
        permission=_permission(),
        target_plan={
            "label": "Magnet $745.00",
            "distance": 2.31,
            "reachable_today": {"remaining_expected_move": 0.44},
        },
        setups=[{"tag": "STICKY DAY (calm/chop)", "bias": "FADE"}],
    )

    labels = {item["label"] for item in result["annotations"]}
    assert "magnet_fade_denied_by_acceptance" in labels
    assert "sticky_upper_rail_drift" in labels
    assert "sticky_trend_conflict" in labels
    assert "upper_edge_exhaustion_watch" in labels
    assert "magnet_target_unreachable_today" in labels
    assert "balance_model_disagreement" in labels


def test_confirmed_pattern_behavior_can_be_token_eligible_but_not_authoritative():
    result = annotate_market_behavior(
        pa=_pa(),
        gp={"regime": "negative"},
        permission=_permission(trap_score=72, rejection_score=70),
        setups=[{"tag": "FAILED BREAKOUT", "bias": "PUTS"}],
        edge_token_position={
            "position_state": "flat",
            "suggested_action": "stand_down",
        },
    )

    labels = {item["label"] for item in result["annotations"]}
    assert "trap_candidate_waiting_confirmation" in labels
    assert "confirmed_rejection_response" in labels
    assert result["token_annotation"]["suggested_action"] == "stand_down"
    assert result["token_annotation"]["note"].startswith("Annotations can explain")
