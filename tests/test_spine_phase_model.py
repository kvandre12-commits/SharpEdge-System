from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_read_view import render_permission_section
from spine_phase_model import annotate_spine_score_phases
from trade_permission import score_trade_permission


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


def _pa(bars, **overrides):
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
        "rng_pos": 92.0,
    }
    pa.update(overrides)
    return pa


def test_phase_model_marks_fresh_acceptance_as_head_and_climactic_volume_as_tail():
    scores = {
        "acceptance_score": {
            "score": 78,
            "bias": "CALLS",
            "reason": "accepted above ORH",
        },
        "volume_score": {
            "score": 82,
            "bias": "CALLS",
            "reason": "volume confirms move",
        },
        "dealer_gamma_score": {
            "score": 72,
            "bias": "NEUTRAL",
            "reason": "negative gamma/OI proxy may support expansion",
        },
    }

    annotated = annotate_spine_score_phases(
        scores,
        pa={
            "spot": 100.0,
            "vs_vwap": 0.08,
            "mom15": 0.4,
            "vol_mult": 4.2,
            "rng_pos": 54.0,
        },
        gp={"regime": "negative", "pin": 101.0},
        setups=[{"tag": "FAILED BREAKDOWN"}],
    )

    assert annotated["acceptance_score"]["phase"] == "head"
    assert annotated["volume_score"]["phase"] == "tail"
    assert annotated["dealer_gamma_score"]["phase"] == "head"


def test_trade_permission_exposes_phase_metadata_without_changing_gate_math():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    assert card["trade_gate"] in {"PERMIT", "CAUTION"}
    assert card["scores"]["structure_score"]["phase"] in {
        "head",
        "body",
        "tail",
        "inactive",
    }
    assert card["scores"]["structure_score"]["phase_reason"]
    assert (
        card["spine_phase_model"]["structure_score"]["phase"]
        == card["scores"]["structure_score"]["phase"]
    )


def test_render_permission_section_shows_phase_badges_and_phase_reason():
    permission = {
        "trade_gate": "PERMIT",
        "trade_permission_score": 73,
        "execution_permission_score": 73,
        "bias": "CALLS",
        "setup_conviction": {
            "setup_gate": "NONE",
            "setup_conviction_score": 0,
            "bias": "NEUTRAL",
            "setup_tag": "NONE",
            "reason": "none",
        },
        "scores": {
            "structure_score": {
                "score": 82,
                "reason": "HH/HL structure intact",
                "phase": "head",
                "phase_reason": "clean sequence is asserting and can still expand",
            },
            "volume_score": {
                "score": 25,
                "reason": "move-volume missing",
                "phase": "tail",
                "phase_reason": "participation has fallen away from the move",
            },
        },
        "market_day": {
            "bucket": "failed_breakdown_long_day",
            "score": 82,
            "bias": "CALLS",
            "allowed_playbooks": ["failed_breakdown_reclaim"],
            "risk_posture": "defined_stop_required",
            "vwap_context": {
                "state": "above_vwap",
                "posture": "upside_acceptance",
                "vs_vwap_pct": 0.22,
            },
            "reason": "battlefield is active",
        },
        "execution_flow": {},
        "bucket_conditioned_spine": {
            "gate": "PERMIT",
            "score": 73,
            "bias": "CALLS",
            "diagnostic_posture": "calls_context_only",
            "reason": "core spine only",
            "best": [],
        },
    }

    html = render_permission_section(
        permission=permission,
        pa={"spot": 100.0, "vwap": 99.8},
        op={"put_wall": 98.5, "call_wall": 101.5},
        micro={"ch_lo": 99.2, "ch_hi": 100.8},
        magnitude={"exp_move_realized_usd": 0.9},
        gp={"pin": 100.3, "regime": "negative"},
        setups=[],
        permission_trend={},
    )

    assert "Phase" in html
    assert "HEAD" in html
    assert "TAIL" in html
    assert "clean sequence is asserting and can still expand" in html
    assert "participation has fallen away from the move" in html
