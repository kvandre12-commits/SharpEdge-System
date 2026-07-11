from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from transition_pressure import build_transition_pressure_packet  # noqa: E402


def _receipt(
    permission: int, trend: int, acceptance: int, volume: int, spot: float
) -> dict:
    return {
        "permission": permission,
        "spot": spot,
        "feature_scores": {
            "trend_score": {"score": trend},
            "acceptance_score": {"score": acceptance},
            "volume_score": {"score": volume},
        },
        "setup_events": [
            {
                "event_type": "FAILED BREAKDOWN",
                "status": "confirmed",
            }
        ],
    }


def test_transition_pressure_packet_builds_release_candidate_attention_state():
    packet = build_transition_pressure_packet(
        pa={
            "spot": 100.0,
            "mom15": 0.02,
            "rng_pos": 86,
            "balance_width_pct": 0.05,
        },
        op={"call_wall": 100.15, "put_wall": 98.8},
        gp={"regime": "positive", "pin": 100.05},
        volatility_structure={
            "compression": True,
            "narrow_channel": True,
            "coil": True,
            "channel_pct": 0.08,
            "volatility_state": "squeeze",
        },
        setups=[{"tag": "FAILED BREAKDOWN", "kind": "ok"}],
        current_receipt=_receipt(68, 62, 74, 58, 100.0),
        prior_receipts=[
            _receipt(55, 51, 60, 48, 99.98),
            _receipt(60, 56, 66, 51, 100.01),
        ],
    )

    assert packet["schema"] == "sharpedge.transition_pressure.v1"
    assert packet["transition_pressure_score"] >= 70
    assert packet["transition_state"] in {"release_candidate", "resolving"}
    assert packet["attention_state"] in {"require_trigger", "execution_takes_over"}
    assert packet["directional_bias"] == "upside_release_possible"
    assert packet["persistence"]["state"] == "building"
    assert packet["persistence"]["label"] == "building_3_bars"
    assert packet["permission_leads_price"]["active"] is True
    assert packet["permission_leads_price"]["streak_reads"] >= 2
    assert packet["deltas"]["permission_delta"]["velocity"] == 8
    assert packet["potential_energy"]["compression_score"]["score"] >= 70


def test_transition_pressure_packet_exposes_level_state_pressure_surface():
    packet = build_transition_pressure_packet(
        pa={
            "spot": 100.0,
            "mom15": 0.01,
            "rng_pos": 84,
            "balance_width_pct": 0.07,
        },
        op={"call_wall": 100.3, "put_wall": 98.8},
        gp={"regime": "positive", "pin": 100.05},
        volatility_structure={
            "compression": True,
            "narrow_channel": True,
            "coil": False,
            "channel_pct": 0.10,
            "volatility_state": "squeeze",
        },
        setups=[],
        current_receipt=_receipt(62, 58, 60, 54, 100.0),
        prior_receipts=[
            _receipt(57, 54, 56, 50, 99.98),
            _receipt(60, 56, 58, 52, 100.01),
        ],
        level_states={
            "ORH": {
                "event_state": "testing_resistance",
                "role": "resistance",
                "summary": "ORH is being tested from nearby resistance",
            },
            "PDC": {
                "event_state": "accepted_above_reference",
                "role": "reference",
                "summary": "PDC is accepted above on recent closes",
            },
        },
    )

    level_state_pressure = packet["potential_energy"]["level_state_pressure"]
    assert level_state_pressure["score"] >= 60
    assert level_state_pressure["bias"] == "upside"
    assert level_state_pressure["state"] == "testing_resistance"
    assert packet["directional_bias"] in {
        "upside_release_possible",
        "two_way_compression",
    }


def test_transition_pressure_packet_can_stay_dormant_when_inputs_are_soft():
    packet = build_transition_pressure_packet(
        pa={
            "spot": 100.0,
            "mom15": 0.12,
            "rng_pos": 48,
            "balance_width_pct": 0.22,
        },
        op={"call_wall": 103.0, "put_wall": 97.0},
        gp={"regime": "unknown", "pin": None},
        volatility_structure={
            "compression": False,
            "narrow_channel": False,
            "coil": False,
            "channel_pct": 0.55,
            "volatility_state": "normal",
        },
        setups=[],
        current_receipt=_receipt(52, 49, 50, 47, 100.0),
        prior_receipts=[
            _receipt(52, 50, 50, 48, 99.7),
            _receipt(53, 51, 51, 49, 99.6),
        ],
    )

    assert packet["transition_pressure_score"] < 52
    assert packet["transition_state"] in {"dormant", "building"}
    assert packet["attention_state"] in {"ignore", "watch"}
    assert packet["directional_bias"] in {"unclear", "two_way_compression"}
    assert packet["persistence"]["state"] in {"holding", "decaying"}
