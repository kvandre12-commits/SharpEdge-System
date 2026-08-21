from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from live_trigger_check import (
    TRIGGER_RESULT_FIELDS,
    live_trigger_check,
    nearest_edge_level,
)


def _range_bucket() -> dict:
    return {
        "bucket": "range_balance_day",
        "allowed_playbooks": [
            "failed_breakdown_reclaim",
            "failed_breakout_reversal",
            "magnet_fade",
        ],
    }


def _assert_contract(result: dict) -> None:
    for field in TRIGGER_RESULT_FIELDS:
        assert field in result
    assert isinstance(result["matched_evidence"], list)
    assert isinstance(result["missing_evidence"], list)
    assert isinstance(result["needs"], list)


def test_nearest_edge_level_reads_weekly_context_box_legend():
    edge = nearest_edge_level(
        100.0,
        {"PDH": 104.0, "PDL": 96.0},
        {
            "weekly_context": {
                "legend": [
                    {"name": "HL1", "label": "weekly higher low", "price": 100.05}
                ]
            }
        },
    )

    assert edge is not None
    assert edge["name"] == "HL1"
    assert edge["source"] == "weekly_context"
    assert edge["distance_pct"] <= 0.20


def test_magnet_fade_context_can_use_weekly_context_edge():
    trigger = live_trigger_check(
        "pin_chop_wait_for_failed_break",
        _range_bucket(),
        {
            "spot": 100.0,
            "vs_vwap": 0.03,
            "weekly_context": {
                "legend": [
                    {"name": "HL1", "label": "weekly higher low", "price": 100.05}
                ]
            },
        },
        {"regime": "positive", "pin_dist_pct": 0.04},
        {"PDH": 104.0, "PDL": 96.0},
    )

    _assert_contract(trigger)
    assert trigger["status"] == "CONTEXT_MATCH"
    assert trigger["matched_playbook"] == "magnet_fade"
    assert trigger["permission_role"] == "weighting_context"
    assert trigger["location"]["edge_name"] == "HL1"
    assert trigger["location"]["edge_source"] == "weekly_context"
    assert "near edge weekly higher low 100.05" in trigger["matched_evidence"]
    assert (
        "defined stop beyond the support/resistance level"
        in trigger["missing_evidence"]
    )


def test_magnet_fade_waits_when_vwap_pin_are_good_but_no_edge():
    trigger = live_trigger_check(
        "pin_chop_wait_for_failed_break",
        _range_bucket(),
        {"spot": 100.0, "vs_vwap": 0.03},
        {"regime": "positive", "pin_dist_pct": 0.04},
        {"PDH": 104.0, "PDL": 96.0},
    )

    _assert_contract(trigger)
    assert trigger["status"] == "WAIT"
    assert trigger["matched_playbook"] == "magnet_fade"
    assert trigger["permission_role"] == "missing_edge_context"
    assert "nearby support/resistance edge" in trigger["missing_evidence"]
    assert "support-resistance" in " ".join(trigger["needs"])


def test_all_live_trigger_branches_return_same_contract_shape():
    scenarios = [
        live_trigger_check(
            "pin_chop_wait_for_failed_break",
            _range_bucket(),
            {"spot": 100.0, "vs_vwap": 0.03},
            {"regime": "positive", "pin_dist_pct": 0.04},
            {"PDH": 104.0, "PDL": 96.0},
        ),
        live_trigger_check(
            "accepted_breakout_runner",
            _range_bucket(),
            {"spot": 100.0, "vs_vwap": 0.25},
            {"regime": "negative", "pin_dist_pct": 1.2},
            {"PDH": 99.8},
        ),
        live_trigger_check(
            "failed_breakdown_reclaim",
            {
                "bucket": "failed_breakdown_long_day",
                "allowed_playbooks": ["failed_breakdown_reclaim"],
            },
            {"spot": 100.0, "vs_vwap": 0.25},
            {"regime": "positive", "pin_dist_pct": 0.5},
            {"PDL": 99.8},
        ),
        live_trigger_check(
            "wait_for_pressure_point",
            {"bucket": "unclassified_day", "allowed_playbooks": []},
            {"spot": 100.0, "vs_vwap": 0.9},
            {"regime": "unknown"},
            {},
        ),
    ]

    for result in scenarios:
        _assert_contract(result)
