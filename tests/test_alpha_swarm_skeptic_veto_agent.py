from __future__ import annotations

from datetime import date

import pytest

from scripts.alpha_swarm.contracts import (
    canonical_json,
    manifest_sha256,
    parse_timestamp,
)
from scripts.alpha_swarm.data_steward import SNAPSHOT_SCHEMA, build_eligibility
from scripts.alpha_swarm.hypothesis_researcher import (
    build_publication as build_hypothesis,
)
from scripts.alpha_swarm.lock_manifest import build_manifest
from scripts.alpha_swarm.options_expression_agent import (
    OPTION_SNAPSHOT_SCHEMA,
)
from scripts.alpha_swarm.options_expression_agent import (
    build_publication as build_expression,
)
from scripts.alpha_swarm.skeptic_veto_agent import build_review


def _manifest():
    return build_manifest(
        run_id="skeptic-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _research_snapshot(manifest, decision="long"):
    slot = manifest["slots"][0]
    value = 0.1 if decision == "long" else -0.1 if decision == "short" else 0.0
    return {
        "schema": SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": "2026-08-10T10:29:50-04:00",
        "feature_available_ts": "2026-08-10T10:29:45-04:00",
        "features": {
            "vs_vwap_pct": value,
            "momentum_15m_pct": value,
            "volume_ratio": 1.5,
        },
        "price_source": {
            "provider": "test",
            "source_sha256": "b" * 64,
            "latest_data_ts": "2026-08-10T10:29:00-04:00",
            "bar_count": 12,
            "spot": 100.0,
        },
        "options_source": {
            "provider": "test",
            "source_sha256": "c" * 64,
            "latest_data_ts": "2026-08-10T10:15:00-04:00",
            "contract_count": 200,
            "spot": 100.0,
        },
        "source_refs": ["snapshot://SPY"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _phase3(manifest, decision="long"):
    slot = manifest["slots"][0]
    snapshot = _research_snapshot(manifest, decision)
    steward = build_eligibility(
        manifest,
        now=parse_timestamp(slot["eligibility_declared_at"], "now"),
        slot_id=slot["slot_id"],
        snapshot=snapshot,
    )
    return build_hypothesis(
        steward,
        now=parse_timestamp(slot["prediction_ts"], "now"),
        manifest=manifest,
        snapshot=snapshot,
    )


def _contract(symbol, option_type, strike, bid, ask, volume=50):
    return {
        "contract_symbol": symbol,
        "option_type": option_type,
        "expiration": "2026-08-21",
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "quote_ts": "2026-08-10T10:45:30-04:00",
        "open_interest": 500,
        "volume": volume,
    }


def _option_snapshot(manifest, liquid=True):
    slot = manifest["slots"][0]
    volume = 50 if liquid else 0
    return {
        "schema": OPTION_SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": "2026-08-10T10:46:00-04:00",
        "spot": 100.0,
        "source": {
            "provider": "test",
            "source_sha256": "d" * 64,
            "source_ref": "chain://SPY",
            "latest_data_ts": "2026-08-10T10:45:00-04:00",
        },
        "contracts": [
            _contract("C100", "call", 100, 0.55, 0.65, volume),
            _contract("C101", "call", 101, 0.25, 0.32, volume),
            _contract("P100", "put", 100, 0.60, 0.70, volume),
            _contract("P99", "put", 99, 0.30, 0.38, volume),
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _now():
    return parse_timestamp("2026-08-10T10:46:00-04:00", "now")


def _phase4(manifest, decision="long", liquid=True):
    phase3 = _phase3(manifest, decision)
    snapshot = None if decision == "stand_down" else _option_snapshot(manifest, liquid)
    return (
        phase3,
        snapshot,
        build_expression(
            phase3,
            now=_now(),
            manifest=manifest,
            option_snapshot=snapshot,
        ),
    )


def test_upstream_not_ready_replays_without_manifest_or_snapshot():
    manifest = _manifest()
    slot = manifest["slots"][0]
    steward = build_eligibility(
        manifest,
        now=parse_timestamp("2026-08-10T10:29:00-04:00", "now"),
        slot_id=slot["slot_id"],
    )
    phase3 = build_hypothesis(
        steward, now=parse_timestamp(steward["declared_at"], "now")
    )
    phase4 = build_expression(
        phase3, now=parse_timestamp(phase3["published_at"], "now")
    )
    review = build_review(
        phase3,
        phase4,
        now=parse_timestamp(phase4["expression_at"], "now"),
    )
    assert review["state"] == "upstream_not_ready"
    assert review["verdict"] is None
    assert review["evaluator_accounting"] == "none"


@pytest.mark.parametrize("decision", ["long", "short"])
def test_exact_directional_replay_accepts_without_modifying_expression(decision):
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest, decision)
    review = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert review["state"] == "paper_expression_accepted"
    assert review["verdict"] == "accept"
    assert review["accepted_expression_sha256"]
    assert review["can_modify_expression"] is False
    assert "expression" not in review


def test_stand_down_remains_abstained_without_snapshot():
    manifest = _manifest()
    phase3, _, phase4 = _phase4(manifest, "stand_down")
    review = build_review(phase3, phase4, now=_now(), manifest=manifest)
    assert review["state"] == "abstained"
    assert review["verdict"] is None
    assert review["evaluator_accounting"] == "stand_down"


def test_no_valid_expression_remains_zero_utility_veto():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest, "long", liquid=False)
    review = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert review["state"] == "vetoed"
    assert review["verdict"] == "veto"
    assert review["evaluator_accounting"] == "zero_utility_rejection"


def test_tampered_expression_is_vetoed_not_repaired():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest)
    phase4["expression"]["entry_debit_dollars"] = 1.0
    review = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert review["state"] == "vetoed"
    assert any("replay" in reason for reason in review["reasons"])
    assert review["accepted_expression_sha256"] is None


def test_mutated_option_snapshot_breaks_replay_and_is_vetoed():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest)
    snapshot["contracts"][0]["ask"] = 0.64
    review = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert review["verdict"] == "veto"
    assert any("replay" in reason for reason in review["reasons"])


def test_phase3_reference_and_phase4_source_tampering_are_vetoed():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest)
    phase4["phase3_publication_sha256"] = "0" * 64
    phase4["agent_source_sha256"] = "1" * 64
    review = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert review["verdict"] == "veto"
    assert len(review["reasons"]) >= 2


def test_review_time_mismatch_is_vetoed():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest)
    review = build_review(
        phase3,
        phase4,
        now=parse_timestamp("2026-08-10T10:47:00-04:00", "now"),
        manifest=manifest,
        option_snapshot=snapshot,
    )
    assert review["verdict"] == "veto"
    assert any("exact" in reason for reason in review["reasons"])


def test_review_is_deterministic_paper_only_and_contains_no_performance_authority():
    manifest = _manifest()
    phase3, snapshot, phase4 = _phase4(manifest)
    first = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    second = build_review(
        phase3, phase4, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["paper_only"] is True
    assert first["authoritative"] is False
    assert first["execution_permitted"] is False
    assert first["broker_action_allowed"] is False
    assert not {"utility", "performance", "score", "broker", "order_id"} & set(first)
