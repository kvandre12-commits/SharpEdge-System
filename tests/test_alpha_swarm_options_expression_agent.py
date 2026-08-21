from __future__ import annotations

import hashlib
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
    build_publication,
)


def _manifest():
    return build_manifest(
        run_id="expression-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _research_snapshot(manifest, *, decision="long"):
    slot = manifest["slots"][0]
    direction = 0.1 if decision == "long" else -0.1 if decision == "short" else 0.0
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
            "vs_vwap_pct": direction,
            "momentum_15m_pct": direction,
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
        "source_refs": ["snapshot://SPY/2026-08-10T10:30"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _phase3(manifest, *, decision="long"):
    slot = manifest["slots"][0]
    snapshot = _research_snapshot(manifest, decision=decision)
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


def _contract(
    symbol, option_type, strike, bid, ask, *, expiration="2026-08-21", volume=50, oi=500
):
    return {
        "contract_symbol": symbol,
        "option_type": option_type,
        "expiration": expiration,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "quote_ts": "2026-08-10T10:45:30-04:00",
        "open_interest": oi,
        "volume": volume,
    }


def _option_snapshot(manifest):
    slot = manifest["slots"][0]
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
            "provider": "test-chain",
            "source_sha256": "d" * 64,
            "source_ref": "option-chain://SPY/2026-08-10T10:46",
            "latest_data_ts": "2026-08-10T10:45:00-04:00",
        },
        "contracts": [
            _contract("SPY-C100", "call", 100, 0.55, 0.65),
            _contract("SPY-C101", "call", 101, 0.25, 0.32),
            _contract("SPY-C102", "call", 102, 0.10, 0.16),
            _contract("SPY-P100", "put", 100, 0.60, 0.70),
            _contract("SPY-P99", "put", 99, 0.30, 0.38),
            _contract("SPY-P98", "put", 98, 0.12, 0.18),
        ],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _now():
    return parse_timestamp("2026-08-10T10:46:00-04:00", "now")


def test_upstream_not_ready_consumes_no_chain():
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
    publication = build_publication(
        phase3, now=parse_timestamp(phase3["published_at"], "now")
    )
    assert publication["state"] == "upstream_not_ready"
    assert publication["expression"] is None
    assert "option_snapshot_sha256" not in publication


@pytest.mark.parametrize(
    ("decision", "option_type", "structure", "long_strike", "short_strike"),
    [
        ("long", "call", "call_debit_spread", 100.0, 101.0),
        ("short", "put", "put_debit_spread", 100.0, 99.0),
    ],
)
def test_directional_candidate_selects_one_defined_risk_spread(
    decision, option_type, structure, long_strike, short_strike
):
    manifest = _manifest()
    phase3 = _phase3(manifest, decision=decision)
    snapshot = _option_snapshot(manifest)
    publication = build_publication(
        phase3, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    expression = publication["expression"]
    assert publication["state"] == "expression_published"
    assert expression["vehicle"] == "debit_spread"
    assert expression["option_type"] == option_type
    assert expression["structure"] == structure
    assert expression["quantity"] == 1
    assert expression["long_leg"]["strike"] == long_strike
    assert expression["short_leg"]["strike"] == short_strike
    assert expression["entry_debit_dollars"] == 40.0
    assert expression["max_loss_dollars"] <= expression["risk_cap_dollars"] == 100.0
    assert expression["variant_index"] == expression["variant_count"] == 1


def test_stand_down_abstains_without_option_snapshot():
    manifest = _manifest()
    publication = build_publication(
        _phase3(manifest, decision="stand_down"), now=_now(), manifest=manifest
    )
    assert publication["state"] == "abstained"
    assert publication["expression"] is None
    assert publication["evaluator_accounting"] == "stand_down"


def test_expression_time_is_exact_and_before_entry():
    manifest = _manifest()
    with pytest.raises(ValueError, match="plus one minute"):
        build_publication(
            _phase3(manifest),
            now=parse_timestamp("2026-08-10T10:47:00-04:00", "now"),
            manifest=manifest,
            option_snapshot=_option_snapshot(manifest),
        )


def test_snapshot_identity_and_source_hash_fail_closed():
    manifest = _manifest()
    snapshot = _option_snapshot(manifest)
    snapshot["symbol"] = "QQQ"
    with pytest.raises(ValueError, match="symbol"):
        build_publication(
            _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
        )
    snapshot = _option_snapshot(manifest)
    snapshot["source"]["source_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        build_publication(
            _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
        )


def test_stale_source_fails_closed():
    manifest = _manifest()
    snapshot = _option_snapshot(manifest)
    snapshot["source"]["latest_data_ts"] = "2026-08-10T10:00:00-04:00"
    with pytest.raises(ValueError, match="stale"):
        build_publication(
            _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
        )


def test_no_liquid_spread_is_zero_utility_rejection_without_equity_fallback():
    manifest = _manifest()
    snapshot = _option_snapshot(manifest)
    for contract in snapshot["contracts"]:
        contract["volume"] = 0
    publication = build_publication(
        _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert publication["state"] == "no_valid_expression"
    assert publication["evaluator_accounting"] == "zero_utility_rejection"
    assert publication["expression"] is None
    assert "equity" not in canonical_json(publication)


def test_quote_width_and_risk_cap_are_hard_filters():
    manifest = _manifest()
    snapshot = _option_snapshot(manifest)
    for contract in snapshot["contracts"]:
        contract["ask"] = contract["bid"] * 2
    publication = build_publication(
        _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert publication["state"] == "no_valid_expression"


def test_candidate_and_contract_tampering_fail_closed():
    manifest = _manifest()
    phase3 = _phase3(manifest)
    phase3["candidate"]["decision"] = "short"
    with pytest.raises(ValueError, match="candidate_id"):
        build_publication(
            phase3,
            now=_now(),
            manifest=manifest,
            option_snapshot=_option_snapshot(manifest),
        )
    snapshot = _option_snapshot(manifest)
    snapshot["contracts"][1]["contract_symbol"] = snapshot["contracts"][0][
        "contract_symbol"
    ]
    with pytest.raises(ValueError, match="unique"):
        build_publication(
            _phase3(manifest), now=_now(), manifest=manifest, option_snapshot=snapshot
        )


def test_expression_is_deterministic_and_contains_no_execution_or_performance_authority():
    manifest = _manifest()
    phase3 = _phase3(manifest)
    snapshot = _option_snapshot(manifest)
    first = build_publication(
        phase3, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    second = build_publication(
        phase3, now=_now(), manifest=manifest, option_snapshot=snapshot
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["paper_only"] is True
    assert first["authoritative"] is False
    assert first["execution_permitted"] is False
    assert first["broker_action_allowed"] is False
    forbidden = {"utility", "performance", "score", "broker", "route", "order_id"}
    assert not forbidden & set(first["expression"])
    expected_hash = hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
    assert first["option_snapshot_sha256"] == expected_hash
