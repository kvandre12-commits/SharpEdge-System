from __future__ import annotations

import copy

from scripts.alpha_swarm.contracts import (
    canonical_json,
    manifest_sha256,
    parse_timestamp,
    payload_sha256,
    source_bundle_sha256,
)
from scripts.alpha_swarm.lock_manifest import evaluator_source_paths
from scripts.alpha_swarm.paper_mark_receipt_publisher import (
    MARK_SNAPSHOT_SCHEMA,
    build_publication,
)
from scripts.alpha_swarm.skeptic_veto_agent import build_review
from tests.test_alpha_swarm_skeptic_veto_agent import (
    _manifest as _base_manifest,
)
from tests.test_alpha_swarm_skeptic_veto_agent import (
    _now,
    _phase4,
)


def _manifest():
    manifest = _base_manifest()
    manifest["evaluator_source_sha256"] = source_bundle_sha256(evaluator_source_paths())
    return manifest


def _accepted_chain(decision="long"):
    manifest = _manifest()
    phase3, option_snapshot, phase4 = _phase4(manifest, decision)
    phase5 = build_review(
        phase3,
        phase4,
        now=_now(),
        manifest=manifest,
        option_snapshot=option_snapshot,
    )
    return manifest, phase3, option_snapshot, phase4, phase5


def _mark_snapshot(
    manifest, phase3, phase4, *, entry_long_ask=0.65, entry_short_bid=0.25
):
    candidate = phase3["candidate"]
    expression = phase4["expression"]
    slot = manifest["slots"][0]
    return {
        "schema": MARK_SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "candidate_sha256": payload_sha256(candidate),
        "expression_sha256": __import__("hashlib")
        .sha256(canonical_json(expression).encode("utf-8"))
        .hexdigest(),
        "captured_at": slot["label_available_ts"],
        "entry": {
            "observed_at": slot["entry_ts"],
            "long_contract_symbol": expression["long_leg"]["contract_symbol"],
            "short_contract_symbol": expression["short_leg"]["contract_symbol"],
            "long_ask": entry_long_ask,
            "short_bid": entry_short_bid,
            "source": {
                "provider": "test",
                "source_sha256": "e" * 64,
                "source_ref": "mark://entry",
            },
        },
        "exit": {
            "observed_at": slot["exit_ts"],
            "long_contract_symbol": expression["long_leg"]["contract_symbol"],
            "short_contract_symbol": expression["short_leg"]["contract_symbol"],
            "long_bid": 0.80,
            "short_ask": 0.20,
            "source": {
                "provider": "test",
                "source_sha256": "f" * 64,
                "source_ref": "mark://exit",
            },
        },
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _label_time(manifest):
    return parse_timestamp(manifest["slots"][0]["label_available_ts"], "label")


def test_upstream_not_ready_publishes_no_mark_or_receipt():
    manifest = _manifest()
    slot = manifest["slots"][0]
    from scripts.alpha_swarm.data_steward import build_eligibility
    from scripts.alpha_swarm.hypothesis_researcher import (
        build_publication as build_hypothesis,
    )
    from scripts.alpha_swarm.options_expression_agent import (
        build_publication as build_expression,
    )

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
    phase5 = build_review(
        phase3, phase4, now=parse_timestamp(phase4["expression_at"], "now")
    )
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=parse_timestamp(phase5["reviewed_at"], "now"),
    )
    assert publication["state"] == "upstream_not_ready"
    assert publication["paper_mark"] is None
    assert publication["evaluation_receipt"] is None


def test_accepted_expression_publishes_conservative_mark_and_receipt():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    snapshot = _mark_snapshot(manifest, phase3, phase4)
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=snapshot,
    )
    mark = publication["paper_mark"]
    receipt = publication["evaluation_receipt"]
    assert publication["state"] == "evaluation_receipt_published"
    assert mark["entry_debit_dollars"] == 40.0
    assert mark["exit_credit_dollars"] == 60.0
    assert mark["entry_method"] == "buy_ask_sell_bid"
    assert mark["exit_method"] == "sell_bid_buy_ask"
    assert receipt["status"] == "evaluated"
    assert receipt["vehicle"] == "debit_spread"
    assert receipt["net_pnl_dollars"] == 19.8
    assert receipt["utility"] == 0.198


def test_short_expression_uses_same_conservative_mark_contract():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain("short")
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=_mark_snapshot(manifest, phase3, phase4),
    )
    assert publication["evaluation_receipt"]["decision"] == "short"
    assert publication["evaluation_receipt"]["status"] == "evaluated"


def test_before_label_maturity_rejects_mark():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=parse_timestamp("2026-08-10T16:14:00-04:00", "now"),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=_mark_snapshot(manifest, phase3, phase4),
    )
    assert publication["state"] == "mark_rejected"
    assert publication["evaluation_receipt"]["status"] == "rejected"


def test_contract_identity_mismatch_becomes_zero_utility_rejection():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    snapshot = _mark_snapshot(manifest, phase3, phase4)
    snapshot["entry"]["long_contract_symbol"] = "WRONG"
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=snapshot,
    )
    assert publication["state"] == "mark_rejected"
    assert publication["evaluation_receipt"]["utility"] == 0.0


def test_entry_debit_over_candidate_risk_is_rejected():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    snapshot = _mark_snapshot(
        manifest, phase3, phase4, entry_long_ask=1.50, entry_short_bid=0.25
    )
    publication = build_publication(
        phase3,
        phase4,
        phase5,
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=snapshot,
    )
    assert publication["state"] == "mark_rejected"
    assert "risk cap" in publication["evaluation_receipt"]["reason"]


def test_stand_down_publishes_locked_abstention_receipt_without_marks():
    manifest = _manifest()
    phase3, _, phase4 = _phase4(manifest, "stand_down")
    phase5 = build_review(phase3, phase4, now=_now(), manifest=manifest)
    publication = build_publication(
        phase3, phase4, phase5, now=_label_time(manifest), manifest=manifest
    )
    assert publication["state"] == "abstention_receipt_published"
    assert publication["paper_mark"] is None
    assert publication["evaluation_receipt"]["status"] == "abstained"
    assert publication["evaluation_receipt"]["utility"] == 0.0


def test_veto_publishes_zero_utility_rejection_without_marks():
    manifest, phase3, option_snapshot, phase4, _ = _accepted_chain()
    tampered = copy.deepcopy(phase4)
    tampered["expression"]["entry_debit_dollars"] = 1.0
    phase5 = build_review(
        phase3, tampered, now=_now(), manifest=manifest, option_snapshot=option_snapshot
    )
    publication = build_publication(
        phase3,
        tampered,
        phase5,
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
    )
    assert publication["state"] == "rejection_receipt_published"
    assert publication["evaluation_receipt"]["status"] == "rejected"
    assert publication["evaluation_receipt"]["utility"] == 0.0


def test_phase5_tampering_fails_chain_validation():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    phase5["accepted_expression_sha256"] = "0" * 64
    import pytest

    with pytest.raises(ValueError, match="replay"):
        build_publication(
            phase3,
            phase4,
            phase5,
            now=_label_time(manifest),
            manifest=manifest,
            option_snapshot=option_snapshot,
            mark_snapshot=_mark_snapshot(manifest, phase3, phase4),
        )


def test_publication_is_deterministic_paper_only_and_never_computes_aggregate_score():
    manifest, phase3, option_snapshot, phase4, phase5 = _accepted_chain()
    snapshot = _mark_snapshot(manifest, phase3, phase4)
    kwargs = dict(
        now=_label_time(manifest),
        manifest=manifest,
        option_snapshot=option_snapshot,
        mark_snapshot=snapshot,
    )
    first = build_publication(phase3, phase4, phase5, **kwargs)
    second = build_publication(phase3, phase4, phase5, **kwargs)
    assert canonical_json(first) == canonical_json(second)
    assert first["paper_only"] is True
    assert first["authoritative"] is False
    assert first["execution_permitted"] is False
    assert first["broker_action_allowed"] is False
    assert first["aggregate_score_computed"] is False
