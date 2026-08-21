from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from scripts.alpha_swarm.contracts import (
    CANDIDATE_SCHEMA,
    PAPER_MARK_SCHEMA,
    ContractError,
    manifest_sha256,
    validate_candidate,
    validate_manifest,
)
from scripts.alpha_swarm.evaluator import (
    evaluate_candidate,
    rejection_receipt,
    score_receipts,
    verify_evaluator_source_lock,
)
from scripts.alpha_swarm.lock_manifest import build_manifest


def _manifest(*, sessions=None, universe=None):
    return build_manifest(
        run_id="phase1-test",
        sessions=sessions or [date(2026, 8, 10), date(2026, 8, 11)],
        universe=universe or ["SPY", "QQQ"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _candidate(manifest, slot_index=0, **overrides):
    slot = manifest["slots"][slot_index]
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "symbol": slot["symbol"],
        "prediction_ts": slot["prediction_ts"],
        "published_at": slot["prediction_ts"],
        "feature_available_ts": slot["prediction_ts"],
        "decision": "long",
        "risk_cap_dollars": 20.0,
        "outcome_field": "return_prediction_to_exit",
        "feature_names": ["vs_vwap", "momentum_15m"],
        "rule_id": "reclaim-v1",
        "rule_version": "1.0.0",
        "variant_index": 1,
        "variant_count": 1,
        "source_refs": ["snapshot://test"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }
    return {**payload, **overrides}


def _equity_mark(manifest, candidate, **overrides):
    slot = next(
        slot for slot in manifest["slots"] if slot["slot_id"] == candidate["slot_id"]
    )
    payload = {
        "schema": PAPER_MARK_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "entry_ts": slot["entry_ts"],
        "exit_ts": slot["exit_ts"],
        "label_available_ts": slot["label_available_ts"],
        "published_at": slot["label_available_ts"],
        "vehicle": "equity",
        "entry_method": "next_complete_bar_plus_adverse_slippage",
        "exit_method": "first_complete_bar_at_or_after_exit_minus_adverse_slippage",
        "entry_price": 100.0,
        "exit_price": 102.0,
        "quantity": 10,
        "source_refs": ["bars://test"],
        "paper_only": True,
        "execution_permitted": False,
    }
    return {**payload, **overrides}


def test_manifest_is_paper_only_and_uses_explicit_forward_label():
    manifest = _manifest()
    validate_manifest(manifest)
    assert manifest["label_contract"]["outcome_field"] == "return_prediction_to_exit"
    assert "ret_1d" in manifest["label_contract"]["forbidden_outcome_fields"]
    assert manifest["execution_permitted"] is False


def test_manifest_hash_is_canonical():
    manifest = _manifest()
    reordered = dict(reversed(list(manifest.items())))
    assert manifest_sha256(manifest) == manifest_sha256(reordered)


def test_candidate_rejects_ambiguous_ret_1d_and_future_features():
    manifest = _manifest()
    with pytest.raises(ContractError, match="ret_1d"):
        validate_candidate(manifest, _candidate(manifest, feature_names=["ret_1d"]))
    with pytest.raises(ContractError, match="after prediction"):
        validate_candidate(
            manifest,
            _candidate(manifest, feature_available_ts="2026-08-10T15:00:00-04:00"),
        )


def test_candidate_and_mark_publication_times_fail_closed():
    manifest = _manifest()
    candidate = _candidate(manifest)
    with pytest.raises(ContractError, match="published_at"):
        validate_candidate(
            manifest,
            {**candidate, "published_at": "2026-08-10T10:46:00-04:00"},
        )
    early_mark = _equity_mark(
        manifest,
        candidate,
        published_at="2026-08-10T15:46:00-04:00",
    )
    with pytest.raises(ContractError, match="before the label"):
        evaluate_candidate(manifest, candidate, early_mark)


def test_equity_long_uses_locked_costs_and_clips_utility():
    manifest = _manifest()
    candidate = _candidate(manifest)
    receipt = evaluate_candidate(manifest, candidate, _equity_mark(manifest, candidate))
    assert receipt["net_pnl_dollars"] == 18.99
    assert receipt["costs_dollars"] == 1.01
    assert receipt["utility"] == 0.9495

    clipped = evaluate_candidate(
        manifest,
        candidate,
        _equity_mark(manifest, candidate, exit_price=110.0),
    )
    assert clipped["utility"] == 1.0


def test_equity_short_inverts_direction():
    manifest = _manifest()
    candidate = _candidate(manifest, decision="short")
    receipt = evaluate_candidate(
        manifest,
        candidate,
        _equity_mark(manifest, candidate, exit_price=98.0),
    )
    assert receipt["net_pnl_dollars"] > 0
    assert receipt["utility"] > 0


def test_debit_spread_uses_conservative_values_and_per_leg_costs():
    manifest = _manifest()
    candidate = _candidate(manifest, risk_cap_dollars=100.0)
    slot = manifest["slots"][0]
    mark = {
        "schema": PAPER_MARK_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "entry_ts": slot["entry_ts"],
        "exit_ts": slot["exit_ts"],
        "label_available_ts": slot["label_available_ts"],
        "published_at": slot["label_available_ts"],
        "vehicle": "debit_spread",
        "entry_method": "buy_ask_sell_bid",
        "exit_method": "sell_bid_buy_ask",
        "entry_debit_dollars": 100.0,
        "exit_credit_dollars": 150.0,
        "leg_count": 2,
        "source_refs": ["options://test"],
        "paper_only": True,
        "execution_permitted": False,
    }
    receipt = evaluate_candidate(manifest, candidate, mark)
    assert receipt["costs_dollars"] == 0.2
    assert receipt["net_pnl_dollars"] == 49.8
    assert receipt["utility"] == 0.498


def test_stand_down_scores_zero_without_mark():
    manifest = _manifest()
    candidate = _candidate(manifest, decision="stand_down", risk_cap_dollars=0)
    receipt = evaluate_candidate(manifest, candidate)
    assert receipt["status"] == "abstained"
    assert receipt["utility"] == 0


def test_rejected_and_missing_slots_remain_zero_in_denominator():
    manifest = _manifest()
    first = _candidate(manifest, 0, decision="stand_down", risk_cap_dollars=0)
    receipts = [
        evaluate_candidate(manifest, first),
        rejection_receipt(manifest, manifest["slots"][1]["slot_id"], "liquidity gate"),
    ]
    score = score_receipts(manifest, receipts)
    assert score["eligible_slot_count"] == 4
    assert score["evaluated_receipt_count"] == 2
    assert score["missing_slot_count"] == 2
    assert score["rejected_slot_count"] == 1
    assert score["observed_mean_utility"] == 0


def test_session_block_bootstrap_is_deterministic():
    manifest = _manifest()
    receipts = []
    for index in range(len(manifest["slots"])):
        candidate = _candidate(manifest, index)
        exit_price = 102.0 if index < 2 else 98.0
        receipts.append(
            evaluate_candidate(
                manifest,
                candidate,
                _equity_mark(manifest, candidate, exit_price=exit_price),
            )
        )
    assert score_receipts(manifest, receipts) == score_receipts(manifest, receipts)


def test_predeclared_ineligible_slots_are_excluded():
    manifest = _manifest()
    unavailable = deepcopy(manifest)
    unavailable["slots"][0]["eligible"] = False
    unavailable["slots"][0]["unavailable_reason"] = (
        "source unavailable before prediction"
    )
    validate_manifest(unavailable)
    score = score_receipts(unavailable, [])
    assert score["eligible_slot_count"] == 3
    assert score["missing_slot_count"] == 3


def test_source_lock_detects_evaluator_changes():
    manifest = _manifest()
    verify_evaluator_source_lock(manifest, "a" * 64)
    with pytest.raises(ContractError, match="changed"):
        verify_evaluator_source_lock(manifest, "b" * 64)
