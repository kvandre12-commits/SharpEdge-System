from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from scripts.alpha_swarm.contracts import manifest_sha256, payload_sha256
from scripts.alpha_swarm.lock_manifest import build_manifest
from scripts.alpha_swarm.variant_equity import (
    build_evaluation_publication,
    build_shared_capture,
    select_complete_bar,
)
from scripts.alpha_swarm.variant_live_pilot import build_evidence_ref
from scripts.alpha_swarm.variant_manifest import (
    VariantManifestError,
    build_variant_manifest,
    validate_variant_manifest,
    variant_manifest_sha256,
)
from scripts.alpha_swarm.variant_rules import VARIANTS, build_publication, decide


def _base_manifest() -> dict:
    return build_manifest(
        run_id="pltr-base-test",
        sessions=[date(2026, 8, 12)],
        universe=["PLTR"],
        locked_at="2026-08-11T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _variant_manifest(tmp_path, base=None) -> dict:
    base = base or _base_manifest()
    return build_variant_manifest(
        run_id="pltr-variant-test",
        locked_at=datetime(2026, 8, 11, 20, 5, tzinfo=UTC),
        base_manifest=base,
        base_manifest_path=tmp_path / "base.json",
        base_input_root=tmp_path / "base",
    )


def _snapshot(base, features=None) -> dict:
    slot = base["slots"][0]
    features = features or {
        "spot": 100.0,
        "vwap": 99.92,
        "vs_vwap_pct": 0.08,
        "momentum_15m_pct": 0.08,
        "volume_ratio": 1.25,
    }
    return {
        "schema": "sharpedge.alpha_swarm.point_in_time_snapshot.v1",
        "run_id": base["run_id"],
        "manifest_sha256": manifest_sha256(base),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": "2026-08-12T14:30:00+00:00",
        "feature_available_ts": "2026-08-12T14:28:01+00:00",
        "features": features,
        "source_refs": ["yahoo://PLTR", "cboe://PLTR"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _steward(base, snapshot) -> dict:
    slot = base["slots"][0]
    return {
        "schema": "sharpedge.alpha_swarm.data_eligibility.v1",
        "run_id": base["run_id"],
        "manifest_sha256": manifest_sha256(base),
        "evaluator_source_sha256": base["evaluator_source_sha256"],
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "declared_at": slot["eligibility_declared_at"],
        "prediction_ts": slot["prediction_ts"],
        "state": "eligible",
        "eligible": True,
        "snapshot_evidence": {
            "snapshot_sha256": payload_sha256(snapshot),
            "feature_available_ts": snapshot["feature_available_ts"],
            "feature_names": sorted(snapshot["features"]),
            "source_refs": snapshot["source_refs"],
        },
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "directional_output_allowed": False,
    }


def _evidence(tmp_path, features=None):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    snapshot = _snapshot(base, features)
    steward = _steward(base, snapshot)
    slot = base["slots"][0]
    evidence = build_evidence_ref(
        variant_manifest=variant_manifest,
        base_manifest=base,
        slot=slot,
        snapshot=snapshot,
        steward=steward,
        attached_at="2026-08-12T14:31:00+00:00",
    )
    return base, variant_manifest, snapshot, steward, slot, evidence


def _provider_capture(symbol, session_date, observed_at, bars):
    return {
        "schema": "sharpedge.alpha_swarm.provider_capture.v1",
        "provider": "yahoo_chart_1m",
        "source_ref": f"yahoo://{symbol}",
        "symbol": symbol,
        "session_date": session_date,
        "observed_at": observed_at,
        "latest_data_ts": bars[-1]["timestamp"],
        "bars": bars,
    }


def test_rule_variants_make_distinct_predeclared_decisions():
    features = {
        "vs_vwap_pct": 0.08,
        "momentum_15m_pct": 0.08,
        "volume_ratio": 1.25,
    }
    assert [decide(variant, features) for variant in VARIANTS] == [
        "long",
        "stand_down",
        "long",
        "stand_down",
    ]
    bearish = {**features, "vs_vwap_pct": -0.2, "momentum_15m_pct": -0.2}
    assert [decide(variant, bearish) for variant in VARIANTS] == [
        "short",
        "stand_down",
        "short",
        "stand_down",
    ]


def test_rule_thresholds_are_inclusive_and_forbid_outcomes():
    balanced = VARIANTS[0]
    assert (
        decide(
            balanced,
            {
                "vs_vwap_pct": 0.05,
                "momentum_15m_pct": 0.05,
                "volume_ratio": 1.2,
            },
        )
        == "long"
    )
    with pytest.raises(ValueError, match="forbidden"):
        decide(
            balanced,
            {
                "vs_vwap_pct": 0.05,
                "momentum_15m_pct": 0.05,
                "volume_ratio": 1.2,
                "score": 99,
            },
        )


def test_variant_manifest_locks_complete_family_and_rejects_late_lock(tmp_path):
    base = _base_manifest()
    manifest = _variant_manifest(tmp_path, base)
    validate_variant_manifest(manifest, base, verify_sources=True)
    assert [item["variant_index"] for item in manifest["variants"]] == [1, 2, 3, 4]
    assert manifest["governance"]["aggregate_score_hidden_during_pilot"] is True
    with pytest.raises(VariantManifestError, match="after first evidence"):
        build_variant_manifest(
            run_id="late",
            locked_at=datetime(2026, 8, 12, 14, 29, tzinfo=UTC),
            base_manifest=base,
            base_manifest_path=tmp_path / "base.json",
            base_input_root=tmp_path / "base",
        )


def test_all_variant_publications_share_one_evidence_hash(tmp_path):
    base, variant_manifest, snapshot, steward, slot, evidence = _evidence(tmp_path)
    locked_hash = variant_manifest_sha256(variant_manifest)
    publications = [
        build_publication(
            base_manifest=base,
            slot=slot,
            snapshot=snapshot,
            steward=steward,
            evidence_ref=evidence,
            variant=variant,
            variant_manifest_sha256=locked_hash,
            observed_at=slot["prediction_ts"],
        )
        for variant in VARIANTS
    ]
    assert len({item["shared_evidence_sha256"] for item in publications}) == 1
    assert [item["candidate"]["decision"] for item in publications] == [
        "long",
        "stand_down",
        "long",
        "stand_down",
    ]
    assert all(item["candidate"]["variant_count"] == 4 for item in publications)


def test_complete_bar_selection_rejects_still_open_bar():
    target = datetime(2026, 8, 12, 14, 50, tzinfo=UTC)
    bars = [
        {
            "timestamp": target.isoformat(),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100.5,
            "volume": 1000,
        }
    ]
    early = _provider_capture(
        "PLTR", "2026-08-12", (target + timedelta(seconds=30)).isoformat(), bars
    )
    with pytest.raises(ValueError, match="no complete"):
        select_complete_bar(early, target_ts=target.isoformat())
    complete = {**early, "observed_at": (target + timedelta(seconds=61)).isoformat()}
    assert select_complete_bar(complete, target_ts=target.isoformat())["high"] == 101.0


def test_directional_variants_get_adverse_equity_marks_and_control_abstains(tmp_path):
    base, variant_manifest, snapshot, steward, slot, evidence = _evidence(tmp_path)
    locked_hash = variant_manifest_sha256(variant_manifest)
    target_entry = datetime.fromisoformat(slot["entry_ts"])
    target_exit = datetime.fromisoformat(slot["exit_ts"])
    entry_provider = _provider_capture(
        "PLTR",
        slot["session_date"],
        (target_entry + timedelta(seconds=90)).isoformat(),
        [
            {
                "timestamp": target_entry.isoformat(),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 1000,
            }
        ],
    )
    exit_provider = _provider_capture(
        "PLTR",
        slot["session_date"],
        (target_exit + timedelta(seconds=90)).isoformat(),
        [
            {
                "timestamp": target_exit.isoformat(),
                "open": 103,
                "high": 104,
                "low": 102,
                "close": 103,
                "volume": 2000,
            }
        ],
    )
    entry = build_shared_capture(
        base_manifest=base,
        slot=slot,
        phase="entry",
        provider_capture=entry_provider,
        variant_manifest_sha256=locked_hash,
    )
    exit_capture = build_shared_capture(
        base_manifest=base,
        slot=slot,
        phase="exit",
        provider_capture=exit_provider,
        variant_manifest_sha256=locked_hash,
    )
    publications = [
        build_publication(
            base_manifest=base,
            slot=slot,
            snapshot=snapshot,
            steward=steward,
            evidence_ref=evidence,
            variant=variant,
            variant_manifest_sha256=locked_hash,
            observed_at=slot["prediction_ts"],
        )
        for variant in VARIANTS
    ]
    evaluations = [
        build_evaluation_publication(
            base_manifest=base,
            candidate_publication=publication,
            entry_capture=entry,
            exit_capture=exit_capture,
            published_at=slot["label_available_ts"],
        )
        for publication in publications
    ]
    assert evaluations[0]["paper_mark"]["entry_price"] == 101.0
    assert evaluations[0]["paper_mark"]["exit_price"] == 102.0
    assert evaluations[0]["evaluation_receipt"]["status"] == "evaluated"
    assert evaluations[1]["paper_mark"] is None
    assert evaluations[1]["evaluation_receipt"]["status"] == "abstained"
    assert all(item["aggregate_score_computed"] is False for item in evaluations)


def test_shared_evidence_detects_snapshot_rewrite(tmp_path):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    snapshot = _snapshot(base)
    steward = _steward(base, snapshot)
    snapshot["features"]["spot"] = 999
    with pytest.raises(ValueError, match="snapshot hash"):
        build_evidence_ref(
            variant_manifest=variant_manifest,
            base_manifest=base,
            slot=base["slots"][0],
            snapshot=snapshot,
            steward=steward,
            attached_at="2026-08-12T14:31:00+00:00",
        )
