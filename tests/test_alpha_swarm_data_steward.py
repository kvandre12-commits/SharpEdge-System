from __future__ import annotations

from datetime import date

from scripts.alpha_swarm.contracts import manifest_sha256, parse_timestamp
from scripts.alpha_swarm.data_steward import (
    ELIGIBILITY_SCHEMA,
    SNAPSHOT_SCHEMA,
    build_eligibility,
)
from scripts.alpha_swarm.lock_manifest import build_manifest


def _manifest():
    return build_manifest(
        run_id="data-steward-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _snapshot(manifest, **overrides):
    slot = manifest["slots"][0]
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": "2026-08-10T10:29:50-04:00",
        "feature_available_ts": "2026-08-10T10:29:45-04:00",
        "features": {"spot": 100.0, "vwap": 99.8, "momentum_15m_pct": 0.1},
        "price_source": {
            "provider": "yahoo",
            "source_sha256": "b" * 64,
            "latest_data_ts": "2026-08-10T10:29:00-04:00",
            "bar_count": 12,
            "spot": 100.0,
        },
        "options_source": {
            "provider": "cboe",
            "source_sha256": "c" * 64,
            "latest_data_ts": "2026-08-10T10:15:00-04:00",
            "contract_count": 1000,
            "spot": 100.2,
        },
        "source_refs": ["snapshot://SPY/2026-08-10T10:30"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }
    return {**payload, **overrides}


def _build(manifest, now, snapshot=None):
    return build_eligibility(
        manifest,
        now=parse_timestamp(now, "now"),
        slot_id=manifest["slots"][0]["slot_id"],
        snapshot=snapshot,
    )


def test_not_due_requires_no_snapshot_and_has_no_accounting_effect():
    artifact = _build(_manifest(), "2026-08-10T10:29:00-04:00")
    assert artifact["schema"] == ELIGIBILITY_SCHEMA
    assert artifact["state"] == "not_due"
    assert artifact["eligible"] is None
    assert artifact["evaluator_accounting"] == "none"
    assert "snapshot_evidence" not in artifact


def test_missing_snapshot_after_due_requires_zero_utility_rejection():
    artifact = _build(_manifest(), "2026-08-10T10:30:00-04:00")
    assert artifact["state"] == "ineligible"
    assert artifact["evaluator_accounting"] == "zero_utility_rejection"


def test_late_declaration_fails_closed_before_snapshot_review():
    manifest = _manifest()
    artifact = _build(
        manifest,
        "2026-08-10T10:46:00-04:00",
        _snapshot(manifest),
    )
    assert artifact["state"] == "ineligible"
    assert "deadline" in artifact["reasons"][0]


def test_valid_direction_neutral_snapshot_is_eligible():
    manifest = _manifest()
    artifact = _build(
        manifest,
        "2026-08-10T10:30:00-04:00",
        _snapshot(manifest),
    )
    assert artifact["state"] == "eligible"
    assert artifact["eligible"] is True
    assert artifact["evaluator_accounting"] == "candidate_allowed"
    assert artifact["snapshot_evidence"]["feature_names"] == [
        "momentum_15m_pct",
        "spot",
        "vwap",
    ]
    assert artifact["directional_output_allowed"] is False
    assert artifact["execution_permitted"] is False


def test_stale_price_and_thin_options_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    snapshot["price_source"] = {
        **snapshot["price_source"],
        "latest_data_ts": "2026-08-10T09:00:00-04:00",
    }
    snapshot["options_source"] = {
        **snapshot["options_source"],
        "contract_count": 20,
    }
    artifact = _build(manifest, "2026-08-10T10:30:00-04:00", snapshot)
    assert artifact["state"] == "ineligible"
    assert any("stale" in reason for reason in artifact["reasons"])
    assert any("contract_count" in reason for reason in artifact["reasons"])


def test_future_features_and_wrong_session_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(
        manifest,
        feature_available_ts="2026-08-10T10:31:00-04:00",
    )
    snapshot["price_source"] = {
        **snapshot["price_source"],
        "latest_data_ts": "2026-08-09T10:29:00-04:00",
    }
    artifact = _build(manifest, "2026-08-10T10:30:00-04:00", snapshot)
    assert artifact["state"] == "ineligible"
    assert any("feature_available_ts" in reason for reason in artifact["reasons"])
    assert any("outside the locked session" in reason for reason in artifact["reasons"])


def test_directional_and_ambiguous_field_names_are_rejected():
    manifest = _manifest()
    for forbidden in ("bias", "score", "long", "short", "ret_1d"):
        snapshot = _snapshot(manifest)
        snapshot["features"] = {forbidden: 1}
        artifact = _build(manifest, "2026-08-10T10:30:00-04:00", snapshot)
        assert artifact["state"] == "ineligible"
        assert any("forbidden field" in reason for reason in artifact["reasons"])


def test_source_hash_and_symbol_mismatch_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest, symbol="QQQ")
    snapshot["options_source"] = {
        **snapshot["options_source"],
        "source_sha256": "not-a-hash",
    }
    artifact = _build(manifest, "2026-08-10T10:30:00-04:00", snapshot)
    assert artifact["state"] == "ineligible"
    assert any("symbol" in reason for reason in artifact["reasons"])
    assert any("SHA-256" in reason for reason in artifact["reasons"])


def test_spot_divergence_and_paper_flags_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest, paper_only=False, execution_permitted=True)
    snapshot["options_source"] = {**snapshot["options_source"], "spot": 105.0}
    artifact = _build(manifest, "2026-08-10T10:30:00-04:00", snapshot)
    assert artifact["state"] == "ineligible"
    assert any("paper_only" in reason for reason in artifact["reasons"])
    assert any("execution_permitted" in reason for reason in artifact["reasons"])
    assert any("divergence" in reason for reason in artifact["reasons"])


def test_output_never_contains_direction_or_alpha_scoring_fields():
    manifest = _manifest()
    artifact = _build(
        manifest,
        "2026-08-10T10:30:00-04:00",
        _snapshot(manifest),
    )
    assert not {"long", "short", "bias", "score"} & set(artifact)
    assert artifact["authoritative"] is False
    assert artifact["paper_only"] is True
