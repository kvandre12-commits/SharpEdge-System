from __future__ import annotations

from datetime import date

import pytest

from scripts.alpha_swarm.contracts import (
    canonical_json,
    manifest_sha256,
    parse_timestamp,
)
from scripts.alpha_swarm.data_steward import SNAPSHOT_SCHEMA, build_eligibility
from scripts.alpha_swarm.hypothesis_researcher import build_publication
from scripts.alpha_swarm.lock_manifest import build_manifest


def _manifest():
    return build_manifest(
        run_id="researcher-test",
        sessions=[date(2026, 8, 10)],
        universe=["SPY"],
        locked_at="2026-08-09T20:00:00+00:00",
        evaluator_source_sha256="a" * 64,
    )


def _snapshot(manifest, *, vs_vwap=0.1, momentum=0.1, volume=1.5):
    slot = manifest["slots"][0]
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
            "vs_vwap_pct": vs_vwap,
            "momentum_15m_pct": momentum,
            "volume_ratio": volume,
        },
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
            "spot": 100.1,
        },
        "source_refs": ["snapshot://SPY/2026-08-10T10:30"],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def _eligible(manifest, snapshot):
    slot = manifest["slots"][0]
    return build_eligibility(
        manifest,
        now=parse_timestamp(slot["eligibility_declared_at"], "now"),
        slot_id=slot["slot_id"],
        snapshot=snapshot,
    )


def _publish(manifest, snapshot, steward=None, now=None):
    slot = manifest["slots"][0]
    return build_publication(
        steward or _eligible(manifest, snapshot),
        now=now or parse_timestamp(slot["prediction_ts"], "now"),
        manifest=manifest,
        snapshot=snapshot,
    )


def test_not_due_and_ineligible_publish_no_candidate():
    manifest = _manifest()
    slot = manifest["slots"][0]
    not_due = build_eligibility(
        manifest,
        now=parse_timestamp("2026-08-10T10:29:00-04:00", "now"),
        slot_id=slot["slot_id"],
    )
    ineligible = build_eligibility(
        manifest,
        now=parse_timestamp(slot["eligibility_declared_at"], "now"),
        slot_id=slot["slot_id"],
    )
    first = build_publication(
        not_due, now=parse_timestamp(not_due["declared_at"], "now")
    )
    second = build_publication(
        ineligible,
        now=parse_timestamp(ineligible["declared_at"], "now"),
    )
    assert first["state"] == "not_ready" and first["candidate"] is None
    assert second["state"] == "data_rejected" and second["candidate"] is None


@pytest.mark.parametrize(
    ("vs_vwap", "momentum", "volume", "expected"),
    [
        (0.05, 0.05, 1.2, "long"),
        (-0.05, -0.05, 1.2, "short"),
        (0.04, 0.2, 2.0, "stand_down"),
        (0.2, 0.2, 1.19, "stand_down"),
    ],
)
def test_fixed_rule_is_inclusive_and_deterministic(vs_vwap, momentum, volume, expected):
    manifest = _manifest()
    snapshot = _snapshot(
        manifest,
        vs_vwap=vs_vwap,
        momentum=momentum,
        volume=volume,
    )
    publication = _publish(manifest, snapshot)
    candidate = publication["candidate"]
    assert publication["state"] == "candidate_published"
    assert candidate["decision"] == expected
    assert candidate["risk_cap_dollars"] == (0.0 if expected == "stand_down" else 100.0)
    assert candidate["variant_index"] == candidate["variant_count"] == 1
    assert canonical_json(publication) == canonical_json(_publish(manifest, snapshot))


def test_candidate_has_locked_provenance_and_no_option_selection():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    publication = _publish(manifest, snapshot)
    candidate = publication["candidate"]
    assert candidate["rule_id"] == "vwap_momentum_volume_v1"
    assert candidate["rule_version"] == "1.0.0"
    assert candidate["prediction_ts"] == manifest["slots"][0]["prediction_ts"]
    assert candidate["paper_only"] is True
    assert candidate["authoritative"] is False
    assert candidate["execution_permitted"] is False
    assert len(candidate["source_refs"]) == 3
    forbidden = {"vehicle", "contract", "strike", "expiry", "quantity", "score"}
    assert not forbidden & set(candidate)
    assert publication["option_selection_allowed"] is False


def test_publication_must_occur_at_exact_prediction_time():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    with pytest.raises(ValueError, match="exact locked prediction_ts"):
        _publish(
            manifest,
            snapshot,
            now=parse_timestamp("2026-08-10T10:46:00-04:00", "now"),
        )


def test_snapshot_hash_and_identity_mismatch_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    steward = _eligible(manifest, snapshot)
    mutated = {**snapshot, "symbol": "QQQ"}
    with pytest.raises(ValueError, match="SHA256"):
        _publish(manifest, mutated, steward=steward)


def test_post_prediction_or_forbidden_features_fail_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    steward = _eligible(manifest, snapshot)
    future = {**snapshot, "feature_available_ts": "2026-08-10T10:46:00-04:00"}
    future_steward = {
        **steward,
        "snapshot_evidence": {
            **steward["snapshot_evidence"],
            "snapshot_sha256": __import__("hashlib")
            .sha256(canonical_json(future).encode("utf-8"))
            .hexdigest(),
            "feature_available_ts": future["feature_available_ts"],
        },
    }
    with pytest.raises(ValueError, match="after prediction"):
        _publish(manifest, future, steward=future_steward)

    forbidden = _snapshot(manifest)
    forbidden["features"] = {**forbidden["features"], "ret_1d": 0.5}
    forbidden_steward = _eligible(manifest, forbidden)
    assert forbidden_steward["state"] == "ineligible"
    publication = build_publication(
        forbidden_steward,
        now=parse_timestamp(forbidden_steward["declared_at"], "now"),
    )
    assert publication["state"] == "data_rejected"
    assert publication["candidate"] is None


def test_missing_required_rule_feature_fails_closed():
    manifest = _manifest()
    snapshot = _snapshot(manifest)
    snapshot["features"].pop("volume_ratio")
    steward = _eligible(manifest, snapshot)
    with pytest.raises(ValueError, match="missing required"):
        _publish(manifest, snapshot, steward=steward)
