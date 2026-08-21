from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path

from scripts.alpha_swarm.contracts import (
    canonical_json,
    manifest_sha256,
    payload_sha256,
)
from scripts.alpha_swarm.lock_manifest import build_manifest
from scripts.alpha_swarm.variant_live_pilot import (
    _run_action,
    event_schedule,
    run_worker,
)
from scripts.alpha_swarm.variant_manifest import build_variant_manifest
from scripts.alpha_swarm.variant_rules import VARIANTS


def _base_manifest() -> dict:
    return build_manifest(
        run_id="pltr-worker-base",
        sessions=[date(2026, 8, 12)],
        universe=["PLTR"],
        locked_at="2026-08-11T20:00:00+00:00",
        evaluator_source_sha256="b" * 64,
    )


def _variant_manifest(tmp_path, base) -> dict:
    return build_variant_manifest(
        run_id="pltr-worker-variants",
        locked_at=datetime(2026, 8, 11, 20, 5, tzinfo=UTC),
        base_manifest=base,
        base_manifest_path=tmp_path / "base_manifest.json",
        base_input_root=tmp_path / "base",
    )


def _snapshot(base) -> dict:
    slot = base["slots"][0]
    return {
        "schema": "sharpedge.alpha_swarm.point_in_time_snapshot.v1",
        "run_id": base["run_id"],
        "manifest_sha256": manifest_sha256(base),
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "captured_at": "2026-08-12T14:30:00+00:00",
        "feature_available_ts": "2026-08-12T14:28:00+00:00",
        "features": {
            "spot": 100,
            "vwap": 99.9,
            "vs_vwap_pct": 0.1,
            "momentum_15m_pct": 0.1,
            "volume_ratio": 1.3,
        },
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
        "slot_id": slot["slot_id"],
        "session_date": slot["session_date"],
        "symbol": slot["symbol"],
        "state": "eligible",
        "eligible": True,
        "snapshot_evidence": {"snapshot_sha256": payload_sha256(snapshot)},
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
    }


def test_variant_schedule_is_frozen_around_base_timestamps(tmp_path):
    base = _base_manifest()
    events = event_schedule(_variant_manifest(tmp_path, base), base)
    assert [item["action"] for item in events] == [
        "attach_evidence",
        "publish_variants",
        "capture_entry",
        "capture_exit",
        "publish_receipts",
    ]
    assert [item["scheduled_at"].isoformat() for item in events] == [
        "2026-08-12T14:31:00+00:00",
        "2026-08-12T14:45:00+00:00",
        "2026-08-12T14:51:30+00:00",
        "2026-08-12T19:46:30+00:00",
        "2026-08-12T20:15:00+00:00",
    ]


def test_worker_before_first_event_records_no_fake_events(tmp_path):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    now = datetime(2026, 8, 11, 21, 0, tzinfo=UTC)

    state = run_worker(
        variant_manifest,
        base,
        output_root=tmp_path / "output",
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )

    assert state["events"] == {}
    assert state["paper_only"] is True
    assert state["execution_permitted"] is False
    assert state["aggregate_score_computed"] is False


def test_late_worker_marks_attach_missed_and_never_backfills(tmp_path):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    now = datetime(2026, 8, 12, 14, 40, tzinfo=UTC)

    state = run_worker(
        variant_manifest,
        base,
        output_root=tmp_path / "output",
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
        once=True,
    )

    receipt = next(iter(state["events"].values()))
    assert receipt["action"] == "attach_evidence"
    assert receipt["status"] == "missed"
    assert not list((tmp_path / "output").rglob("candidate.json"))


def test_shared_attach_and_publication_leave_base_artifacts_byte_identical(tmp_path):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    slot = base["slots"][0]
    snapshot = _snapshot(base)
    steward = _steward(base, snapshot)
    base_root = Path(variant_manifest["base_manifest"]["input_root"])
    slot_root = base_root / slot["session_date"] / slot["slot_id"]
    slot_root.mkdir(parents=True)
    snapshot_path = slot_root / "research_snapshot.json"
    steward_path = slot_root / "phase2_eligibility.json"
    snapshot_path.write_text(canonical_json(snapshot) + "\n", encoding="utf-8")
    steward_path.write_text(canonical_json(steward) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (snapshot_path, steward_path)}
    output_root = tmp_path / "variants"
    events = {item["action"]: item for item in event_schedule(variant_manifest, base)}

    _run_action(
        variant_manifest,
        base,
        events["attach_evidence"],
        output_root,
        base_root,
    )
    _run_action(
        variant_manifest,
        base,
        events["publish_variants"],
        output_root,
        base_root,
    )

    candidates = list(output_root.rglob("candidate.json"))
    assert len(candidates) == len(VARIANTS)
    assert {json.loads(path.read_text())["variant_id"] for path in candidates} == {
        item["variant_id"] for item in VARIANTS
    }
    assert all(path.read_bytes() == before[path] for path in before)


def test_restart_reuses_identical_append_only_publications(tmp_path):
    base = _base_manifest()
    variant_manifest = _variant_manifest(tmp_path, base)
    slot = base["slots"][0]
    snapshot = _snapshot(base)
    steward = _steward(base, snapshot)
    base_root = Path(variant_manifest["base_manifest"]["input_root"])
    slot_root = base_root / slot["session_date"] / slot["slot_id"]
    slot_root.mkdir(parents=True)
    (slot_root / "research_snapshot.json").write_text(
        canonical_json(snapshot) + "\n", encoding="utf-8"
    )
    (slot_root / "phase2_eligibility.json").write_text(
        canonical_json(steward) + "\n", encoding="utf-8"
    )
    output_root = tmp_path / "variants"
    events = {item["action"]: item for item in event_schedule(variant_manifest, base)}
    for _ in range(2):
        _run_action(
            variant_manifest,
            base,
            events["attach_evidence"],
            output_root,
            base_root,
        )
        _run_action(
            variant_manifest,
            base,
            events["publish_variants"],
            output_root,
            base_root,
        )
    assert len(list(output_root.rglob("candidate.json"))) == len(VARIANTS)


def test_variant_modules_are_bounded_and_do_not_import_aggregate_scoring():
    package = Path("scripts/alpha_swarm")
    paths = [
        package / "variant_rules.py",
        package / "variant_manifest.py",
        package / "variant_equity.py",
        package / "variant_live_pilot.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) < 600, path
        assert "score_receipts" not in source
        assert "robinhood" not in source.lower()
        assert "broker_client" not in source.lower()
