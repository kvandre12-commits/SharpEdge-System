from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from bucket_conditioned_spine import build_bucket_conditioned_spine
from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES
from spine_realtime_auditor import (
    evaluate_snapshots,
    run_once,
    snapshot_from_signal,
    summarize_feature_edges,
)
from trade_permission_context import BULLISH, NEUTRAL, ScorePart


def _signal(ts: str, spot: float, bias: str = "CALLS") -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "trade_permission": {
            "trade_gate": "CAUTION",
            "trade_permission_score": 64,
            "bias": bias,
            "authority_self_audit": {"score_spine_role": "diagnostic_advisory"},
            "market_day": {"bucket": "unclassified_day"},
            "bucket_conditioned_spine": {
                "features": [
                    {
                        "name": "trend_score",
                        "score": 82,
                        "bias": "CALLS",
                        "weight": 0.10,
                        "reason": "trend aligned",
                    },
                    {
                        "name": "pressure_score",
                        "score": 64,
                        "bias": "CALLS",
                        "weight": 0.07,
                        "reason": "buying pressure",
                    },
                    {
                        "name": "location_score",
                        "score": 42,
                        "bias": "NEUTRAL",
                        "weight": 0.10,
                        "reason": "mid location",
                    },
                ]
            },
        },
    }


def test_snapshot_from_signal_extracts_core_feature_packet() -> None:
    snapshot = snapshot_from_signal(_signal("2026-07-21T14:00:00+00:00", 100.0))

    assert snapshot is not None
    assert snapshot["spot"] == 100.0
    assert snapshot["bias"] == "CALLS"
    assert snapshot["features"]["pressure_score"]["score"] == 64
    assert snapshot["score_spine_role"] == "diagnostic_advisory"


def test_evaluate_snapshots_scores_forward_directional_outcome() -> None:
    first = snapshot_from_signal(_signal("2026-07-21T14:00:00+00:00", 100.0))
    second = snapshot_from_signal(_signal("2026-07-21T14:05:00+00:00", 100.08))

    evaluations = evaluate_snapshots([first, second], 300, 3.0)  # type: ignore[list-item]
    summary = summarize_feature_edges(evaluations, min_samples=1)

    assert evaluations[0]["trade_success"] is True
    assert summary["trend_score"]["action"] == "increase_weight_shadow"
    assert summary["pressure_score"]["weight_delta"] == 0.01
    assert summary["location_score"]["samples"] == 0


def test_flat_forward_move_is_ignored_not_counted_as_failure() -> None:
    first = snapshot_from_signal(_signal("2026-07-21T14:00:00+00:00", 100.0))
    second = snapshot_from_signal(_signal("2026-07-21T14:05:00+00:00", 100.01))

    evaluations = evaluate_snapshots([first, second], 300, 3.0)  # type: ignore[list-item]
    summary = summarize_feature_edges(evaluations, min_samples=1)

    assert evaluations[0]["trade_success"] is None
    assert evaluations[0]["features"] == []
    assert summary["trend_score"]["samples"] == 0
    assert summary["pressure_score"]["action"] == "observe"


def test_run_once_writes_audit_and_adjustment_overlay(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first_dir = cache_dir / "20260721T140000Z" / "outputs"
    second_dir = cache_dir / "20260721T140500Z" / "outputs"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    (first_dir / "signal.json").write_text(
        json.dumps(_signal("2026-07-21T14:00:00+00:00", 100.0)),
        encoding="utf-8",
    )
    (second_dir / "signal.json").write_text(
        json.dumps(_signal("2026-07-21T14:05:00+00:00", 100.08)),
        encoding="utf-8",
    )
    live = tmp_path / "signal.json"
    live.write_text(
        json.dumps(_signal("2026-07-21T14:06:00+00:00", 100.09)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    adjustment_path = tmp_path / "adjustments.json"

    audit = run_once(
        signal_path=live,
        cache_dir=cache_dir,
        output_dir=output_dir,
        adjustment_path=adjustment_path,
        horizon_seconds=300,
        min_move_bps=3.0,
        min_samples=1,
    )

    overlay = json.loads(adjustment_path.read_text(encoding="utf-8"))
    assert audit["evaluation_count"] >= 1
    assert (output_dir / "latest.txt").exists()
    assert overlay["authority"] == "diagnostic_shadow_overlay"
    assert overlay["adjustments"]["pressure_score"]["weight_delta"] == 0.01


def test_bucket_spine_consumes_realtime_overlay_only_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    overlay_path = tmp_path / "adjustments.json"
    overlay_path.write_text(
        json.dumps(
            {
                "authority": "diagnostic_shadow_overlay",
                "enabled": True,
                "max_abs_weight_delta": 0.03,
                "adjustments": {
                    "pressure_score": {
                        "weight_delta": 0.02,
                        "reason": "test pressure edge",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parts = {
        name: ScorePart(60, BULLISH if name == "pressure_score" else NEUTRAL, name)
        for name in CORE_EXECUTION_SPINE_PART_NAMES
    }

    monkeypatch.delenv("SHARPEDGE_SPINE_REALTIME_ADJUST", raising=False)
    base = build_bucket_conditioned_spine(parts, {"bucket": "unclassified_day"})
    base_pressure = next(
        row for row in base["features"] if row["name"] == "pressure_score"
    )
    assert base_pressure["weight"] == 0.07
    assert base["realtime_adjustments"]["enabled"] is False

    monkeypatch.setenv("SHARPEDGE_SPINE_REALTIME_ADJUST", "1")
    monkeypatch.setenv("SHARPEDGE_SPINE_REALTIME_ADJUSTMENTS", str(overlay_path))
    adjusted = build_bucket_conditioned_spine(parts, {"bucket": "unclassified_day"})
    adjusted_pressure = next(
        row for row in adjusted["features"] if row["name"] == "pressure_score"
    )

    assert adjusted_pressure["weight"] == 0.09
    assert adjusted["realtime_adjustments"]["enabled"] is True
    assert adjusted["realtime_adjustments"]["applied"][0]["name"] == "pressure_score"
