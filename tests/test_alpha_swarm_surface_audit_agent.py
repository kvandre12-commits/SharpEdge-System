from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.alpha_swarm import surface_audit_agent as agent
from scripts.alpha_swarm import surface_audit_ledger as ledger
from scripts.alpha_swarm.surface_audit_output import render_html, render_markdown


def _comparison(now: datetime) -> dict:
    symbols = [
        {
            "symbol": symbol,
            "available": True,
            "source_path": f"/evidence/{symbol}.json",
            "source_sha256": symbol.lower() * 8,
        }
        for symbol in ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN")
    ]
    return {
        "schema": agent.COMPARISON_SCHEMA,
        "generated_at": (now - timedelta(seconds=30)).isoformat(),
        "headline": "Both stand down <safely>",
        "safety": {
            "paper_only": True,
            "authoritative": False,
            "execution_permitted": False,
            "can_mutate_paper_artifacts": False,
            "can_override_approval_decision": False,
            "aggregate_score_computed": False,
            "hindsight_use": "comparison_only",
        },
        "paper_worker": {
            "state": "running",
            "pid": 123,
            "heartbeat_at": (now - timedelta(seconds=20)).isoformat(),
        },
        "paper_surface": {
            "available_symbol_count": 6,
            "symbols": symbols,
            "current_session_event_counts": {"completed": 42, "pending": 6},
        },
        "temporal_alignment": {
            "state": "later_live_snapshot",
            "prediction_to_live_minutes": 311.0,
        },
        "spy_comparison": {
            "action_alignment": "both_no_action",
            "direction_change": "mixed_to_bearish",
            "paper_decision": "stand_down",
            "live_execution_posture": "neutral",
        },
    }


def _args(tmp_path, input_path):
    return argparse.Namespace(
        input=input_path,
        output_json=tmp_path / "audit" / "latest.json",
        output_markdown=tmp_path / "audit" / "latest.md",
        output_html=tmp_path / "audit.html",
        ledger=tmp_path / "audit" / "recommendation_ledger.jsonl",
        interval_seconds=0,
    )


def test_healthy_report_explains_without_claiming_authority():
    now = datetime(2026, 8, 11, 19, 50, tzinfo=UTC)

    report = agent.build_audit(_comparison(now), now, "abc123")

    assert report["schema"] == agent.SCHEMA
    assert report["why"]["status"] == "healthy_with_observations"
    assert {item["finding_id"] for item in report["findings"]} == {
        "PROCESS-001",
        "PROCESS-002",
    }
    assert report["how"]["automatic_changes_allowed"] is False
    assert report["safety"] == {
        "observational_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "self_modification_allowed": False,
        "automatic_parameter_tuning_allowed": False,
        "frozen_artifact_mutation_allowed": False,
        "aggregate_score_computed": False,
    }


def test_safety_violation_is_critical_and_actionable():
    now = datetime(2026, 8, 11, 19, 50, tzinfo=UTC)
    comparison = _comparison(now)
    comparison["safety"]["execution_permitted"] = True

    report = agent.build_audit(comparison, now, None)

    finding = next(
        item for item in report["findings"] if item["finding_id"] == "SAFETY-002"
    )
    assert finding["severity"] == "critical"
    assert report["why"]["status"] == "action_required"
    assert any(
        item["recommendation_key"] == "quarantine_unsafe_comparison_contract"
        for item in report["how"]["recommendations"]
    )


def test_stale_comparison_and_worker_have_deterministic_rules():
    now = datetime(2026, 8, 11, 19, 50, tzinfo=UTC)
    comparison = _comparison(now)
    comparison["generated_at"] = (now - timedelta(seconds=400)).isoformat()
    comparison["paper_worker"]["heartbeat_at"] = (
        now - timedelta(seconds=400)
    ).isoformat()

    report = agent.build_audit(comparison, now, None)

    findings = {item["finding_id"]: item for item in report["findings"]}
    assert findings["FRESH-004"]["severity"] == "high"
    assert findings["WORKER-004"]["severity"] == "high"
    assert report["why"]["status"] == "action_required"


def test_ledger_appends_only_material_changes_and_verifies_chain(tmp_path):
    now = datetime(2026, 8, 11, 19, 50, tzinfo=UTC)
    recommendations = agent.build_audit(_comparison(now), now, None)["how"][
        "recommendations"
    ]
    path = tmp_path / "ledger.jsonl"

    first = ledger.append_if_changed(path, recommendations, now)
    second = ledger.append_if_changed(path, recommendations, now + timedelta(seconds=5))
    changed = [dict(item) for item in recommendations]
    changed[0] = {**changed[0], "severity": "medium"}
    third = ledger.append_if_changed(path, changed, now + timedelta(seconds=10))

    events = ledger.read_verified_events(path)
    assert first["changed"] is True
    assert second["changed"] is False
    assert third["changed"] is True
    assert len(events) == 2
    assert events[1]["previous_event_hash"] == events[0]["event_hash"]
    assert events[1]["superseded"] == [changed[0]["recommendation_key"]]


def test_corrupt_ledger_fails_closed_but_publishes_diagnostic(tmp_path):
    now = datetime.now(UTC)
    input_path = tmp_path / "comparison.json"
    input_path.write_text(json.dumps(_comparison(now)), encoding="utf-8")
    args = _args(tmp_path, input_path)
    args.ledger.parent.mkdir(parents=True)
    args.ledger.write_text('{"schema":"wrong"}\n', encoding="utf-8")

    report = agent.run_once(args)

    assert report["recommendation_ledger"]["verified"] is False
    assert report["recommendation_ledger"]["append_suppressed"] is True
    assert any(item["finding_id"] == "LEDGER-001" for item in report["findings"])
    assert args.output_json.exists()
    assert args.output_markdown.exists()
    assert args.output_html.exists()
    assert args.ledger.read_text() == '{"schema":"wrong"}\n'


def test_run_once_handles_malformed_input_and_escapes_html(tmp_path):
    input_path = tmp_path / "comparison.json"
    input_path.write_text("[]", encoding="utf-8")
    args = _args(tmp_path, input_path)
    source_before = input_path.read_bytes()

    report = agent.run_once(args)
    markdown = render_markdown(report)
    html = render_html(markdown + "\n<script>bad()</script>")

    assert input_path.read_bytes() == source_before
    assert report["why"]["status"] == "action_required"
    assert report["findings"][0]["finding_id"] == "INPUT-001"
    assert "comparison input is not a JSON object" in report["findings"][0]["evidence"]
    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert json.loads(args.output_json.read_text())["schema"] == agent.SCHEMA


def test_ledger_rejects_tampering(tmp_path):
    now = datetime(2026, 8, 11, 19, 50, tzinfo=UTC)
    path = tmp_path / "ledger.jsonl"
    recommendations = agent.build_audit(_comparison(now), now, None)["how"][
        "recommendations"
    ]
    ledger.append_if_changed(path, recommendations, now)
    payload = json.loads(path.read_text())
    payload["opened"] = ["invented"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ledger.LedgerError, match="invalid ledger event hash"):
        ledger.read_verified_events(path)
