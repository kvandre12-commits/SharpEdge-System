#!/usr/bin/env python3
"""Deterministic WHY/HOW supervisor for the Paper Boy surface comparison."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha_swarm.surface_audit_contract import (  # noqa: E402
    COMPARISON_SCHEMA,
    EXPECTED_SYMBOL_COUNT,
    FRESHNESS_CRITICAL_SECONDS,
    FRESHNESS_HIGH_SECONDS,
    FRESHNESS_MEDIUM_SECONDS,
    HEARTBEAT_HIGH_SECONDS,
    HEARTBEAT_MEDIUM_SECONDS,
    RULESET_VERSION,
    SCHEMA,
    SEVERITY_ORDER,
    finding as _finding,
    parse_timestamp as _parse_timestamp,
    recommendation as _recommendation,
)
from scripts.alpha_swarm.surface_audit_ledger import (  # noqa: E402
    LedgerError,
    append_if_changed,
)
from scripts.alpha_swarm.surface_audit_output import (  # noqa: E402
    load_input,
    publish_report,
)
from scripts.alpha_swarm.surface_audit_runtime import (  # noqa: E402
    parse_args,
    run_loop,
)

DEFAULT_INPUT = ROOT / "outputs/alpha_swarm_pilot/surface_comparison/latest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "alpha_swarm_pilot" / "surface_audit"
DEFAULT_HTML = ROOT / "cockpit" / "paper_boy_audit.html"


def _safety_findings(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    safety = comparison.get("safety")
    expected = {
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "can_mutate_paper_artifacts": False,
        "can_override_approval_decision": False,
        "aggregate_score_computed": False,
        "hindsight_use": "comparison_only",
    }
    if not isinstance(safety, dict):
        return [
            _finding(
                "SAFETY-001",
                "safety",
                "critical",
                "The comparison safety contract is missing or malformed.",
                ["JSON path $.safety is not an object"],
                _recommendation(
                    key="restore_comparison_safety_contract",
                    rule_id="SAFETY-001",
                    category="safety",
                    severity="critical",
                    why_code="safety_contract_missing",
                    steps=[
                        "Stop treating the comparison as trusted",
                        "Restore all explicit negative-authority declarations",
                    ],
                    validation=[
                        "Run safety-contract tests",
                        "Confirm every expected field exactly matches",
                    ],
                ),
            )
        ]
    mismatches = [
        f"$.safety.{field}: expected {wanted!r}, observed {safety.get(field)!r}"
        for field, wanted in expected.items()
        if safety.get(field) != wanted
    ]
    if mismatches:
        findings.append(
            _finding(
                "SAFETY-002",
                "safety",
                "critical",
                "One or more no-authority invariants are absent or violated.",
                mismatches,
                _recommendation(
                    key="quarantine_unsafe_comparison_contract",
                    rule_id="SAFETY-002",
                    category="safety",
                    severity="critical",
                    why_code="negative_authority_invariant_failed",
                    steps=[
                        "Stop the audit input from being treated as safe",
                        "Review the comparison producer before restarting its report loop",
                    ],
                    validation=[
                        "All safety fields match the supervisor contract",
                        "No execution or mutation path is registered",
                    ],
                ),
            )
        )
    return findings


def _freshness_findings(
    comparison: dict[str, Any], generated_at: datetime
) -> list[dict[str, Any]]:
    report_time = _parse_timestamp(comparison.get("generated_at"))
    if report_time is None:
        return [
            _finding(
                "FRESH-001",
                "freshness",
                "high",
                "Comparison freshness cannot be established.",
                ["$.generated_at is missing or invalid"],
                _recommendation(
                    key="restore_comparison_timestamp",
                    rule_id="FRESH-001",
                    category="freshness",
                    severity="high",
                    why_code="comparison_timestamp_missing",
                    steps=["Restore a timezone-aware generated_at field"],
                    validation=["Supervisor can calculate non-negative input age"],
                ),
            )
        ]
    age = (generated_at - report_time.astimezone(UTC)).total_seconds()
    if age < -60:
        return [
            _finding(
                "FRESH-002",
                "freshness",
                "high",
                "The comparison timestamp is materially in the future.",
                [f"comparison age seconds={age:.1f}"],
                _recommendation(
                    key="repair_surface_clock_alignment",
                    rule_id="FRESH-002",
                    category="freshness",
                    severity="high",
                    why_code="future_comparison_timestamp",
                    steps=["Inspect producer and system timezone handling"],
                    validation=[
                        "Generated timestamps are monotonic and not future-dated"
                    ],
                ),
            )
        ]
    thresholds = (
        (FRESHNESS_CRITICAL_SECONDS, "critical", "FRESH-005"),
        (FRESHNESS_HIGH_SECONDS, "high", "FRESH-004"),
        (FRESHNESS_MEDIUM_SECONDS, "medium", "FRESH-003"),
    )
    for threshold, severity, rule_id in thresholds:
        if age > threshold:
            return [
                _finding(
                    rule_id,
                    "freshness",
                    severity,
                    "The comparison report missed its expected refresh tolerance.",
                    [f"age_seconds={age:.1f}", f"threshold_seconds={threshold}"],
                    _recommendation(
                        key="restore_surface_comparison_cadence",
                        rule_id=rule_id,
                        category="freshness",
                        severity=severity,
                        why_code="comparison_report_stale",
                        steps=[
                            "Inspect the independent comparison-loop PID and log",
                            "Restart only after preserving failure evidence",
                        ],
                        validation=[
                            "Two consecutive reports arrive within 180 seconds"
                        ],
                    ),
                )
            ]
    return []


def _worker_findings(
    comparison: dict[str, Any], generated_at: datetime
) -> list[dict[str, Any]]:
    worker = comparison.get("paper_worker") or {}
    if worker.get("state") != "running":
        return [
            _finding(
                "WORKER-001",
                "reliability",
                "high",
                "Paper Boy is not reported as running.",
                [f"worker_state={worker.get('state')!r}", f"pid={worker.get('pid')!r}"],
                _recommendation(
                    key="recover_paper_worker_forward_only",
                    rule_id="WORKER-001",
                    category="reliability",
                    severity="high",
                    why_code="paper_worker_stopped",
                    steps=[
                        "Preserve transient state",
                        "Reconstruct only from immutable receipts if needed",
                        "Resume forward-only before the next locked event",
                    ],
                    validation=[
                        "Worker PID is alive",
                        "Heartbeat advances",
                        "No past event is backfilled",
                    ],
                ),
            )
        ]
    heartbeat = _parse_timestamp(worker.get("heartbeat_at"))
    if heartbeat is None:
        return [
            _finding(
                "WORKER-002",
                "reliability",
                "high",
                "Paper Boy heartbeat is unavailable.",
                ["$.paper_worker.heartbeat_at is missing or invalid"],
                _recommendation(
                    key="restore_paper_worker_heartbeat",
                    rule_id="WORKER-002",
                    category="reliability",
                    severity="high",
                    why_code="worker_heartbeat_missing",
                    steps=[
                        "Inspect worker state publication without touching event receipts"
                    ],
                    validation=["Heartbeat advances at the configured cadence"],
                ),
            )
        ]
    age = (generated_at - heartbeat.astimezone(UTC)).total_seconds()
    if age > HEARTBEAT_HIGH_SECONDS:
        severity, rule_id = "high", "WORKER-004"
    elif age > HEARTBEAT_MEDIUM_SECONDS:
        severity, rule_id = "medium", "WORKER-003"
    else:
        return []
    return [
        _finding(
            rule_id,
            "reliability",
            severity,
            "Paper Boy heartbeat is stale even though the worker surface says running.",
            [f"heartbeat_age_seconds={age:.1f}"],
            _recommendation(
                key="investigate_stale_worker_heartbeat",
                rule_id=rule_id,
                category="reliability",
                severity=severity,
                why_code="worker_heartbeat_stale",
                steps=[
                    "Verify the PID command line",
                    "Inspect the worker log and next locked event",
                ],
                validation=["Heartbeat advances twice without backfill"],
            ),
        )
    ]


def _evidence_findings(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    paper = comparison.get("paper_surface") or {}
    available = int(paper.get("available_symbol_count") or 0)
    if available != EXPECTED_SYMBOL_COUNT:
        findings.append(
            _finding(
                "EVIDENCE-001",
                "evidence",
                "high",
                "The latest paper surface does not contain all expected symbols.",
                [f"available={available}", f"expected={EXPECTED_SYMBOL_COUNT}"],
                _recommendation(
                    key="restore_six_symbol_surface_coverage",
                    rule_id="EVIDENCE-001",
                    category="evidence",
                    severity="high",
                    why_code="paper_symbol_coverage_incomplete",
                    steps=[
                        "Identify missing immutable publication artifacts",
                        "Report missing evidence without fabricating it",
                    ],
                    validation=["Six symbols are present with source hashes"],
                ),
            )
        )
    missing_provenance = [
        row.get("symbol", "unknown")
        for row in paper.get("symbols", [])
        if row.get("available")
        and (not row.get("source_path") or not row.get("source_sha256"))
    ]
    if missing_provenance:
        findings.append(
            _finding(
                "EVIDENCE-002",
                "evidence",
                "medium",
                "Some paper rows lack explicit source provenance.",
                [f"symbols={','.join(missing_provenance)}"],
                _recommendation(
                    key="add_explicit_paper_provenance",
                    rule_id="EVIDENCE-002",
                    category="evidence",
                    severity="medium",
                    why_code="paper_source_provenance_missing",
                    steps=["Add source path and SHA-256 to each derived row"],
                    validation=["Every available row has both provenance fields"],
                ),
            )
        )
    bad_events = {
        status: count
        for status, count in (paper.get("current_session_event_counts") or {}).items()
        if status in {"failed", "missed", "blocked"} and count
    }
    if bad_events:
        findings.append(
            _finding(
                "EVIDENCE-003",
                "evidence",
                "high",
                "The current paper session contains failed, missed, or blocked events.",
                [f"{key}={value}" for key, value in sorted(bad_events.items())],
                _recommendation(
                    key="audit_current_session_event_failures",
                    rule_id="EVIDENCE-003",
                    category="evidence",
                    severity="high",
                    why_code="current_session_event_failure",
                    steps=[
                        "Preserve receipts",
                        "Classify root cause",
                        "Do not backfill",
                    ],
                    validation=[
                        "Next session executes equivalent events within tolerance"
                    ],
                ),
            )
        )
    return findings


def _process_findings(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    temporal = comparison.get("temporal_alignment") or {}
    gap = temporal.get("prediction_to_live_minutes")
    if temporal.get("state") == "later_live_snapshot" and gap is not None:
        findings.append(
            _finding(
                "PROCESS-001",
                "learning_loop",
                "info",
                "The live surface is intentionally later than Paper Boy's frozen prediction.",
                [f"prediction_to_live_minutes={gap}", "same_decision_window=false"],
                _recommendation(
                    key="archive_named_surface_checkpoints",
                    rule_id="PROCESS-001",
                    category="learning_loop",
                    severity="low",
                    why_code="continuous_report_overwrites_horizon_path",
                    steps=[
                        "Propose append-only comparison snapshots at noon, 14:00, locked exit, and receipt publication",
                        "Keep snapshots observational and outside evaluator inputs",
                    ],
                    validation=[
                        "A replay reconstructs how the live thesis changed without altering the frozen candidate"
                    ],
                ),
            )
        )
    comparison_block = comparison.get("spy_comparison") or {}
    if comparison_block.get("action_alignment") == "both_no_action":
        findings.append(
            _finding(
                "PROCESS-002",
                "learning_loop",
                "info",
                "Both systems currently preserve a no-action posture for different observations.",
                [
                    f"direction_change={comparison_block.get('direction_change')}",
                    f"paper_decision={comparison_block.get('paper_decision')}",
                    f"live_posture={comparison_block.get('live_execution_posture')}",
                ],
                _recommendation(
                    key="queue_post_pilot_gate_blocker_taxonomy",
                    rule_id="PROCESS-002",
                    category="learning_loop",
                    severity="low",
                    why_code="stand_down_causes_need_repeated_sample",
                    steps=[
                        "After the locked pilot ends, count predeclared gate-block reasons by slot",
                        "Review repeated blockers offline before proposing any new rule version",
                    ],
                    validation=[
                        "No aggregate score is exposed during the pilot",
                        "Any rule change creates a new manifest and version",
                    ],
                ),
            )
        )
    return findings


def _deduplicate_recommendations(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for finding in findings:
        recommendation = finding.get("how")
        if not recommendation:
            continue
        key = recommendation["recommendation_key"]
        previous = selected.get(key)
        if (
            previous is None
            or SEVERITY_ORDER[recommendation["severity"]]
            > SEVERITY_ORDER[previous["severity"]]
        ):
            selected[key] = recommendation
    return [selected[key] for key in sorted(selected)]


def _finalize(report: dict[str, Any]) -> None:
    findings = report["findings"]
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for finding in findings:
        counts[finding["severity"]] += 1
    report["why"] = {
        "status": (
            "action_required"
            if counts["critical"] or counts["high"]
            else "healthy_with_observations"
        ),
        "severity_counts": counts,
        "summary": [finding["why"] for finding in findings],
    }
    report["how"] = {
        "recommendations": _deduplicate_recommendations(findings),
        "automatic_changes_allowed": False,
        "next_gate": "operator_review_then_offline_validation",
    }


def build_audit(
    comparison: dict[str, Any], generated_at: datetime, input_sha256: str | None
) -> dict[str, Any]:
    findings = []
    if comparison.get("schema") != COMPARISON_SCHEMA:
        findings.append(
            _finding(
                "INPUT-001",
                "input",
                "critical",
                "The expected surface-comparison contract is unavailable.",
                [f"observed_schema={comparison.get('schema')!r}"],
                _recommendation(
                    key="restore_surface_comparison_input",
                    rule_id="INPUT-001",
                    category="input",
                    severity="critical",
                    why_code="comparison_contract_unavailable",
                    steps=[
                        "Inspect the comparison loop and latest.json publication",
                        "Preserve malformed input for diagnosis",
                    ],
                    validation=[f"Input schema equals {COMPARISON_SCHEMA}"],
                ),
            )
        )
    else:
        findings.extend(_safety_findings(comparison))
        findings.extend(_freshness_findings(comparison, generated_at))
        findings.extend(_worker_findings(comparison, generated_at))
        findings.extend(_evidence_findings(comparison))
        findings.extend(_process_findings(comparison))
    report = {
        "schema": SCHEMA,
        "ruleset_version": RULESET_VERSION,
        "generated_at": generated_at.isoformat(),
        "input_observation": {
            "schema": comparison.get("schema"),
            "generated_at": comparison.get("generated_at"),
            "sha256": input_sha256,
            "headline": comparison.get("headline"),
        },
        "findings": findings,
        "closed_loop": {
            "observe": "read immutable paper evidence and current live surface",
            "compare": "publish horizon-aware comparison",
            "audit": "apply deterministic WHY rules",
            "propose": "fingerprint HOW recommendations",
            "operator_review": "required before implementation",
            "offline_validate": "required before a new rule version",
            "version_release": "new manifest required for strategy or evaluator changes",
            "execution": "never authorized by this loop",
        },
        "thresholds": {
            "comparison_freshness_seconds": {
                "medium": FRESHNESS_MEDIUM_SECONDS,
                "high": FRESHNESS_HIGH_SECONDS,
                "critical": FRESHNESS_CRITICAL_SECONDS,
            },
            "worker_heartbeat_seconds": {
                "medium": HEARTBEAT_MEDIUM_SECONDS,
                "high": HEARTBEAT_HIGH_SECONDS,
            },
        },
        "safety": {
            "observational_only": True,
            "authoritative": False,
            "execution_permitted": False,
            "self_modification_allowed": False,
            "automatic_parameter_tuning_allowed": False,
            "frozen_artifact_mutation_allowed": False,
            "aggregate_score_computed": False,
        },
    }
    _finalize(report)
    return report


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    comparison, input_sha, input_error = load_input(args.input)
    report = build_audit(comparison, generated_at, input_sha)
    if input_error:
        report["findings"][0]["evidence"].append(input_error)
        _finalize(report)
    try:
        report["recommendation_ledger"] = append_if_changed(
            args.ledger, report["how"]["recommendations"], generated_at
        )
    except LedgerError as exc:
        report["findings"].append(
            _finding(
                "LEDGER-001",
                "ledger",
                "critical",
                "The recommendation ledger failed hash-chain verification; append is suppressed.",
                [str(exc)],
                _recommendation(
                    key="preserve_and_review_corrupt_recommendation_ledger",
                    rule_id="LEDGER-001",
                    category="ledger",
                    severity="critical",
                    why_code="ledger_chain_invalid",
                    steps=[
                        "Preserve the ledger unchanged",
                        "Review the first invalid line and prior hash",
                    ],
                    validation=[
                        "Independent verification passes before any future append"
                    ],
                ),
            )
        )
        _finalize(report)
        report["recommendation_ledger"] = {
            "verified": False,
            "changed": False,
            "append_suppressed": True,
            "error": str(exc),
        }
    publish_report(report, args.output_json, args.output_markdown, args.output_html)
    return report


def main() -> int:
    args = parse_args(
        input_path=DEFAULT_INPUT,
        output_root=DEFAULT_OUTPUT_ROOT,
        html_path=DEFAULT_HTML,
    )
    return run_loop(args, run_once)


if __name__ == "__main__":
    raise SystemExit(main())
