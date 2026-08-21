"""Stable contract and record builders for the Paper Boy surface audit."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SCHEMA = "sharpedge.paper_live_surface_audit.v1"
RULESET_VERSION = "1.0.0"
COMPARISON_SCHEMA = "sharpedge.paper_live_surface_comparison.v1"
SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
FRESHNESS_MEDIUM_SECONDS = 180
FRESHNESS_HIGH_SECONDS = 360
FRESHNESS_CRITICAL_SECONDS = 900
HEARTBEAT_MEDIUM_SECONDS = 120
HEARTBEAT_HIGH_SECONDS = 300
EXPECTED_SYMBOL_COUNT = 6


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def recommendation(
    *,
    key: str,
    rule_id: str,
    category: str,
    severity: str,
    why_code: str,
    steps: list[str],
    validation: list[str],
    guardrails: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "recommendation_key": key,
        "rule_id": rule_id,
        "category": category,
        "severity": severity,
        "why_code": why_code,
        "steps": steps,
        "validation": validation,
        "guardrails": guardrails
        or [
            "operator review required before implementation",
            "offline or report-only validation first",
            "no frozen artifact or execution-path mutation",
        ],
    }


def finding(
    rule_id: str,
    category: str,
    severity: str,
    why: str,
    evidence: list[str],
    how: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "finding_id": rule_id,
        "rule_version": RULESET_VERSION,
        "category": category,
        "severity": severity,
        "status": "open",
        "why": why,
        "evidence": evidence,
        "how": how,
    }
