#!/usr/bin/env python3
"""Summarize recent operator journal activity into a compact review artifact."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTDIR = Path("outputs")
JOURNAL_JSONL = OUTDIR / "operator_journal_append.jsonl"
WATCHLIST_JSON = OUTDIR / "operator_watchlist.json"
OUT_JSON = OUTDIR / "operator_session_review.json"
OUT_TXT = OUTDIR / "operator_session_review.txt"
LOOKBACK_ENTRIES = int(os.getenv("OPERATOR_SESSION_REVIEW_LOOKBACK", "20"))


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def top_counts(values: list[str], limit: int = 5) -> list[dict[str, Any]]:
    counter = Counter(v for v in values if v)
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]


def summarize_counts(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item['value']}={item['count']}" for item in items)


def build_review() -> dict[str, Any]:
    entries = read_jsonl(JOURNAL_JSONL)
    watchlist = read_json(WATCHLIST_JSON)
    recent = entries[-LOOKBACK_ENTRIES:]
    latest = recent[-1] if recent else {}

    action_counts = top_counts([str(e.get("operator_action", "")) for e in recent], limit=3)
    status_counts = top_counts([str(e.get("watchlist_status", "")) for e in recent], limit=3)
    broker_counts = top_counts(
        [str(e.get("broker_integration_status", "")) for e in recent], limit=3
    )

    blockers: list[str] = []
    flags: list[str] = []
    for entry in recent:
        blockers.extend([str(v) for v in entry.get("blocking_reasons", [])])
        flags.extend([str(v) for v in entry.get("risk_flags", [])])

    return {
        "schema_version": "operator_session_review.v1",
        "created_ts": utc_now(),
        "lookback_entries": LOOKBACK_ENTRIES,
        "journal_entries_total": len(entries),
        "journal_entries_reviewed": len(recent),
        "current_watchlist_active_count": watchlist.get("active_count", 0),
        "latest_entry": {
            "created_ts": latest.get("created_ts"),
            "symbol": latest.get("symbol"),
            "operator_action": latest.get("operator_action"),
            "watchlist_status": latest.get("watchlist_status"),
            "headline": latest.get("headline"),
            "blocking_reasons": latest.get("blocking_reasons", []),
            "risk_flags": latest.get("risk_flags", []),
        },
        "distributions": {
            "operator_actions": action_counts,
            "watchlist_statuses": status_counts,
            "broker_integration_statuses": broker_counts,
        },
        "top_blockers": top_counts(blockers),
        "top_risk_flags": top_counts(flags),
    }


def render_text(review: dict[str, Any]) -> str:
    latest = review["latest_entry"]
    actions = summarize_counts(review["distributions"]["operator_actions"])
    statuses = summarize_counts(review["distributions"]["watchlist_statuses"])
    blockers = summarize_counts(review["top_blockers"])
    flags = summarize_counts(review["top_risk_flags"])
    return "\n".join(
        [
            "SHARPEDGE OPERATOR SESSION REVIEW",
            f"Created: {review['created_ts']}",
            f"Journal entries reviewed: {review['journal_entries_reviewed']} / {review['journal_entries_total']}",
            f"Current active watchlist count: {review['current_watchlist_active_count']}",
            "",
            f"Latest action: {latest.get('operator_action', 'none')}",
            f"Latest watchlist status: {latest.get('watchlist_status', 'none')}",
            f"Latest headline: {latest.get('headline', 'none')}",
            "",
            f"Top actions: {actions}",
            f"Top statuses: {statuses}",
            f"Top blockers: {blockers}",
            f"Top risk flags: {flags}",
        ]
    ) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    review = build_review()
    OUT_JSON.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(review), encoding="utf-8")
    print(json.dumps(review, indent=2, sort_keys=True))
    print(f"operator_session_review_entries={review['journal_entries_reviewed']}")


if __name__ == "__main__":
    main()
