#!/usr/bin/env python3
"""Build lightweight historical hints from the manual trade journal.

This artifact is intentionally non-authoritative. It summarizes tiny manually
logged trade history and written notes into machine-readable hints that can help
operator-facing app layers with context. It must never override the main safety
contract in agent_v1_decision.py.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUTDIR = Path("outputs")
DB_PATH = Path(os.getenv("TRADE_JOURNAL_DB", str(Path.home() / "trade_journal" / "trades.db"))).expanduser()
NOTES_DIR = Path(os.getenv("TRADE_JOURNAL_NOTES_DIR", "trade_journal"))
OUT_JSON = OUTDIR / "trade_journal_hints.json"
OUT_TXT = OUTDIR / "trade_journal_hints.txt"
MIN_PATTERN_SAMPLE_N = int(os.getenv("TRADE_HINT_MIN_PATTERN_SAMPLE_N", "5"))
TOP_PATTERN_LIMIT = int(os.getenv("TRADE_HINT_TOP_PATTERN_LIMIT", "3"))
DEFAULT_METRICS = [
    "MFE",
    "MAE",
    "convexity_capture_ratio",
    "hold_duration_minutes",
    "entry_timing_quality",
    "exit_timing_quality",
    "state_transition_classification",
]
SECTION_HEADERS = {
    "future_metrics": "future metrics to track",
    "hypotheses": "recursive learning hypothesis",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_trades(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trades ORDER BY trade_date DESC, id DESC").fetchall()
    except sqlite3.Error:
        return []
    finally:
        try:
            con.close()
        except UnboundLocalError:
            pass
    return [dict(row) for row in rows]


def normalize_text(value: Any, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    return text or default


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_hhmm(value: Any) -> int | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", text):
        return None
    hours, minutes = text.split(":", maxsplit=1)
    total = int(hours) * 60 + int(minutes)
    return total if 0 <= total < 24 * 60 else None


def hold_minutes(row: dict[str, Any]) -> int | None:
    entry = parse_hhmm(row.get("entry_time"))
    exit_ = parse_hhmm(row.get("exit_time"))
    if entry is None or exit_ is None or exit_ < entry:
        return None
    return exit_ - entry


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extract_section_bullets(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    capture = False
    bullets: list[str] = []
    target = header.lower().rstrip(":")
    for line in lines:
        stripped = line.strip()
        heading_text = stripped.lstrip("#").strip().lower().rstrip(":")
        if stripped.startswith("##") and capture:
            break
        if heading_text.startswith(target):
            capture = True
            continue
        if not capture:
            continue
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
            continue
        if stripped and not stripped.startswith("→") and not stripped.startswith("#"):
            if bullets:
                break
    return bullets


def read_note_hints(notes_dir: Path) -> dict[str, Any]:
    note_files = sorted(notes_dir.glob("*.md")) if notes_dir.exists() else []
    metrics: list[str] = []
    hypotheses: list[str] = []
    categories: list[str] = []

    for path in note_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        metrics.extend(extract_section_bullets(text, SECTION_HEADERS["future_metrics"]))
        hypotheses.extend(extract_section_bullets(text, SECTION_HEADERS["hypotheses"]))
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("category:"):
                categories.append(stripped.split(":", maxsplit=1)[1].strip())

    unique_metrics = dedupe_preserve_order(metrics)
    unique_hypotheses = dedupe_preserve_order(hypotheses)

    return {
        "note_count": len(note_files),
        "note_files": [str(path) for path in note_files],
        "categories": sorted(set(categories)),
        "future_metrics": unique_metrics,
        "hypotheses": unique_hypotheses,
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value.strip())
    return ordered


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}
    fields = ["risk_dollars", "pnl_r", "entry_time", "exit_time", "notes"]
    summary: dict[str, Any] = {}
    for field in fields:
        present = sum(1 for row in rows if row.get(field) not in (None, ""))
        summary[field] = {
            "present": present,
            "missing": total - present,
            "coverage": round(present / total, 3),
        }
    return summary


def pattern_confidence_label(closed_trades: int) -> str:
    if closed_trades >= max(10, MIN_PATTERN_SAMPLE_N * 2):
        return "EMERGING"
    if closed_trades >= MIN_PATTERN_SAMPLE_N:
        return "EARLY"
    if closed_trades >= 1:
        return "LOW_SAMPLE"
    return "NO_CLOSED_TRADES"


def sample_recommendation(label: str, total_pnl: float, win_rate: float | None) -> str:
    if label == "LOW_SAMPLE":
        return "observe_only"
    if total_pnl > 0 and (win_rate or 0.0) >= 0.55:
        return "prefer_when_context_matches"
    if total_pnl < 0:
        return "review_before_repeating"
    return "observe_only"


def summarize_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            normalize_text(row.get("setup")),
            normalize_text(row.get("vwap_behavior")),
        )
        groups[key].append(row)

    patterns: list[dict[str, Any]] = []
    for (setup, behavior), group_rows in groups.items():
        pnl_values = [value for row in group_rows if (value := as_float(row.get("pnl"))) is not None]
        r_values = [value for row in group_rows if (value := as_float(row.get("pnl_r"))) is not None]
        holds = [value for row in group_rows if (value := hold_minutes(row)) is not None]
        closed_trades = len(pnl_values)
        wins = sum(1 for value in pnl_values if value > 0)
        losses = sum(1 for value in pnl_values if value < 0)
        flat = sum(1 for value in pnl_values if value == 0)
        total_pnl = sum(pnl_values) if pnl_values else 0.0
        win_rate = (wins / closed_trades) if closed_trades else None
        exit_reason_counts = Counter(
            normalize_text(row.get("exit_reason"), default="")
            for row in group_rows
            if normalize_text(row.get("exit_reason"), default="")
        )
        confidence_label = pattern_confidence_label(closed_trades)

        patterns.append(
            {
                "pattern_id": f"{setup}__{behavior}".lower(),
                "condition": {"setup": setup, "vwap_behavior": behavior},
                "trades": len(group_rows),
                "closed_trades": closed_trades,
                "wins": wins,
                "losses": losses,
                "flat": flat,
                "win_rate": round_or_none(win_rate, 3),
                "avg_pnl": round_or_none(average(pnl_values), 2),
                "total_pnl": round(total_pnl, 2),
                "avg_r": round_or_none(average(r_values), 3),
                "avg_hold_minutes": round_or_none(average([float(v) for v in holds]), 1),
                "preferred_exit_reason": exit_reason_counts.most_common(1)[0][0] if exit_reason_counts else None,
                "confidence_label": confidence_label,
                "recommendation": sample_recommendation(confidence_label, total_pnl, win_rate),
            }
        )

    return sorted(
        patterns,
        key=lambda item: (
            item["total_pnl"],
            item["closed_trades"],
            item["win_rate"] or 0.0,
        ),
        reverse=True,
    )[:TOP_PATTERN_LIMIT]


def build_actionable_hints(
    patterns: list[dict[str, Any]],
    note_hints: dict[str, Any],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    for pattern in patterns:
        condition = pattern["condition"]
        label = pattern["confidence_label"]
        evidence = {
            "trades": pattern["trades"],
            "closed_trades": pattern["closed_trades"],
            "win_rate": pattern["win_rate"],
            "avg_pnl": pattern["avg_pnl"],
            "avg_r": pattern["avg_r"],
            "preferred_exit_reason": pattern["preferred_exit_reason"],
        }
        if pattern["total_pnl"] <= 0 and pattern["closed_trades"] == 0:
            continue
        summary = (
            f"Observed journal winners cluster around {condition['setup']} + {condition['vwap_behavior']}."
        )
        recommendation = "Use as secondary context only."
        if label == "LOW_SAMPLE":
            recommendation = (
                f"Treat as low-sample watch context until at least {MIN_PATTERN_SAMPLE_N} closed trades exist."
            )
        elif pattern["recommendation"] == "prefer_when_context_matches":
            recommendation = "Prefer this pattern when current context matches, but never override the contract."
        elif pattern["recommendation"] == "review_before_repeating":
            recommendation = "Review what broke before trusting this pattern again."

        hints.append(
            {
                "hint_id": f"pattern:{condition['setup']}:{condition['vwap_behavior']}".lower(),
                "hint_type": "pattern_repeat_watch",
                "summary": summary,
                "condition": condition,
                "confidence_label": label,
                "recommendation": recommendation,
                "authoritative": False,
                "evidence": evidence,
            }
        )

        exit_reason = pattern.get("preferred_exit_reason")
        if exit_reason and pattern["wins"] >= 1:
            hints.append(
                {
                    "hint_id": f"exit:{condition['setup']}:{condition['vwap_behavior']}".lower(),
                    "hint_type": "exit_discipline",
                    "summary": (
                        f"Profitable {condition['setup']} + {condition['vwap_behavior']} trades usually exited via {exit_reason}."
                    ),
                    "condition": condition,
                    "confidence_label": label,
                    "recommendation": "Preserve planned exits; do not improvise just because dopamine asked nicely.",
                    "authoritative": False,
                    "evidence": {
                        "wins": pattern["wins"],
                        "preferred_exit_reason": exit_reason,
                        "avg_hold_minutes": pattern["avg_hold_minutes"],
                    },
                }
            )
            break

    risk_coverage = coverage.get("risk_dollars", {})
    if risk_coverage and risk_coverage.get("coverage", 0.0) < 1.0:
        hints.append(
            {
                "hint_id": "logging:risk_dollars",
                "hint_type": "logging_discipline",
                "summary": "Some journal rows are missing planned risk dollars.",
                "confidence_label": "HIGH",
                "recommendation": "Log risk_dollars every time so PnL-in-R stops being interpretive dance.",
                "authoritative": False,
                "evidence": risk_coverage,
            }
        )

    if note_hints.get("future_metrics"):
        hints.append(
            {
                "hint_id": "research:future_metrics",
                "hint_type": "research_backlog",
                "summary": "Journal notes already define the next metrics worth collecting.",
                "confidence_label": "HIGH",
                "recommendation": "Prefer filling missing execution-attribution fields before inventing new strategy knobs.",
                "authoritative": False,
                "evidence": {"metric_count": len(note_hints["future_metrics"])},
            }
        )

    return hints[: max(3, TOP_PATTERN_LIMIT + 1)]


def build_payload() -> dict[str, Any]:
    rows = read_trades(DB_PATH)
    note_hints = read_note_hints(NOTES_DIR)
    patterns = summarize_patterns(rows)
    coverage = coverage_summary(rows)
    metrics = dedupe_preserve_order(note_hints.get("future_metrics", []) + DEFAULT_METRICS)

    total_trades = len(rows)
    closed_trades = sum(1 for row in rows if as_float(row.get("pnl")) is not None)
    hints = build_actionable_hints(patterns, note_hints, coverage)

    return {
        "schema_version": "trade_journal_hints.v1",
        "created_ts": utc_now(),
        "symbols": sorted({normalize_text(row.get("symbol")) for row in rows}) if rows else [],
        "data_sources": {
            "trade_db": str(DB_PATH),
            "journal_notes_dir": str(NOTES_DIR),
        },
        "sample_state": {
            "total_trades": total_trades,
            "closed_trades": closed_trades,
            "note_count": note_hints.get("note_count", 0),
            "minimum_pattern_sample_n": MIN_PATTERN_SAMPLE_N,
            "low_sample": closed_trades < MIN_PATTERN_SAMPLE_N,
        },
        "top_patterns": patterns,
        "actionable_hints": hints,
        "metric_collection_priorities": metrics[:7],
        "research_hypotheses": note_hints.get("hypotheses", []),
        "note_categories": note_hints.get("categories", []),
        "field_coverage": coverage,
        "usage_constraints": [
            "Non-authoritative: never override agent_v1_decision permissions or blockers.",
            "Low-sample historical patterns are observation aids, not deployment approval.",
            "Use these hints to improve review quality and logging discipline, not to force trades.",
        ],
    }


def render_text(payload: dict[str, Any]) -> str:
    sample = payload["sample_state"]
    patterns = payload.get("top_patterns", [])
    top = patterns[0] if patterns else {}
    top_condition = top.get("condition", {})
    top_summary = "none"
    if top_condition:
        top_summary = (
            f"{top_condition.get('setup')} + {top_condition.get('vwap_behavior')}"
            f" | trades={top.get('trades')} closed={top.get('closed_trades')}"
            f" win_rate={top.get('win_rate')} avg_r={top.get('avg_r')}"
            f" label={top.get('confidence_label')}"
        )
    metrics = payload.get("metric_collection_priorities", [])
    hypotheses = payload.get("research_hypotheses", [])
    hints = payload.get("actionable_hints", [])
    lines = [
        "SHARPEDGE TRADE JOURNAL HINTS",
        f"Created: {payload['created_ts']}",
        f"Trades logged: {sample['total_trades']} (closed={sample['closed_trades']})",
        f"Note count: {sample['note_count']}",
        f"Low sample: {sample['low_sample']}",
        f"Minimum pattern sample: {sample['minimum_pattern_sample_n']}",
        "",
        f"Top pattern: {top_summary}",
        "",
        "Actionable hints:",
    ]
    lines.extend([f"- {hint['summary']} {hint['recommendation']}" for hint in hints] or ["- none"])
    lines.extend(["", "Metric collection priorities:"])
    lines.extend([f"- {metric}" for metric in metrics] or ["- none"])
    lines.extend(["", "Research hypotheses:"])
    lines.extend([f"- {item}" for item in hypotheses] or ["- none"])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TXT.write_text(render_text(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"trade_journal_hints_trades={payload['sample_state']['total_trades']}")


if __name__ == "__main__":
    main()
