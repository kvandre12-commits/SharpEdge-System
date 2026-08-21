"""Presentation and artifact I/O for the Paper Boy surface audit."""

from __future__ import annotations

from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Boy Closed-Loop WHY/HOW Audit",
        "",
        f"**Status: {report['why']['status']}**",
        "",
        "> Advisory only. This supervisor may explain and recommend; it cannot "
        "self-edit, tune rules, expose the hidden score, authorize, or execute.",
        "",
        "## WHY",
        "",
    ]
    if not report["findings"]:
        lines.append("- No findings.")
    for finding in report["findings"]:
        lines.extend(
            [
                f"### {finding['finding_id']} — {finding['severity'].upper()}",
                "",
                finding["why"],
                *[f"- Evidence: `{item}`" for item in finding["evidence"]],
                "",
            ]
        )
    lines.extend(["## HOW", ""])
    recommendations = report["how"]["recommendations"]
    if not recommendations:
        lines.append("- No material recommendation is active.")
    for recommendation in recommendations:
        lines.extend(
            [
                f"### {recommendation['recommendation_key']} — "
                f"{recommendation['severity'].upper()}",
                "",
                *[f"- Step: {step}" for step in recommendation["steps"]],
                *[f"- Validate: {step}" for step in recommendation["validation"]],
                *[f"- Guardrail: {step}" for step in recommendation["guardrails"]],
                "",
            ]
        )
    ledger = report.get("recommendation_ledger") or {}
    lines.extend(
        [
            "## Recommendation ledger",
            "",
            f"- Verified: **{ledger.get('verified')}**",
            f"- Material set changed: **{ledger.get('changed')}**",
            f"- Event: `{ledger.get('event_id')}`",
            f"- Transitions: "
            f"`{json.dumps(ledger.get('transitions') or {}, sort_keys=True)}`",
            "",
            "## Closed-loop boundary",
            "",
            *[f"- **{key}:** {value}" for key, value in report["closed_loop"].items()],
            "",
        ]
    )
    return "\n".join(lines)


def render_html(markdown: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Boy WHY/HOW Audit</title><style>
body{{background:#0a0f16;color:#e8edf5;font:16px/1.5 system-ui;margin:0}}
main{{max-width:980px;margin:auto;padding:24px}}pre{{white-space:pre-wrap;background:#111a27;
padding:20px;border:1px solid #30425c;border-radius:12px}}.safe{{color:#8ce99a}}
</style></head><body><main><div class="safe">ADVISORY • NO SELF-MODIFICATION • NO EXECUTION</div>
<pre>{escape(markdown)}</pre></main></body></html>"""


def load_input(path: Path) -> tuple[dict[str, Any], str | None, str | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}, None, "comparison input is missing"
    digest = sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, digest, f"comparison input is invalid: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return {}, digest, "comparison input is not a JSON object"
    return payload, digest, None


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def publish_report(
    report: dict[str, Any], json_path: Path, markdown_path: Path, html_path: Path
) -> None:
    markdown = render_markdown(report)
    atomic_write(json_path, json.dumps(report, indent=2) + "\n")
    atomic_write(markdown_path, markdown)
    atomic_write(html_path, render_html(markdown))
