#!/usr/bin/env python3
"""Build a local operator surface for workflow + connector context."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

BG = "#0d1117"
SURFACE = "#161b22"
FG = "#e6edf3"
MUTE = "#7d8590"
GRID = "#30363d"
BLUE = "#58a6ff"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _read_json(name: str) -> dict[str, Any]:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl_tail(name: str, limit: int = 3) -> list[dict[str, Any]]:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return []
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    return list(reversed(entries))


def _run_git(*args: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _card(title: str, body: str, accent: str = BLUE) -> str:
    return (
        f'<section style="background:{SURFACE};border:1px solid {GRID};border-left:5px solid {accent};'
        'border-radius:12px;padding:14px 14px 12px;margin-bottom:12px">'
        f'<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{MUTE};margin-bottom:8px">{_esc(title)}</div>'
        f"{body}</section>"
    )


def _chip(text: Any, color: str = BLUE) -> str:
    return (
        f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:4px 8px;'
        f'border:1px solid {color};border-radius:999px;color:{color};font-size:11px">'
        f"{_esc(text)}</span>"
    )


def _chip_block(items: list[Any], color: str, empty: str = "none") -> str:
    if not items:
        return f'<div style="color:{MUTE}">{_esc(empty)}</div>'
    return "".join(_chip(item, color) for item in items)


def _list_block(items: list[str], empty: str = "none") -> str:
    if not items:
        return f'<div style="color:{MUTE}">{_esc(empty)}</div>'
    rendered = "".join(
        f'<li style="margin:4px 0;color:{FG}">{_esc(item)}</li>' for item in items
    )
    return f'<ul style="margin:0 0 0 18px;padding:0">{rendered}</ul>'


def _kv_rows(rows: list[tuple[str, Any]]) -> str:
    rendered = "".join(
        f'<div><div style="color:{MUTE};font-size:11px">{_esc(label)}</div>'
        f'<div style="color:{FG};font-weight:bold;font-size:15px">{_esc(value if value not in (None, "") else "n/a")}</div></div>'
        for label, value in rows
    )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));'
        f'gap:10px">{rendered}</div>'
    )


def _status_color(value: Any) -> str:
    text = str(value or "").lower()
    if any(word in text for word in ("block", "hold", "down", "stale", "disabled")):
        return RED
    if any(word in text for word in ("review", "probe", "manual", "pending")):
        return AMBER
    if any(word in text for word in ("ready", "permit", "live", "confirmed")):
        return GREEN
    return BLUE


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _artifact_age_row(name: str) -> tuple[str, str, str]:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return name, "missing", RED
    age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    color = GREEN if age_seconds <= 900 else AMBER if age_seconds <= 7200 else RED
    return name, _human_age(max(age_seconds, 0)), color


def _recent_work_card() -> str:
    status_lines = _run_git("status", "--short")[:12]
    commit_lines = _run_git("log", "--since=2 hours ago", "--oneline", "-n", "8")
    body = (
        '<div style="font-size:18px;font-weight:bold;color:#e6edf3;margin-bottom:8px">What we touched recently</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px">'
        f'<div><div style="color:{MUTE};font-size:11px;margin-bottom:6px">Working tree changes</div>{_list_block(status_lines, empty="working tree clean")}</div>'
        f'<div><div style="color:{MUTE};font-size:11px;margin-bottom:6px">Recent commits (last ~2h)</div>{_list_block(commit_lines, empty="no recent commits in that window")}</div>'
        "</div>"
    )
    return _card("recent work", body, PURPLE)


def _connector_card() -> str:
    audit = _read_json("chatgpt_robinhood_connector_audit.json")
    if not audit:
        body = (
            '<div style="font-size:18px;font-weight:bold;color:#e6edf3">No connector audit artifact yet</div>'
            f'<div style="color:{MUTE};font-size:13px;margin-top:6px">Expected file: {_esc(OUTPUTS_DIR / "chatgpt_robinhood_connector_audit.json")}</div>'
            '<div style="color:#adbac7;font-size:13px;margin-top:8px">That means the connector-aware review surface has nothing fresh to display yet.</div>'
        )
        return _card("connector surface", body, AMBER)

    requested = audit.get("requested_action") or {}
    observed = audit.get("connector_observation") or {}
    follow_up = audit.get("operator_follow_up") or {}
    body = (
        _kv_rows(
            [
                ("created", audit.get("created_at")),
                ("task", requested.get("task_type")),
                ("symbol", requested.get("symbol")),
                ("connector status", observed.get("connector_status")),
                ("fill status", observed.get("fill_status")),
                ("broker order id", observed.get("broker_order_id")),
            ]
        )
        + f'<div style="margin-top:10px;color:{FG};font-size:14px">{_esc(observed.get("summary") or "No connector summary provided.")}</div>'
        + f'<div style="margin-top:10px;color:{MUTE};font-size:11px">Follow-up prompts</div>'
        + _list_block(
            (follow_up.get("prompts") or [])[:5], empty="no follow-up prompts"
        )
    )
    return _card("connector surface", body, GREEN)


def _hero_card(
    workflow: dict[str, Any],
    approval: dict[str, Any],
    brief: dict[str, Any],
    watchlist: dict[str, Any],
) -> str:
    blockers = (
        (brief.get("risk") or {}).get("blocking_reasons")
        or workflow.get("blockers")
        or []
    )
    risk_flags = (brief.get("risk") or {}).get("risk_flags") or []
    headline = brief.get("headline") or "No operator headline yet."
    summary = brief.get("summary") or {}
    body = (
        f'<div style="font-size:22px;font-weight:bold;color:{FG};line-height:1.3">{_esc(headline)}</div>'
        f'<div style="color:{MUTE};font-size:13px;margin-top:6px">Action {_chip(brief.get("operator_action") or workflow.get("operator_action") or "n/a", _status_color(brief.get("operator_action") or workflow.get("operator_action")))}'
        f"Readiness {_chip(workflow.get('readiness') or 'n/a', _status_color(workflow.get('readiness')))}"
        f"Approval {_chip(approval.get('decision') or workflow.get('approval_decision') or 'n/a', _status_color(approval.get('decision') or workflow.get('approval_decision')))}"
        f"Broker {_chip(summary.get('broker_integration_status') or 'n/a', _status_color(summary.get('broker_integration_status')))}"
        "</div>"
        + _kv_rows(
            [
                ("symbol", workflow.get("symbol") or brief.get("symbol")),
                ("risk state", summary.get("risk_state")),
                ("watchlist active", watchlist.get("active_count", 0)),
                ("blockers", len(blockers)),
                ("risk flags", len(risk_flags)),
                ("next steps", len(brief.get("next_steps") or [])),
            ]
        )
    )
    return _card("current state", body, _status_color(workflow.get("readiness")))


def _artifact_freshness_card() -> str:
    rows = [
        _artifact_age_row(name)
        for name in [
            "signal.json",
            "workflow_state.json",
            "approval_decision.json",
            "operator_brief.json",
            "operator_watchlist.json",
            "operator_journal_append.jsonl",
            "chatgpt_robinhood_connector_audit.json",
        ]
    ]
    body = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid {GRID}">'
        f'<span style="color:{MUTE};font-size:12px">{_esc(name)}</span>'
        f'<span style="color:{color};font-size:12px;font-weight:bold">{_esc(age)}</span></div>'
        for name, age, color in rows
    )
    return _card("artifact freshness", body, CYAN)


def _watchlist_card(watchlist: dict[str, Any]) -> str:
    items = (watchlist.get("items") or [])[:4]
    omitted = (watchlist.get("omitted_candidates") or [])[:3]
    if not items and not omitted:
        return _card(
            "watchlist",
            f'<div style="color:{MUTE}">No watchlist artifacts yet.</div>',
            BLUE,
        )

    def render_item(item: dict[str, Any], *, removed: bool = False) -> str:
        color = RED if removed else _status_color(item.get("status"))
        meta = [
            item.get("setup_type"),
            item.get("option_side"),
            f"{item.get('dte_target')} DTE" if item.get("dte_target") else None,
            item.get("status"),
        ]
        if removed and item.get("invalidation_reason"):
            meta.append(f"removed: {item.get('invalidation_reason')}")
        return (
            f'<div style="padding:8px 0;border-bottom:1px solid {GRID}">'
            f'<div style="color:{FG};font-size:14px;font-weight:bold">{_esc(item.get("headline") or item.get("item_id") or "watchlist item")}</div>'
            f'<div style="color:{MUTE};font-size:12px;margin-top:4px">'
            f"{''.join(_chip(part, color) for part in meta if part)}</div>"
            f'<div style="color:{MUTE};font-size:12px;margin-top:4px">spot {_esc(item.get("spot") or "n/a")} • ATM {_esc(item.get("atm_strike") or "n/a")} • dealer {_esc(item.get("dealer_state_hint") or "n/a")}</div>'
            "</div>"
        )

    body = (
        f'<div style="color:{MUTE};font-size:12px;margin-bottom:6px">Active count: {_esc(watchlist.get("active_count", 0))}</div>'
        + "".join(render_item(item) for item in items)
    )
    if omitted:
        body += (
            f'<div style="color:{MUTE};font-size:12px;margin-top:10px;margin-bottom:4px">Omitted tactical ideas</div>'
            + "".join(render_item(item, removed=True) for item in omitted)
        )
    return _card("watchlist", body, BLUE)


def _journal_card(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return _card(
            "latest operator journal",
            f'<div style="color:{MUTE}">No operator journal entries yet.</div>',
            PURPLE,
        )
    body = "".join(
        f'<div style="padding:8px 0;border-bottom:1px solid {GRID}">'
        f'<div style="color:{FG};font-size:14px;font-weight:bold">{_esc(entry.get("headline") or entry.get("operator_action") or "journal entry")}</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:4px">{_esc(entry.get("created_ts") or "n/a")} • '
        f"{_esc(entry.get('watchlist_status') or 'n/a')} • {_esc(entry.get('risk_state') or 'n/a')}</div>"
        f'<div style="margin-top:4px">{_chip_block(entry.get("blocking_reasons") or [], RED, empty="no blockers")}</div>'
        "</div>"
        for entry in entries
    )
    return _card("latest operator journal", body, PURPLE)


def render() -> str:
    workflow = _read_json("workflow_state.json")
    plan = _read_json("execution_plan.json")
    approval = _read_json("approval_decision.json")
    brief = _read_json("operator_brief.json")
    review = _read_json("operator_session_review.json")
    watchlist = _read_json("operator_watchlist.json")
    journal_entries = _read_jsonl_tail("operator_journal_append.jsonl", limit=3)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    blockers = plan.get("blocking_reasons") or workflow.get("blockers") or []
    risk_flags = (brief.get("risk") or {}).get("risk_flags") or []
    plan_steps = plan.get("steps") or brief.get("next_steps") or []

    workflow_body = _kv_rows(
        [
            ("run id", workflow.get("run_id")),
            ("symbol", workflow.get("symbol")),
            ("stage", workflow.get("lifecycle_stage")),
            ("operator action", workflow.get("operator_action")),
            ("readiness", workflow.get("readiness")),
            ("approval", workflow.get("approval_decision")),
        ]
    )
    approval_body = _kv_rows(
        [
            ("decision", approval.get("decision")),
            ("trade allowed", approval.get("trade_allowed")),
            ("broker order allowed", approval.get("broker_order_allowed")),
            ("required human action", approval.get("required_human_action")),
        ]
    )
    brief_body = _kv_rows(
        [
            ("headline", brief.get("headline")),
            (
                "broker integration",
                (brief.get("summary") or {}).get("broker_integration_status"),
            ),
            ("monitoring mode", (brief.get("summary") or {}).get("monitoring_mode")),
            ("watchlist active", watchlist.get("active_count")),
            ("latest session reviewed", review.get("latest_headline")),
            ("next steps", len(brief.get("next_steps") or [])),
        ]
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="45">
  <title>SharpEdge Operator Surface</title>
</head>
<body style="margin:0;min-height:100vh;overflow-y:auto;-webkit-overflow-scrolling:touch;touch-action:pan-y;overscroll-behavior-y:contain;background:{BG};color:{FG};font-family:ui-monospace,SFMono-Regular,Menlo,monospace">
  <div style="padding:14px 14px 28px;max-width:1100px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px">
      <div>
        <h1 style="margin:0;font-size:26px">SharpEdge Operator Surface</h1>
        <div style="color:{MUTE};font-size:12px;margin-top:4px">workflow + connector + recent work • updated {now}</div>
      </div>
      <div style="color:{MUTE};font-size:12px">first-class surfaces: cockpit.html • operator_surface.html</div>
    </div>
    {_hero_card(workflow, approval, brief, watchlist)}
    {_artifact_freshness_card()}
    {_recent_work_card()}
    {_connector_card()}
    {_watchlist_card(watchlist)}
    {_journal_card(journal_entries)}
    {_card("workflow state", workflow_body, BLUE)}
    {_card("approval state", approval_body, RED if blockers else GREEN)}
    {_card("operator brief", brief_body, AMBER)}
    {_card("blocking reasons", _chip_block(blockers, RED, empty="none"), RED if blockers else GREEN)}
    {_card("risk flags", _chip_block(risk_flags, AMBER, empty="none"), AMBER)}
    {_card("next steps", _list_block(plan_steps, empty="no next steps recorded"), BLUE)}
  </div>
</body>
</html>"""


def main() -> int:
    out = OUT_DIR / "operator_surface.html"
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
