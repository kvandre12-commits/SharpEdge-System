"""Render helpers for the SharpEdge level-state engine block."""

from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#7d8590"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _accent(state: str, role: str) -> str:
    if state in {"failed_break_reclaimed", "holding_above_support"}:
        return GREEN
    if state in {"failed_break_rejected", "holding_below_resistance"}:
        return RED
    if state.startswith("testing"):
        return AMBER
    return PURPLE if role == "reference" else BLUE


def render_level_state_block(level_states: dict[str, dict[str, Any]] | None = None) -> str:
    level_states = level_states or {}
    if not level_states:
        return ""
    cards = []
    for name in ("ORH", "ORL", "PDH", "PDL", "PDC"):
        state = level_states.get(name)
        if not state:
            continue
        event_state = str(state.get("event_state") or "unknown")
        role = str(state.get("role") or "reference")
        accent = _accent(event_state, role)
        acceptance = (state.get("acceptance") or {}).get("state") or "mixed"
        candidate = state.get("failed_break_candidate")
        candidate_line = (
            f'<div style="color:{accent};font-size:10px;font-weight:bold;margin-top:4px">{_esc(candidate)}</div>'
            if candidate
            else ""
        )
        cards.append(
            f'<div style="padding:8px;border:1px solid #30363d;border-radius:6px;background:#161b22">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;flex-wrap:wrap">'
            f'<div style="color:{FG};font-size:12px;font-weight:bold">{_esc(name)} {_esc(role.upper())}</div>'
            f'<div style="color:{MUTE};font-size:11px">{_esc(state.get("close_relation") or "unknown")} / {_esc(acceptance)}</div></div>'
            f'<div style="color:{accent};font-size:11px;font-weight:bold;margin-top:4px">{_esc(event_state.replace("_", " ").upper())}</div>'
            f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("summary") or "")}</div>'
            f"{candidate_line}</div>"
        )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {PURPLE};border-radius:6px;background:#0d1117">'
        f'<div style="color:{PURPLE};font-weight:bold;font-size:13px">LEVEL STATE ENGINE</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">'
        "Reference levels are classified before setup and authority layers interpret them.</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:8px">'
        f"{''.join(cards)}</div></div>"
    )


__all__ = ["render_level_state_block"]
