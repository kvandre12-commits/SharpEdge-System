"""Render the cockpit timeframe agreement block."""

from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#7d8590"
SURFACE = "#161b22"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _color(kind: str) -> str:
    return {
        "ok": GREEN,
        "bad": RED,
        "warn": AMBER,
        "info": BLUE,
    }.get(kind, BLUE)


def _row_html(item: dict[str, Any]) -> str:
    color = _color(str(item.get("kind") or "info"))
    timeframe = _esc(item.get("timeframe"))
    label = _esc(item.get("label"))
    score = int(item.get("score") or 0)
    detail = _esc(item.get("detail"))
    return (
        f'<div style="border:1px solid #30363d;background:{SURFACE};padding:10px;border-radius:8px;margin-top:8px">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">'
        f'<div style="color:{MUTE};font-size:11px;text-transform:uppercase">{timeframe}</div>'
        f'<div style="color:{color};font-size:15px;font-weight:bold">{label} ({score})</div>'
        f"</div>"
        f'<div style="color:#adbac7;font-size:12px;margin-top:4px;{WRAP}">{detail}</div>'
        f"</div>"
    )


def render_timeframe_agreement_block(packet: dict[str, Any] | None) -> str:
    packet = packet or {}
    if not packet:
        return ""

    summary = _esc(packet.get("summary"))
    timeframes = packet.get("timeframes") or {}
    rows = [
        _row_html(timeframes[name])
        for name in ("weekly", "daily", "intraday")
        if isinstance(timeframes.get(name), dict)
    ]
    if not rows:
        return ""

    return (
        f'<div style="border-left:4px solid {BLUE};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{FG};font-weight:bold;font-size:14px">TIMEFRAME AGREEMENT</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:4px;{WRAP}">{summary}</div>'
        f"{''.join(rows)}"
        f"</div>"
    )


__all__ = ["render_timeframe_agreement_block"]
