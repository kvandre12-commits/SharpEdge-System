"""Render helpers for SharpEdge Line Authority Engine."""

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
SURFACE = "#161b22"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_price(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"${value:.2f}"


def _fmt_pct(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:+.3f}%"


def _color_for_bias(bias: str) -> str:
    if bias == "CALLS":
        return GREEN
    if bias == "PUTS":
        return RED
    return AMBER


def render_line_authority_block(packet: dict[str, Any] | None = None) -> str:
    packet = packet or {}
    lines = list(packet.get("lines") or [])[:8]
    if not lines:
        return ""
    summary = packet.get("summary") or {}
    bias = str(summary.get("bias") or "NEUTRAL")
    accent = _color_for_bias(bias)
    cards = []
    for line in lines:
        line_bias = str(line.get("bias") or "NEUTRAL")
        color = _color_for_bias(line_bias)
        counts = line.get("acceptance_counts") or {}
        cards.append(
            f'<div style="padding:8px;border:1px solid #30363d;border-radius:6px;background:{SURFACE}">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;flex-wrap:wrap">'
            f'<div style="color:{FG};font-size:12px;font-weight:bold">{_esc(line.get("name") or "rail")} '
            f'<span style="color:{MUTE};font-size:10px">{_esc(line.get("role") or "reference")}</span></div>'
            f'<div style="color:{color};font-size:11px;font-weight:bold">{_esc(line_bias)} / {int(line.get("score") or 0)}</div></div>'
            f'<div style="color:{color};font-size:11px;font-weight:bold;margin-top:4px">'
            f"{_esc(str(line.get('event') or 'nearby').replace('_', ' ').upper())}</div>"
            f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">'
            f"{_fmt_price(line.get('price'))} · distance {_fmt_pct(line.get('distance_pct'))} · "
            f"above/below/at {int(counts.get('above') or 0)}/{int(counts.get('below') or 0)}/{int(counts.get('at_level') or 0)} · "
            f"{_esc(line.get('reason') or '')}</div></div>"
        )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {accent};border-radius:6px;background:#0d1117">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline;flex-wrap:wrap">'
        f'<div style="color:{accent};font-weight:bold;font-size:13px">LINE AUTHORITY ENGINE</div>'
        f'<div style="color:{accent};font-size:12px;font-weight:bold">{_esc(bias)} / {int(summary.get("score") or 0)}</div></div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">'
        f"Graph rails are advisory authority objects: VWAP, PDH/PDL, OR lines, balance rails, and midpoints. "
        f"Not weighted in execution permission yet.</div>"
        f'<div style="color:{FG};font-size:12px;margin-top:6px;{WRAP}">{_esc(summary.get("reason") or "")}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:8px">'
        f"{''.join(cards)}</div></div>"
    )


__all__ = ["render_line_authority_block"]
