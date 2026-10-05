"""Render setup badges/cards and higher-timeframe context sections."""

from __future__ import annotations

import html
from typing import Any

from gate_workflows import primary_trade_setup

FG = "#e6edf3"
MUTE = "#7d8590"
SURFACE = "#161b22"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"

HANDOFF_TAG = "EXHAUSTION -> RUNNER HANDOFF"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_price(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def active_setup_level_badge(setups: list[dict[str, Any]] | None = None) -> str:
    if not setups:
        return ""
    setup = primary_trade_setup(setups)
    level_name = setup.get("level_name")
    level_price = setup.get("level_price")
    if not level_name or not isinstance(level_price, (int, float)):
        return ""

    color = (
        GREEN
        if setup.get("tag") == "FAILED BREAKDOWN"
        else RED
        if setup.get("tag") == "FAILED BREAKOUT"
        else BLUE
    )
    trigger = setup.get("trigger_price")
    bars_ago = setup.get("bars_ago")
    extras = []
    if isinstance(trigger, (int, float)):
        extras.append(f"trigger ${trigger:.2f}")
    if isinstance(bars_ago, int):
        extras.append(f"{bars_ago}m ago")
    extra_text = f" • {' • '.join(extras)}" if extras else ""
    return (
        f'<div style="margin-top:8px;padding:8px 10px;border:1px solid {color};'
        f'border-radius:8px;background:#0d1117;color:{FG};font-size:12px">'
        f'<span style="color:{color};font-weight:bold">ACTIVE SETUP LEVEL</span> '
        f"{_esc(level_name)} {_fmt_price(level_price)}"
        f'<span style="color:{MUTE}">{_esc(extra_text)}</span></div>'
    )


def _context_section(
    context: dict[str, Any] | None,
    *,
    title: str,
    subtitle: str,
    image_src: str,
    default_note: str,
    default_headline: str,
    default_detail: str,
) -> str:
    if not context:
        return ""
    kind = str(context.get("kind") or "info")
    accent = {"ok": GREEN, "warn": AMBER, "bad": RED, "info": BLUE}.get(kind, BLUE)
    legend = context.get("legend") or []
    chips = "".join(
        f'<div style="border:1px solid {item.get("color", BLUE)};'
        f'color:{item.get("color", BLUE)};padding:3px 7px;border-radius:999px;font-size:10px">'
        f"{_esc(item.get('label', 'pivot'))} {_fmt_price(item.get('price'))}</div>"
        for item in legend
    )
    note = context.get("panel_note") or default_note
    headline = context.get("headline") or default_headline
    detail = context.get("detail") or default_detail
    return (
        f'<div style="margin-top:10px;padding:12px;border:1px solid #30363d;'
        f'border-radius:8px;background:{SURFACE}">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">'
        f'<div style="color:{BLUE};font-size:12px;font-weight:bold">{_esc(title)}</div>'
        f'<div style="color:{MUTE};font-size:11px">{_esc(subtitle)}</div></div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px">{_esc(note)}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">{chips}</div>'
        f'<div style="margin-top:8px;border-left:4px solid {accent};background:#0d1117;'
        f'padding:10px 12px;border-radius:6px">'
        f'<div style="color:{accent};font-weight:bold;font-size:15px">{_esc(headline)}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:4px">{_esc(detail)}</div></div>'
        f'<img src="{image_src}" style="width:100%;border:1px solid #21262d;'
        f'border-radius:8px;margin-top:8px"></div>'
    )


def weekly_context_section(weekly_context: dict[str, Any] | None = None) -> str:
    lookback_days = int((weekly_context or {}).get("lookback_days") or 0)
    subtitle = f"{lookback_days}-day carry map" if lookback_days else "5-day carry map"
    return _context_section(
        weekly_context,
        title="WEEKLY CONTEXT",
        subtitle=subtitle,
        image_src="cockpit_weekly_context.svg",
        default_note="Middle chart = weekly carry map. Bottom chart = monthly structure map.",
        default_headline="Weekly structure read",
        default_detail="No weekly context read available.",
    )


def monthly_context_section(monthly_context: dict[str, Any] | None = None) -> str:
    lookback_months = int((monthly_context or {}).get("lookback_months") or 0)
    subtitle = (
        f"{lookback_months}-month structure map"
        if lookback_months
        else "6-month structure map"
    )
    return _context_section(
        monthly_context,
        title="MONTHLY CONTEXT",
        subtitle=subtitle,
        image_src="cockpit_monthly_context.svg",
        default_note="Bottom chart = monthly structure map built from prior month rails + current month open.",
        default_headline="Monthly structure read",
        default_detail="No monthly context read available.",
    )


def setup_section(setups: list[dict[str, Any]] | None = None) -> str:
    if not setups:
        return (
            '<div style="border:1px dashed #30363d;background:#0d1117;'
            "padding:12px;margin:8px 0;border-radius:6px;color:#7d8590;"
            'font-size:13px">No failed-break or exhaustion setup right now '
            "- stand down, wait for the trap.</div>"
        )
    color_map = {"ok": GREEN, "bad": RED, "warn": AMBER, "info": BLUE}
    blocks = []
    for setup in setups:
        color = color_map.get(setup.get("kind"), BLUE)
        tag = str(setup.get("tag") or "")
        is_handoff = tag == HANDOFF_TAG
        shell_style = (
            f"border:3px solid {AMBER};background:linear-gradient(180deg, #1f1608 0%, {SURFACE} 28%);"
            f"padding:12px;margin:8px 0;border-radius:10px;box-shadow:0 0 0 1px #6e4f12 inset"
            if is_handoff
            else f"border:2px solid {color};background:{SURFACE};padding:12px;margin:8px 0;border-radius:8px"
        )
        banner = (
            f'<div style="display:inline-block;padding:4px 8px;border-radius:999px;background:{AMBER};'
            f'color:#0d1117;font-size:10px;font-weight:bold;letter-spacing:0.06em;margin-bottom:8px">'
            "PHASE PROMOTION • NOT JUST A VWAP FADE</div>"
            if is_handoff
            else ""
        )
        loud_note = (
            f'<div style="color:{AMBER};font-size:12px;font-weight:bold;margin-top:6px">'
            "Manage this like continuation now — the fade has already handed off.</div>"
            if is_handoff
            else ""
        )
        blocks.append(
            f'<div style="{shell_style}">'
            f"{banner}"
            f'<div style="color:{color};font-weight:bold;font-size:17px">'
            f"{_esc(tag)} &#8594; {_esc(setup.get('bias'))}</div>"
            f'<div style="color:#adbac7;font-size:13px;margin-top:4px">'
            f"{_esc(setup.get('detail'))}</div>"
            f"{loud_note}</div>"
        )
    return "".join(blocks)


__all__ = [
    "active_setup_level_badge",
    "monthly_context_section",
    "setup_section",
    "weekly_context_section",
]
