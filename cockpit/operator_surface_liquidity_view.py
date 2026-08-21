from __future__ import annotations

import html
from typing import Any, Callable


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _pressure_meter(
    liquidity: dict[str, Any], *, fg: str, mute: str, red: str, green: str
) -> str:
    put_pct = int(liquidity.get("put_pressure_pct") or 0)
    call_pct = int(liquidity.get("call_pressure_pct") or 0)
    dominant_side = liquidity.get("dominant_side") or "unknown"
    return (
        '<div style="margin-top:10px">'
        f'<div style="color:{mute};font-size:11px;margin-bottom:6px">Pressure split</div>'
        f'<div style="display:flex;height:12px;border-radius:999px;overflow:hidden;background:#0b1220;border:1px solid #22324d">'
        f'<div style="width:{put_pct}%;background:{red}"></div>'
        f'<div style="width:{call_pct}%;background:{green}"></div>'
        "</div>"
        f'<div style="display:flex;justify-content:space-between;gap:10px;margin-top:6px;font-size:12px">'
        f'<span style="color:{red}"><b>PUT</b> {put_pct}% • {_esc(liquidity.get("put_pressure_score") or 0)}</span>'
        f'<span style="color:{fg}">lead: {_esc(dominant_side)}</span>'
        f'<span style="color:{green}"><b>CALL</b> {call_pct}% • {_esc(liquidity.get("call_pressure_score") or 0)}</span>'
        "</div>"
        "</div>"
    )


def render_options_liquidity_card(
    brief: dict[str, Any],
    *,
    card: Callable[[str, str, str], str],
    chip: Callable[[Any, str], str],
    list_block: Callable[[list[str], str], str],
    status_color: Callable[[Any], str],
    fg: str,
    mute: str,
    blue: str,
    cyan: str,
    red: str,
    green: str,
) -> str:
    liquidity = brief.get("options_liquidity_read") or {}
    if not liquidity.get("available"):
        return card(
            "options liquidity",
            f'<div style="color:{mute}">No options-liquidity read available yet.</div>',
            cyan,
        )

    def section(title: str, items: list[str], color: str) -> str:
        if not items:
            return ""
        return (
            f'<div style="margin-top:10px">'
            f'<div style="color:{mute};font-size:11px;margin-bottom:6px">{_esc(title)}</div>'
            f"<div>{''.join(chip(item, color) for item in items)}</div>"
            "</div>"
        )

    body = (
        f'<div style="font-size:16px;font-weight:bold;color:{fg};line-height:1.35">{_esc(liquidity.get("plain_english") or liquidity.get("headline") or "No liquidity narrative yet.")}</div>'
        f'<div style="margin-top:8px;color:{mute};font-size:12px">Generated {_esc(liquidity.get("generated_at_utc") or "n/a")}</div>'
        f'<div style="margin-top:8px">'
        f"{chip('stance: ' + str(liquidity.get('stance') or 'unknown'), blue)}"
        f"{chip('balance: ' + str(liquidity.get('bias_alignment') or 'unknown'), status_color(liquidity.get('bias_alignment')))}"
        "</div>"
        f'<div style="margin-top:10px;color:{fg};font-size:13px"><b>Liquidity spot:</b> {_esc(liquidity.get("liquidity_spot") or "n/a")}</div>'
        f'<div style="margin-top:6px;color:{mute};font-size:13px"><b>Flow balance:</b> {_esc(liquidity.get("flow_balance") or "n/a")}</div>'
        f'<div style="margin-top:6px;color:{mute};font-size:13px"><b>Quote quality:</b> {_esc(liquidity.get("quote_quality_context") or "n/a")}</div>'
        + _pressure_meter(liquidity, fg=fg, mute=mute, red=red, green=green)
        + section("Put side", list(liquidity.get("put_flow") or []), red)
        + section("Call side", list(liquidity.get("call_flow") or []), green)
        + (
            f'<div style="margin-top:10px;color:{mute};font-size:12px"><b>Put summary:</b> {_esc(liquidity.get("put_side_summary") or "n/a")}</div>'
            if liquidity.get("put_side_summary")
            else ""
        )
        + (
            f'<div style="margin-top:6px;color:{mute};font-size:12px"><b>Call summary:</b> {_esc(liquidity.get("call_side_summary") or "n/a")}</div>'
            if liquidity.get("call_side_summary")
            else ""
        )
        + f'<div style="margin-top:10px;color:{mute};font-size:11px;margin-bottom:4px">Watch next</div>{list_block(list(liquidity.get("watch_next") or [])[:3], empty="none")}'
    )
    return card("options liquidity", body, cyan)
