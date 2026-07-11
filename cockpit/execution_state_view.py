"""Render execution state packets for the live cockpit."""

from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#7d8590"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _color_for_bias(bias: str) -> str:
    return {"CALLS": GREEN, "PUTS": RED}.get(str(bias or "").upper(), AMBER)


def _block(title: str, color: str, lines: list[str]) -> str:
    body = "".join(lines)
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {color};'
        f'border-radius:6px;background:#0d1117">'
        f'<div style="color:{color};font-weight:bold;font-size:13px">{_esc(title)}</div>'
        f"{body}</div>"
    )


def render_structure_state_block(structure_state: dict[str, Any] | None = None) -> str:
    state = structure_state or {}
    if not state:
        return ""
    bias = str(state.get("bias") or "NEUTRAL")
    color = _color_for_bias(bias)
    name = str(state.get("state") or "unknown").replace("_", " ").upper()
    quality = str(state.get("sequence_quality") or "insufficient").upper()
    swing_highs = int(state.get("swing_high_count") or 0)
    swing_lows = int(state.get("swing_low_count") or 0)
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(name)} / {_esc(bias)} / quality {_esc(quality)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">swing highs: {_esc(swing_highs)} • swing lows: {_esc(swing_lows)}</div>',
    ]
    if state.get("has_sequence"):
        lines.append(
            f'<div style="color:{MUTE};font-size:11px;margin-top:4px">'
            f"spacing ok: {_esc(state.get('spacing_ok'))} • amplitude ok: {_esc(state.get('amplitude_ok'))} • fresh: {_esc(state.get('freshness_ok'))}</div>"
        )
    lines.append(
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("reason") or "")}</div>'
    )
    return _block("STRUCTURE STATE", color, lines)


def render_acceptance_state_block(
    acceptance_state: dict[str, Any] | None = None,
) -> str:
    state = acceptance_state or {}
    if not state:
        return ""
    bias = str(state.get("bias") or "NEUTRAL")
    color = _color_for_bias(bias)
    rep = state.get("representative_level") or {}
    rep_label = "n/a"
    if rep:
        rep_label = (
            f"{rep.get('level_name', '?')} {float(rep.get('level_price') or 0.0):.2f}"
        )
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(str(state.get("state") or "unknown").replace("_", " ").upper())} / {_esc(bias)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">accepted levels: {_esc(state.get("accepted_level_count") or 0)} • representative: {_esc(rep_label)}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("reason") or "")}</div>',
    ]
    return _block("ACCEPTANCE STATE", color, lines)


def render_location_state_block(location_state: dict[str, Any] | None = None) -> str:
    state = location_state or {}
    if not state:
        return ""
    bias = str(state.get("bias") or "NEUTRAL")
    color = _color_for_bias(bias)
    nearest = state.get("nearest_reference") or {}
    nearest_text = "n/a"
    if nearest:
        nearest_text = (
            f"{nearest.get('reference_name', '?')} "
            f"{float(nearest.get('reference_price') or 0.0):.2f}"
        )
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(str(state.get("state") or "unknown").replace("_", " ").upper())} / {_esc(bias)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">nearest ref: {_esc(nearest_text)} • refs tracked: {_esc(state.get("reference_count") or 0)}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("reason") or "")}</div>',
    ]
    return _block("LOCATION STATE", color, lines)


def render_dealer_state_block(dealer_state: dict[str, Any] | None = None) -> str:
    state = dealer_state or {}
    if not state:
        return ""
    bias = str(state.get("bias") or "NEUTRAL")
    color = _color_for_bias(bias)
    gamma = state.get("gamma_state") or {}
    pin = state.get("pin_state") or {}
    wall = state.get("wall_state") or {}
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(str(state.get("state") or "unknown").replace("_", " ").upper())} / {_esc(bias)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">gamma: {_esc(gamma.get("state") or "n/a")} • pin: {_esc(pin.get("state") or "n/a")} • wall: {_esc(wall.get("state") or "n/a")}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("reason") or "")}</div>',
    ]
    return _block("DEALER STATE", color, lines)


def render_volume_state_block(volume_state: dict[str, Any] | None = None) -> str:
    state = volume_state or {}
    if not state:
        return ""
    confirmation = str(state.get("confirmation") or "missing").upper()
    direction = str(state.get("move_direction") or "flat").upper()
    color = GREEN if confirmation == "CONFIRMED" else AMBER
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(confirmation)} / move {_esc(direction)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">local {float(state.get("local_mult") or 0.0):.2f}x • session {float(state.get("session_mult") or 0.0):.2f}x • aligned {float(state.get("aligned_volume_share") or 0.0):.0%}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("reason") or "")}</div>',
    ]
    return _block("VOLUME STATE", color, lines)


def render_trend_state_block(trend_state: dict[str, Any] | None = None) -> str:
    state = trend_state or {}
    if not state:
        return ""
    bias = str(state.get("bias") or "NEUTRAL")
    color = _color_for_bias(bias)
    components = state.get("component_states") or {}
    component_text = (
        " • ".join(f"{name}:{value}" for name, value in components.items()) or "n/a"
    )
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(str(state.get("state") or "unknown").replace("_", " ").upper())} / {_esc(bias)}</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">{_esc(component_text)}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("detail") or state.get("reason") or "")}</div>',
    ]
    return _block("TREND STATE", color, lines)


def render_time_state_block(time_state: dict[str, Any] | None = None) -> str:
    state = time_state or {}
    if not state:
        return ""
    color = AMBER
    clock = str(state.get("clock") or "n/a")
    minutes = state.get("minutes_since_open")
    lines = [
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(str(state.get("state") or "unknown").replace("_", " ").upper())} / NEUTRAL</div>',
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">clock: {_esc(clock)} • minutes since open: {_esc(minutes)}</div>',
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(state.get("detail") or state.get("reason") or "")}</div>',
    ]
    return _block("TIME STATE", color, lines)


def render_execution_state_packets_block(
    permission: dict[str, Any] | None = None,
) -> str:
    permission = permission or {}
    blocks = [
        render_structure_state_block(permission.get("structure_state") or {}),
        render_acceptance_state_block(permission.get("acceptance_state") or {}),
        render_location_state_block(permission.get("location_state") or {}),
        render_dealer_state_block(permission.get("dealer_state") or {}),
        render_volume_state_block(permission.get("volume_state") or {}),
        render_trend_state_block(permission.get("trend_state") or {}),
        render_time_state_block(permission.get("time_state") or {}),
    ]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    return (
        '<h3 style="font-size:14px;color:#e6edf3;margin:14px 0 4px">EXECUTION STATE PACKETS</h3>'
        '<div style="color:#7d8590;font-size:11px;margin-bottom:6px">'
        "state-first live audit of the execution brain</div>" + "".join(blocks)
    )


__all__ = [
    "render_acceptance_state_block",
    "render_dealer_state_block",
    "render_execution_state_packets_block",
    "render_location_state_block",
    "render_structure_state_block",
    "render_time_state_block",
    "render_trend_state_block",
    "render_volume_state_block",
]
