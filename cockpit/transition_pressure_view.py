"""Render the transition pressure cockpit block."""

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
PURPLE = "#bc8cff"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _accent(state: str) -> str:
    return {
        "dormant": MUTE,
        "building": BLUE,
        "pressurized": AMBER,
        "release_candidate": PURPLE,
        "resolving": GREEN,
    }.get(state, BLUE)


def _delta_chip(label: str, payload: dict[str, Any]) -> str:
    velocity = int(payload.get("velocity") or 0)
    acceleration = int(payload.get("acceleration") or 0)
    color = GREEN if velocity > 0 else RED if velocity < 0 else MUTE
    return (
        f'<div style="border:1px solid #30363d;background:#0d1117;padding:8px 10px;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:10px;text-transform:uppercase">{_esc(label)}</div>'
        f'<div style="color:{color};font-size:15px;font-weight:bold;margin-top:2px">{velocity:+d}</div>'
        f'<div style="color:#adbac7;font-size:11px;margin-top:2px">accel {acceleration:+d} • {_esc(payload.get("status"))}</div>'
        f"</div>"
    )


def render_transition_pressure_block(packet: dict[str, Any] | None) -> str:
    packet = packet or {}
    if not packet:
        return ""
    state = str(packet.get("transition_state") or "building")
    accent = _accent(state)
    score = int(packet.get("transition_pressure_score") or 0)
    bias = str(packet.get("directional_bias") or "unclear").replace("_", " ")
    attention = str(packet.get("attention_state") or "watch").replace("_", " ")
    persistence = packet.get("persistence") or {}
    persistence_label = str(persistence.get("label") or "new_1_bar").replace("_", " ")
    reason = _esc(packet.get("reason"))
    lead = packet.get("permission_leads_price") or {}
    lead_color = GREEN if lead.get("active") else MUTE
    lead_text = (
        "permission leading price"
        if lead.get("active")
        else "permission not leading price yet"
    )
    deltas = packet.get("deltas") or {}
    delta_grid = "".join(
        _delta_chip(label, deltas.get(key) or {})
        for key, label in (
            ("permission_delta", "Permission Δ"),
            ("trend_delta", "Trend Δ"),
            ("acceptance_delta", "Acceptance Δ"),
            ("participation_delta", "Participation Δ"),
        )
    )
    energy = packet.get("potential_energy") or {}
    energy_bits = [
        f"compression {int(((energy.get('compression_score') or {}).get('score') or 0))}",
        f"failed auction {int(((energy.get('failed_auction_score') or {}).get('score') or 0))}",
        f"location {int(((energy.get('location_pressure') or {}).get('score') or 0))}",
        f"gamma {int(((energy.get('gamma_constraint') or {}).get('score') or 0))}",
    ]
    return (
        f'<div style="border-left:4px solid {accent};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">'
        f'<div style="color:{FG};font-weight:bold;font-size:14px">TRANSITION PRESSURE</div>'
        f'<div style="color:{accent};font-size:15px;font-weight:bold">{_esc(state.upper())} ({score})</div>'
        f"</div>"
        f'<div style="color:#adbac7;font-size:13px;margin-top:4px;{WRAP}">{reason}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px">'
        f'<div style="border:1px solid {accent};color:{accent};padding:3px 8px;border-radius:999px;font-size:11px">bias: {_esc(bias)}</div>'
        f'<div style="border:1px solid {BLUE};color:{BLUE};padding:3px 8px;border-radius:999px;font-size:11px">attention: {_esc(attention)}</div>'
        f'<div style="border:1px solid {PURPLE};color:{PURPLE};padding:3px 8px;border-radius:999px;font-size:11px">persistence: {_esc(persistence_label)}</div>'
        f'<div style="border:1px solid {lead_color};color:{lead_color};padding:3px 8px;border-radius:999px;font-size:11px">{_esc(lead_text)}</div>'
        f"</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:10px">{delta_grid}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:8px;{WRAP}">'
        f"Potential energy surfaces: {_esc(' • '.join(energy_bits))}</div>"
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">{_esc(lead.get("reason") or "")}</div>'
        f"</div>"
    )


__all__ = ["render_transition_pressure_block"]
