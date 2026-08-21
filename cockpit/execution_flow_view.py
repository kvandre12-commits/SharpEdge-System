"""Render helpers for the bucket brain + bucket-conditioned spine cockpit blocks.

View may derive presentation. View may not derive meaning. These helpers consume
upstream protocols, compress them for human cognition, and must not mutate or
reinterpret market state, bucket context, authority, or permission.
"""

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
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


BUCKET_DISPLAY_LABELS = {
    "a_plus_trend_day": "A+ TREND DAY",
    "failed_breakdown_long_day": "FAILED BREAKDOWN / LONG RECLAIM DAY",
    "failed_breakout_short_day": "FAILED BREAKOUT / SHORT REJECTION DAY",
    "range_balance_day": "RANGE / BALANCE DAY",
    "trap_noise_day": "TRAP-NOISE DAY",
    "news_vol_shock_day": "NEWS / VOL SHOCK DAY",
    "unclassified_day": "AWAITING CLEAN DAY TYPE",
}


def bucket_display_label(bucket: Any) -> str:
    text = str(bucket or "unknown")
    return BUCKET_DISPLAY_LABELS.get(text, text.replace("_", " ").upper())


def render_market_day_block(market_day: dict[str, Any], flow: dict[str, Any]) -> str:
    if not market_day:
        return ""
    bucket = market_day.get("bucket", "unknown")
    bucket_label = bucket_display_label(bucket)
    score = market_day.get("score", "NA")
    bias = market_day.get("bias", "NEUTRAL")
    playbooks = ", ".join(market_day.get("allowed_playbooks") or []) or "none"
    posture = market_day.get("risk_posture") or "n/a"
    vwap = market_day.get("vwap_context") or {}
    vwap_line = (
        f'<div style="color:{FG};font-size:12px;margin-top:5px">VWAP: '
        f"{_esc(vwap.get('state', 'unknown'))} / {_esc(vwap.get('posture', 'unknown'))} "
        f"({_esc(vwap.get('vs_vwap_pct', 'n/a'))}%)</div>"
    )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {AMBER};'
        f'border-radius:6px;background:#0d1117">'
        f'<div style="color:{AMBER};font-weight:bold;font-size:13px">'
        f"TODAY'S LIVE BATTLEFIELD: {_esc(bucket_label)} / {score} / {_esc(bias)}</div>"
        f'<div style="color:{FG};font-size:12px;margin-top:5px;{WRAP}">'
        f"Allowed playbooks: {_esc(playbooks)}</div>"
        f'<div style="color:{FG};font-size:12px;margin-top:4px">'
        f"Risk posture: {_esc(posture)}</div>"
        f"{vwap_line}"
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">'
        f"{_esc(market_day.get('reason'))}</div></div>"
    )


def render_bucket_conditioned_spine_block(spine: dict[str, Any]) -> str:
    if not spine:
        return ""
    bias = spine.get("bias", "NEUTRAL")
    gate = spine.get("gate", "BLOCK")
    color = {"CALLS": GREEN, "PUTS": RED}.get(bias, AMBER)
    action = spine.get("diagnostic_posture") or "watch_only_context"
    best = spine.get("best") or []
    best_html = "".join(
        f'<li style="margin:2px 0;{WRAP}">{_esc(item.get("name"))} {item.get("score")} — {_esc(item.get("reason"))}</li>'
        for item in best[:3]
    )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {color};'
        f'border-radius:6px;background:#0d1117">'
        f'<div style="color:{color};font-weight:bold;font-size:13px">'
        f"DIAGNOSTIC EXECUTION READ: {_esc(gate)} / {spine.get('score', 0)} / {_esc(bias)}</div>"
        f'<div style="color:{FG};font-size:12px;margin-top:5px">'
        f"Bucket-conditioned diagnostic posture: {_esc(action)} • bias strength {_esc(spine.get('bias_strength', 0))}</div>"
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">'
        f"{_esc(spine.get('reason'))}</div>"
        f'<ul style="color:#adbac7;font-size:11px;margin:6px 0 0 18px;padding:0">{best_html}</ul></div>'
    )


def render_authority_adjudication_block(packet: dict[str, Any]) -> str:
    if not packet:
        return ""
    doing = packet.get("cockpit_read") or packet.get("we_are_doing_this") or {}
    may_be = packet.get("this_may_be_occurring") or {}
    action = str(doing.get("action") or "watch_only")
    bias = str(doing.get("bias") or "NEUTRAL")
    accent = {
        "stand_down": AMBER,
        "watch_only": BLUE,
        "watch_edges": PURPLE,
        "candidate_calls": GREEN,
        "candidate_puts": RED,
    }.get(action, {"CALLS": GREEN, "PUTS": RED}.get(bias, CYAN))
    because = packet.get("because") or []
    despite = packet.get("despite") or []
    voices = packet.get("competing_voices") or []
    overridden = packet.get("overridden_voices") or []
    voice_cards = "".join(
        f'<div style="padding:8px;border:1px solid #30363d;border-radius:6px;background:#161b22">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;flex-wrap:wrap">'
        f'<div style="color:{FG};font-size:12px;font-weight:bold">{_esc(voice.get("label") or voice.get("voice_id") or "voice")}</div>'
        f'<div style="color:{MUTE};font-size:11px">{_esc(voice.get("score") if voice.get("score") is not None else "n/a")} / {_esc(voice.get("bias") or "NEUTRAL")}</div></div>'
        f'<div style="color:{CYAN};font-size:11px;margin-top:4px">{_esc(voice.get("stance") or "")}</div>'
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(voice.get("summary") or "")}</div>'
        f'<div style="color:{MUTE};font-size:10px;margin-top:4px">{_esc("ADVISORY ONLY" if voice.get("advisory_only") else voice.get("source") or "")}</div></div>'
        for voice in voices[:5]
    )
    because_html = "".join(
        f'<li style="margin:3px 0;{WRAP}">{_esc(item)}</li>' for item in because
    )
    despite_html = "".join(
        f'<li style="margin:3px 0;{WRAP}">{_esc(item)}</li>' for item in despite
    )
    overridden_html = (
        f'<div style="color:{MUTE};font-size:11px;margin-top:8px">Overridden / capped voices: {_esc(", ".join(str(item) for item in overridden))}</div>'
        if overridden
        else ""
    )
    despite_block = (
        f'<div><div style="color:{AMBER};font-size:11px;margin-bottom:4px">Despite</div><ul style="color:{FG};font-size:11px;margin:0 0 0 18px;padding:0">{despite_html}</ul></div>'
        if despite_html
        else ""
    )
    engine = str(doing.get("authority_engine") or "legacy")
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {accent};border-radius:6px;background:#0d1117">'
        f'<div style="color:{accent};font-weight:bold;font-size:13px">COCKPIT READ ADJUDICATION • {_esc(engine.upper())} ADVISORY</div>'
        f'<div style="color:{FG};font-size:12px;margin-top:6px;{WRAP}">{_esc(packet.get("summary") or "")}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:8px">'
        f'<div style="padding:8px;border:1px solid #30363d;border-radius:6px;background:#161b22">'
        f'<div style="color:{MUTE};font-size:11px">Context that may be occurring</div>'
        f'<div style="color:{FG};font-size:13px;font-weight:bold;margin-top:4px">{_esc(may_be.get("label") or "No active setup thesis")}</div>'
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(may_be.get("reason") or "")}</div></div>'
        f'<div style="padding:8px;border:1px solid #30363d;border-radius:6px;background:#161b22">'
        f'<div style="color:{MUTE};font-size:11px">Cockpit read posture</div>'
        f'<div style="color:{accent};font-size:13px;font-weight:bold;margin-top:4px">{_esc(doing.get("action") or "watch_only")}</div>'
        f'<div style="color:{FG};font-size:11px;margin-top:4px">{_esc(doing.get("gate") or "BLOCK")} / {_esc(doing.get("bias") or "NEUTRAL")} / {_esc(doing.get("bucket") or "unknown")}</div></div>'
        "</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:8px">'
        f'<div><div style="color:{GREEN};font-size:11px;margin-bottom:4px">Because</div><ul style="color:{FG};font-size:11px;margin:0 0 0 18px;padding:0">{because_html}</ul></div>'
        f"{despite_block}</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:10px">{voice_cards}</div>'
        f"{overridden_html}</div>"
    )


__all__ = [
    "render_authority_adjudication_block",
    "render_bucket_conditioned_spine_block",
    "render_market_day_block",
]
