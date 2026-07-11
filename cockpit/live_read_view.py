"""View helpers for the SharpEdge Live Read cockpit."""

from __future__ import annotations

import html
from typing import Any

from execution_flow_view import (
    render_authority_adjudication_block,
    render_bucket_conditioned_spine_block,
    render_market_day_block,
)
from execution_state_view import (
    render_execution_state_packets_block,
    render_structure_state_block,
)
from execution_hierarchy import (
    ADVISORY_SURFACE_PART_NAMES,
    CONTEXT_GOVERNOR_PART_NAMES,
    CORE_EXECUTION_SPINE_PART_NAMES,
    SECONDARY_CONFIRMATION_PART_NAMES,
    SUSPECT_DRIFT_VOICE_PART_NAMES,
    part_label,
)
from level_state_view import render_level_state_block
from setup_context_view import (
    active_setup_level_badge,
    monthly_context_section,
    setup_section,
    weekly_context_section,
)
from targeting import infer_target, reachability_context
from timeframe_agreement_view import render_timeframe_agreement_block
from transition_pressure_view import render_transition_pressure_block

_active_setup_level_badge = active_setup_level_badge

FG = "#e6edf3"
MUTE = "#7d8590"
SURFACE = "#161b22"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"

HANDOFF_TAG = "EXHAUSTION -> RUNNER HANDOFF"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_price(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def summarize_permission_scores(
    permission: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    scored = []
    ignored = set(ADVISORY_SURFACE_PART_NAMES)
    for name, item in (permission.get("scores") or {}).items():
        if name in ignored:
            continue
        label = part_label(name)
        scored.append(
            {
                "name": name,
                "label": label,
                "score": int(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            }
        )
    ranked = sorted(scored, key=lambda row: row["score"], reverse=True)
    return {"best": ranked[:3], "worst": list(reversed(ranked[-3:]))}


def _fmt_ts_time(value: Any) -> str:
    text = str(value or "")
    return text[11:16] if len(text) >= 16 and "T" in text else text[:5] or "n/a"


def _phase_badge(item: dict[str, Any]) -> str:
    phase = str(item.get("phase") or "").lower()
    if not phase:
        return ""
    color = {
        "head": GREEN,
        "body": BLUE,
        "tail": AMBER,
        "inactive": MUTE,
    }.get(phase, MUTE)
    return (
        f'<div style="display:inline-block;padding:2px 6px;border:1px solid {color};'
        f'border-radius:999px;color:{color};font-size:10px;font-weight:bold">{_esc(phase.upper())}</div>'
    )


def render_permission_score_trend(permission_trend: dict[str, Any]) -> str:
    points = permission_trend.get("points") or []
    if not points:
        return ""
    cells = "".join(
        f'<div style="min-width:88px">'
        f'<div style="color:{MUTE};font-size:11px">{_esc(point.get("time", "?"))}</div>'
        f'<div style="color:{FG};font-weight:bold;font-size:16px">{_esc(point.get("score", "?"))}</div>'
        f'<div style="color:{CYAN};font-size:10px;margin-top:4px">'
        f"{'<br>'.join(_esc(marker) for marker in (point.get('event_markers') or []))}</div></div>"
        for point in points
    )
    direction = str(permission_trend.get("direction", "new")).upper()
    delta = permission_trend.get("delta")
    delta_text = f"{delta:+d}" if isinstance(delta, int) else "n/a"
    changes = "".join(
        f'<li style="margin:2px 0"><span style="color:{GREEN if item.get("delta", 0) > 0 else RED}">{"▲" if item.get("delta", 0) > 0 else "▼"} {_esc(item.get("feature", "?"))} {item.get("delta", 0):+d}</span></li>'
        for item in (permission_trend.get("largest_changes_since_last_update") or [])
    )
    changes_html = (
        f'<div style="margin-top:8px"><div style="color:{MUTE};font-size:11px">Largest changes since last update</div><ul style="color:{FG};font-size:12px;margin:6px 0 0 16px">{changes}</ul></div>'
        if changes
        else ""
    )
    setup_transitions = (
        permission_trend.get("setup_transitions_since_last_update") or []
    )
    setup_items = "".join(
        f'<li style="margin:2px 0"><span style="color:{CYAN}">{_esc(item.get("label", "setup"))}</span>'
        f'<span style="color:{MUTE}"> • {_esc(_fmt_ts_time(item.get("ts")))}</span></li>'
        for item in setup_transitions
    )
    setup_html = (
        f'<div style="margin-top:8px"><div style="color:{MUTE};font-size:11px">Setup lifecycle since last update</div><ul style="color:{FG};font-size:12px;margin:6px 0 0 16px">{setup_items}</ul></div>'
        if setup_items
        else ""
    )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid #30363d;border-radius:6px;background:#0d1117">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">'
        f'<div style="color:{BLUE};font-weight:bold;font-size:12px">PERMISSION SCORE TREND</div>'
        f'<div style="color:{MUTE};font-size:11px">Direction: {_esc(direction)} • delta {delta_text}</div></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px">{cells}</div>'
        f"{changes_html}{setup_html}</div>"
    )


def render_permission_overview(
    permission: dict[str, Any],
    permission_trend: dict[str, Any] | None = None,
) -> str:
    if not permission:
        return ""
    return render_permission_score_trend(permission_trend or {})


def render_edge_token_block(edge_token_position: dict[str, Any] | None = None) -> str:
    edge_token_position = edge_token_position or {}
    if not edge_token_position:
        return ""

    action = str(edge_token_position.get("suggested_action") or "stand_down")
    token = edge_token_position.get("current_token") or {}
    closing_token = edge_token_position.get("closing_token") or {}
    side = str(token.get("side") or closing_token.get("side") or "NEUTRAL")
    color = {
        "enter_call": GREEN,
        "enter_put": RED,
        "hold": BLUE,
        "close_position": AMBER,
        "flip_to_call": PURPLE,
        "flip_to_put": PURPLE,
        "stand_down": MUTE,
    }.get(action, BLUE)
    policy = edge_token_position.get("policy") or {}
    status_line = [
        f"position: {str(edge_token_position.get('position_state') or 'flat').upper()}",
        f"contracts held: {edge_token_position.get('contracts_held', 0)}",
        f"per token: {policy.get('contracts_per_token', 1)}",
    ]
    details = []
    if token.get("event_type"):
        details.append(f"active token: {token.get('event_type')} / {side}")
    if token.get("status"):
        details.append(f"status: {str(token.get('status')).upper()}")
    if token.get("observation_count"):
        details.append(f"seen {token.get('observation_count')}x")
    if token.get("level_name"):
        details.append(
            f"level: {token.get('level_name')} {_fmt_price(token.get('level_price'))}"
        )
    if closing_token.get("clear_reason"):
        details.append(
            f"closing token: {closing_token.get('event_type', 'prior token')} ({closing_token.get('clear_reason')})"
        )
    details_html = (
        f'<div style="color:{CYAN};font-size:11px;margin-top:6px">'
        f"{' • '.join(_esc(item) for item in details)}</div>"
        if details
        else ""
    )
    return (
        f'<div style="margin-top:10px;padding:12px;border:1px solid {color};border-radius:8px;background:{SURFACE}">'
        f'<div style="color:{color};font-weight:bold;font-size:16px">EDGE TOKEN ENGINE: {_esc(action.upper())}</div>'
        f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(edge_token_position.get("action_reason") or "")}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px;color:{MUTE};font-size:11px">'
        f"{''.join(f'<div>{_esc(item)}</div>' for item in status_line)}"
        f"</div>{details_html}"
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px">Policy: buy one contract per active token and exit when it clears (invalidated / expired).</div>'
        "</div>"
    )


def render_regime_refinement_block(
    regime_refinement: dict[str, Any] | None = None,
) -> str:
    regime_refinement = regime_refinement or {}
    if not regime_refinement:
        return ""
    annotations = regime_refinement.get("annotations") or []
    primary = str(regime_refinement.get("primary_behavior") or "unclassified_balance")
    summary = str(regime_refinement.get("behavior_summary") or "")
    visible = annotations[:4]
    items = "".join(
        f'<li style="margin:5px 0"><span style="color:{CYAN};font-weight:bold">{_esc(item.get("label", "behavior"))}</span>'
        f'<span style="color:{MUTE}"> • {_esc(item.get("bucket", "bucket"))}</span>'
        f'<div style="color:{FG};font-size:12px;margin-top:2px">{_esc(item.get("behavior", ""))}</div></li>'
        for item in visible
    )
    token_note = regime_refinement.get("token_annotation") or {}
    eligible = token_note.get("eligible_behavior_labels") or []
    eligible_html = (
        f'<div style="color:{AMBER};font-size:11px;margin-top:8px">Token-context labels: {_esc(", ".join(eligible))}</div>'
        if eligible
        else ""
    )
    return (
        f'<div style="margin-top:10px;padding:12px;border:1px solid #30363d;border-radius:8px;background:{SURFACE}">'
        f'<div style="color:{PURPLE};font-weight:bold;font-size:14px">PHONE COMPANION: MARKET BEHAVIOR</div>'
        f'<div style="color:{FG};font-weight:bold;font-size:16px;margin-top:4px">{_esc(primary)}</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:4px">{_esc(summary)}</div>'
        f'<ul style="margin:8px 0 0 16px;padding:0">{items}</ul>'
        f"{eligible_html}"
        f'<div style="color:{MUTE};font-size:11px;margin-top:8px">Pure annotator only: no permission, token, or approval changes.</div>'
        "</div>"
    )


def _cluster_strip_levels(
    spot: float | None,
    exp_low: float | None,
    exp_high: float | None,
    ch_lo: float | None,
    ch_hi: float | None,
    pin: float | None,
    put_wall: float | None,
    call_wall: float | None,
) -> list[tuple[str, float, str]]:
    core = [
        ("Channel lo", ch_lo, PURPLE),
        ("Exp low", exp_low, CYAN),
        ("Magnet", pin, AMBER),
        ("Spot", spot, FG),
        ("Exp high", exp_high, CYAN),
        ("Channel hi", ch_hi, PURPLE),
    ]
    clean_core = [level for level in core if isinstance(level[1], (int, float))]
    if not clean_core:
        return []

    core_values = [value for _label, value, _color in clean_core]
    core_lo = min(core_values)
    core_hi = max(core_values)
    core_span = max(core_hi - core_lo, 0.25)
    max_gap = max(core_span * 1.25, 0.5)

    levels = []
    if isinstance(put_wall, (int, float)) and put_wall >= core_lo - max_gap:
        levels.append(("Put wall", put_wall, GREEN))
    levels.extend(clean_core)
    if isinstance(call_wall, (int, float)) and call_wall <= core_hi + max_gap:
        levels.append(("Call wall", call_wall, RED))
    return levels


def render_location_strip(
    pa: dict[str, Any],
    op: dict[str, Any],
    micro: dict[str, Any],
    magnitude: dict[str, Any],
    gp: dict[str, Any],
) -> str:
    spot = pa.get("spot")
    exp_move = magnitude.get("exp_move_realized_usd")
    exp_low = (
        spot - exp_move
        if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float))
        else None
    )
    exp_high = (
        spot + exp_move
        if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float))
        else None
    )
    clean = _cluster_strip_levels(
        spot=spot,
        exp_low=exp_low,
        exp_high=exp_high,
        ch_lo=micro.get("ch_lo"),
        ch_hi=micro.get("ch_hi"),
        pin=gp.get("pin"),
        put_wall=op.get("put_wall"),
        call_wall=op.get("call_wall"),
    )
    if len(clean) < 2:
        return ""
    values = [level[1] for level in clean]
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 0.25)
    markers = []
    labels = []
    for label, value, color in clean:
        left = ((value - lo) / span) * 100
        markers.append(
            f'<div title="{_esc(label)} {_fmt_price(value)}" style="position:absolute;left:calc({left:.2f}% - 2px);top:2px;width:4px;height:20px;background:{color};border-radius:2px"></div>'
        )
        labels.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin-right:12px">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{color}"></span>'
            f"<span>{_esc(label)} {_esc(_fmt_price(value))}</span></div>"
        )
    return (
        f'<div style="margin-top:10px">'
        f'<div style="color:{MUTE};font-size:11px;margin-bottom:6px">LOCATION STRIP • magnet / expected-move bands / channel walls</div>'
        f'<div style="position:relative;height:24px;background:#0d1117;border:1px solid #30363d;border-radius:8px">'
        f"{''.join(markers)}</div>"
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;color:{MUTE};font-size:11px">'
        f"{''.join(labels)}</div></div>"
    )


def render_setup_conviction_block(permission: dict[str, Any]) -> str:
    setup = (permission or {}).get("setup_conviction") or {}
    score = setup.get("setup_conviction_score", 0)
    gate = str(setup.get("setup_gate") or "NONE")
    bias = str(setup.get("bias") or "NEUTRAL")
    color = {
        "ACTIONABLE": GREEN,
        "EMERGING": BLUE,
        "CONTEXT": PURPLE,
        "WATCH": AMBER,
    }.get(gate, MUTE)
    setup_tag = setup.get("setup_tag") or "No active setup"
    reason = setup.get("reason") or "No active setup card"
    is_handoff = str(setup_tag) == HANDOFF_TAG
    entry_gate = (setup.get("entry_gate") or {}).get("gate_id") or "none"
    context_gate = (setup.get("context_gate") or {}).get("gate_id") or "none"
    lifecycle = setup.get("event_lifecycle") or {}
    lifecycle_bits = []
    if lifecycle.get("status"):
        lifecycle_bits.append(f"status: {str(lifecycle.get('status')).upper()}")
    if lifecycle.get("observation_count"):
        lifecycle_bits.append(f"seen {lifecycle.get('observation_count')}x")
    if lifecycle.get("level_name"):
        lifecycle_bits.append(
            f"level: {lifecycle.get('level_name')} {_fmt_price(lifecycle.get('level_price'))}"
        )
    lifecycle_line = (
        f'<div style="color:{CYAN};font-size:11px;margin-top:6px">'
        f"{' • '.join(_esc(bit) for bit in lifecycle_bits)}</div>"
        if lifecycle_bits
        else ""
    )
    lifecycle_times = []
    if lifecycle.get("first_seen_ts"):
        lifecycle_times.append(
            f"first seen {_fmt_ts_time(lifecycle.get('first_seen_ts'))}"
        )
    if lifecycle.get("last_confirmed_ts"):
        lifecycle_times.append(
            f"last confirmed {_fmt_ts_time(lifecycle.get('last_confirmed_ts'))}"
        )
    times_line = (
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">'
        f"{' • '.join(_esc(bit) for bit in lifecycle_times)}</div>"
        if lifecycle_times
        else ""
    )
    banner = (
        f'<div style="display:inline-block;padding:4px 8px;border-radius:999px;background:{AMBER};'
        f'color:#0d1117;font-size:10px;font-weight:bold;letter-spacing:0.06em;margin-bottom:8px">'
        "PHASE PROMOTION • CONTINUATION MANAGEMENT</div>"
        if is_handoff
        else ""
    )
    reason_note = (
        f'<div style="color:{AMBER};font-size:12px;font-weight:bold;margin-top:6px">'
        "This setup has graduated beyond a simple VWAP fade.</div>"
        if is_handoff
        else ""
    )
    shell_style = (
        f"border:3px solid {AMBER};background:linear-gradient(180deg, #1f1608 0%, {SURFACE} 28%);padding:12px;margin:8px 0;border-radius:10px;box-shadow:0 0 0 1px #6e4f12 inset"
        if is_handoff
        else f"border:2px solid {color};background:{SURFACE};padding:12px;margin:8px 0;border-radius:8px"
    )
    return (
        f'<div style="{shell_style}">'
        f"{banner}"
        f'<div style="color:{color};font-weight:bold;font-size:18px">SETUP CONVICTION: {gate} / {score} / {bias}</div>'
        f'<div style="color:{FG};font-size:13px;margin-top:4px">{_esc(setup_tag)}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:4px;{WRAP}">{_esc(reason)}</div>'
        f"{reason_note}"
        f"{lifecycle_line}"
        f"{times_line}"
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px;color:{MUTE};font-size:11px">'
        f"<div>entry gate: {_esc(entry_gate)}</div><div>context gate: {_esc(context_gate)}</div></div></div>"
    )


def _interaction_badge(classification: Any) -> str:
    text = str(classification or "").lower()
    color = {
        "strongly_good": GREEN,
        "weakly_good": BLUE,
        "strongly_bad": RED,
        "weakly_bad": AMBER,
    }.get(text, MUTE)
    label = text.replace("_", " ").upper() or "UNCLASSIFIED"
    return (
        f'<span style="display:inline-block;padding:2px 6px;border:1px solid {color};'
        f'border-radius:999px;color:{color};font-size:10px;font-weight:bold">{_esc(label)}</span>'
    )


def render_execution_expansion_potential_block(permission: dict[str, Any]) -> str:
    packet = (permission or {}).get("execution_expansion_potential") or {}
    if not packet:
        return ""
    summary = packet.get("summary") or {}
    surface = packet.get("surface") or {}
    mechanisms = packet.get("mechanisms") or []
    state = str(summary.get("state") or "mixed").lower()
    accent = {
        "high_confirmation_high_fuel": GREEN,
        "low_confirmation_high_fuel": PURPLE,
        "high_confirmation_low_fuel": BLUE,
        "mixed": AMBER,
    }.get(state, CYAN)
    items = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid #30363d">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;flex-wrap:wrap">'
        f'<div style="color:{FG};font-size:12px;font-weight:bold">{_esc(item.get("label") or item.get("mechanism_id") or "mechanism")}</div>'
        f'<span style="display:inline-block;padding:2px 6px;border:1px solid {accent};border-radius:999px;color:{accent};font-size:10px;font-weight:bold">{_esc(str(item.get("strength") or "").upper())}</span></div>'
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(item.get("reason") or "")}</div>'
        "</div>"
        for item in mechanisms[:4]
    )
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {accent};border-radius:6px;background:#0d1117">'
        f'<div style="color:{accent};font-weight:bold;font-size:12px">EXPANSION POTENTIAL</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;color:{MUTE};font-size:11px">'
        f"<div>surface: {_esc(surface.get('score') or 'n/a')} / {_esc(surface.get('bias') or 'NEUTRAL')}</div>"
        f"<div>participation: {_esc(summary.get('participation_confirmation') or 'n/a')}</div>"
        f"<div>fuel: {_esc(summary.get('expansion_fuel') or 'n/a')}</div>"
        f"<div>state: {_esc(summary.get('state') or 'n/a')}</div>"
        "</div>"
        f'<div style="color:#adbac7;font-size:11px;margin-top:6px;{WRAP}">{_esc(summary.get("note") or "")}</div>'
        f'<div style="margin-top:8px">{items}</div></div>'
    )


def render_execution_vector_interactions_block(permission: dict[str, Any]) -> str:
    packet = (permission or {}).get("execution_vector_interactions") or {}
    if not packet:
        return ""
    summary = packet.get("summary") or {}
    favorable = packet.get("best") or []
    warnings = packet.get("warnings") or []
    balance = str(summary.get("interaction_balance") or "mixed").lower()
    accent = {
        "favorable": GREEN,
        "adverse": RED,
        "mixed": AMBER,
        "sparse": CYAN,
    }.get(balance, CYAN)

    def render_items(items: list[dict[str, Any]], empty: str) -> str:
        if not items:
            return f'<div style="color:{MUTE};font-size:11px">{_esc(empty)}</div>'
        return "".join(
            f'<div style="padding:6px 0;border-bottom:1px solid #30363d">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;flex-wrap:wrap">'
            f'<div style="color:{FG};font-size:12px;font-weight:bold">{_esc(item.get("label") or item.get("interaction_id") or "interaction")}</div>'
            f"{_interaction_badge(item.get('classification'))}</div>"
            f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(item.get("reason") or "")}</div>'
            "</div>"
            for item in items[:2]
        )

    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid {accent};border-radius:6px;background:#0d1117">'
        f'<div style="color:{accent};font-weight:bold;font-size:12px">VECTOR INTERACTIONS</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;color:{MUTE};font-size:11px">'
        f"<div>balance: {_esc(summary.get('interaction_balance') or 'n/a')}</div>"
        f"<div>good: {_esc(summary.get('favorable_count') or 0)}</div>"
        f"<div>bad: {_esc(summary.get('warning_count') or 0)}</div>"
        "</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:8px">'
        f'<div><div style="color:{MUTE};font-size:11px;margin-bottom:4px">Good combos</div>{render_items(favorable, "no favorable combos surfaced")}</div>'
        f'<div><div style="color:{MUTE};font-size:11px;margin-bottom:4px">Bad combos</div>{render_items(warnings, "no warning combos surfaced")}</div>'
        "</div></div>"
    )


def render_permission_section(
    permission: dict[str, Any],
    pa: dict[str, Any],
    op: dict[str, Any],
    micro: dict[str, Any],
    magnitude: dict[str, Any],
    gp: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
    permission_trend: dict[str, Any] | None = None,
) -> str:
    if not permission:
        return ""
    gate = permission.get("trade_gate", "BLOCK")
    score = permission.get("trade_permission_score", 0)
    bias = permission.get("bias", "NEUTRAL")
    color = {"PERMIT": GREEN, "CAUTION": AMBER}.get(gate, RED)
    setup_block = render_setup_conviction_block(permission)
    market_day_block = render_market_day_block(
        permission.get("market_day") or {}, permission.get("execution_flow") or {}
    )
    spine_block = render_bucket_conditioned_spine_block(
        permission.get("bucket_conditioned_spine") or {}
    )
    expansion_block = render_execution_expansion_potential_block(permission)
    interactions_block = render_execution_vector_interactions_block(permission)
    authority_block = render_authority_adjudication_block(
        permission.get("authority_adjudication") or {}
    )
    scores = permission.get("scores", {})
    rows = []
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        item = scores.get(name)
        if not item:
            continue
        label = part_label(name)
        phase_badge = _phase_badge(item)
        phase_reason = item.get("phase_reason")
        phase_reason_html = (
            f'<div style="color:{MUTE};font-size:10px;margin-top:3px">{_esc(phase_reason)}</div>'
            if phase_reason
            else ""
        )
        rows.append(
            f'<tr><td style="padding:3px 8px;color:{MUTE}">{label}</td>'
            f'<td style="padding:3px 8px;color:{FG};text-align:right">{item["score"]}</td>'
            f'<td style="padding:3px 8px;text-align:center">{phase_badge}</td>'
            f'<td style="padding:3px 8px;color:#adbac7;{WRAP}">{_esc(item["reason"])}{phase_reason_html}</td></tr>'
        )
    supporting_groups = [
        ("Secondary confirmations", SECONDARY_CONFIRMATION_PART_NAMES),
        ("Context governors", CONTEXT_GOVERNOR_PART_NAMES),
        ("Suspect drift voices", SUSPECT_DRIFT_VOICE_PART_NAMES),
        ("Advisory surfaces", ADVISORY_SURFACE_PART_NAMES),
    ]
    supporting_bits = []
    for label, names in supporting_groups:
        present = [part_label(name) for name in names if scores.get(name)]
        if present:
            supporting_bits.append(f"{label}: {', '.join(present)}")
    supporting_summary = (
        f'<div style="color:{MUTE};font-size:11px;margin-top:8px;{WRAP}">'
        f"Supporting surfaces off main spine: {' • '.join(_esc(bit) for bit in supporting_bits)}</div>"
        if supporting_bits
        else ""
    )
    reach = reachability_context(pa, op, permission, magnitude, gp, micro, setups)
    reach_color = {"within": GREEN, "stretch": AMBER, "beyond": RED}.get(
        reach["status"], BLUE
    )
    reach_text = (
        f"Strategic target: {_esc(reach['target_label'])} {_esc(_fmt_price(reach['target_price']))} • "
        f"distance {_esc(_fmt_price(reach['distance']))} • "
        f"remaining expected move {_esc(_fmt_price(reach['expected_move']))}"
        if reach["distance"] is not None
        else f"Strategic target: {_esc(reach['target_label'])} • remaining expected move {_esc(_fmt_price(reach['expected_move']))}"
    )
    reachable_today = reach.get("reachable_today") or {}
    reachable_text = ""
    if reachable_today.get("label"):
        reachable_text = (
            f'<div style="color:{FG};font-size:12px;margin-top:4px">'
            f"Reachable today: {_esc(reachable_today['label'])} {_esc(_fmt_price(reachable_today.get('price')))} "
            f"({reachable_today.get('status', 'unknown')}, distance {_esc(_fmt_price(reachable_today.get('distance')))})</div>"
        )
    likely_travel = reach.get("likely_travel")
    likely_text = (
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px">Today\'s likely travel: {_esc(likely_travel)}</div>'
        if likely_travel
        else ""
    )
    location_strip = render_location_strip(pa, op, micro, magnitude, gp)
    return (
        f"{setup_block}"
        f'<div style="border:2px solid {color};background:{SURFACE};padding:12px;margin:8px 0;border-radius:8px">'
        f'<div style="color:{color};font-weight:bold;font-size:18px">EXECUTION PERMISSION: {gate} / {score} / {bias}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:4px;{WRAP}">Bucket brain defines the battlefield. The bucket-conditioned spine is the primary execution authority. Main table shows core spine vectors only.</div>'
        f"{market_day_block}"
        f"{spine_block}"
        f"{authority_block}"
        f"{expansion_block}"
        f"{interactions_block}"
        f'<div style="margin-top:10px;padding:10px;border:1px solid #30363d;border-radius:6px;background:#0d1117">'
        f'<div style="color:{reach_color};font-weight:bold;font-size:12px">REMAINING EXPECTED MOVE VS DISTANCE TO TARGET</div>'
        f'<div style="color:{FG};font-size:13px;margin-top:4px;{WRAP}">{reach_text}</div>'
        f"{reachable_text}"
        f"{likely_text}"
        f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">{_esc(reach["reason"])}</div></div>'
        f"{location_strip}"
        f'<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:10px">'
        f'<tr style="color:{MUTE};font-size:10px;text-transform:uppercase">'
        f'<th style="padding:3px 8px;text-align:left;font-weight:normal">Surface</th>'
        f'<th style="padding:3px 8px;text-align:right;font-weight:normal">Score</th>'
        f'<th style="padding:3px 8px;text-align:center;font-weight:normal">Phase</th>'
        f'<th style="padding:3px 8px;text-align:left;font-weight:normal">Reason</th></tr>'
        f"{''.join(rows)}</table>"
        f"{supporting_summary}</div>"
    )


def render_live_read_html(
    pa: dict[str, Any],
    op: dict[str, Any],
    lines: list[tuple[str, str, str]],
    setups: list[dict[str, Any]] | None = None,
    permission: dict[str, Any] | None = None,
    micro: dict[str, Any] | None = None,
    magnitude: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
    permission_trend: dict[str, Any] | None = None,
    edge_token_position: dict[str, Any] | None = None,
    regime_refinement: dict[str, Any] | None = None,
    weekly_context: dict[str, Any] | None = None,
    monthly_context: dict[str, Any] | None = None,
    stamp: str = "",
    level_states: dict[str, Any] | None = None,
    timeframe_agreement: dict[str, Any] | None = None,
    transition_pressure: dict[str, Any] | None = None,
) -> str:
    color_map = {"ok": GREEN, "bad": RED, "warn": AMBER, "info": BLUE}
    sign = "+" if pa.get("day_chg", 0) >= 0 else ""
    cards = []
    for title, kind, detail in lines:
        color = color_map.get(kind, BLUE)
        cards.append(
            f'<div style="border-left:4px solid {color};background:{SURFACE};padding:10px 12px;margin:8px 0;border-radius:6px">'
            f'<div style="color:{color};font-weight:bold;font-size:15px">{_esc(title)}</div>'
            f'<div style="color:#adbac7;font-size:13px;margin-top:3px;{WRAP}">{_esc(detail)}</div></div>'
        )
    day_color = GREEN if pa.get("day_chg", 0) >= 0 else RED
    return f"""<!DOCTYPE html><html><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<meta http-equiv=\"refresh\" content=\"45\">
<title>SharpEdge Cockpit</title></head>
<body style=\"margin:0;min-height:100vh;overflow-x:hidden;overflow-y:auto;-webkit-overflow-scrolling:touch;touch-action:pan-y;overscroll-behavior-y:contain;background:#0d1117;color:#e6edf3;font-family:monospace\">
<div style=\"padding:12px 12px 28px\">
<div style=\"display:flex;justify-content:space-between;align-items:baseline\">
<h2 style=\"margin:0;font-size:18px\">SharpEdge Live Read - SPY</h2>
<span style=\"color:#7d8590;font-size:12px\">updated {stamp} | auto 45s</span>
</div>
<div style=\"font-size:26px;font-weight:bold;margin:6px 0\">${pa["spot"]:.2f}
<span style=\"font-size:16px;color:{day_color}\">{sign}{pa["day_chg"]:.2f}% today</span></div>
<img src=\"cockpit_chart.svg\" style=\"display:block;width:100%;border:1px solid #21262d;border-radius:8px\">\n{active_setup_level_badge(setups)}
{render_transition_pressure_block(transition_pressure)}
{render_timeframe_agreement_block(timeframe_agreement)}
{render_level_state_block(level_states or {})}
{render_execution_state_packets_block(permission or {})}
{render_permission_overview(permission or {}, permission_trend or {})}
<h3 style=\"font-size:14px;color:#e6edf3;margin:14px 0 4px\">BUCKET-CONDITIONED EXECUTION SPINE
(primary authority lane)</h3>
{render_permission_section(permission or {}, pa, op, micro or {}, magnitude or {}, gp or {}, setups, permission_trend or {})}
{weekly_context_section(weekly_context)}
{monthly_context_section(monthly_context)}
<h3 style=\"font-size:14px;color:#e6edf3;margin:14px 0 4px\">SETUPS
(failed breaks + exhaustion + compression)</h3>
{setup_section(setups)}
<h3 style=\"font-size:14px;color:#7d8590;margin:14px 0 4px\">THE READ
(context)</h3>
{"".join(cards)}
<p style=\"color:#484f58;font-size:11px;margin-top:14px\">Free data (Yahoo 1m + CBOE delayed options). Decision support only - you own every trade.</p>
</div></body></html>"""


__all__ = [
    "infer_target",
    "reachability_context",
    "render_execution_expansion_potential_block",
    "render_structure_state_block",
    "render_execution_state_packets_block",
    "render_execution_vector_interactions_block",
    "render_live_read_html",
    "render_location_strip",
    "render_permission_overview",
    "render_permission_score_trend",
    "render_regime_refinement_block",
    "render_permission_section",
    "render_setup_conviction_block",
    "summarize_permission_scores",
]
