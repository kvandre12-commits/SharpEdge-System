"""View helpers for the SharpEdge Live Read cockpit."""

from __future__ import annotations

import html
import os
from typing import Any

from auction_audit_view import render_under_hood_audit_block
from candle_coach_view import render_candle_coach_block
from execution_flow_view import (
    render_authority_adjudication_block,
    render_bucket_conditioned_spine_block,
    render_market_day_block,
)
from execution_hierarchy import (
    ADVISORY_SURFACE_PART_NAMES,
    CONTEXT_GOVERNOR_PART_NAMES,
    CORE_EXECUTION_SPINE_PART_NAMES,
    SECONDARY_CONFIRMATION_PART_NAMES,
    SUSPECT_DRIFT_VOICE_PART_NAMES,
    part_label,
)
from execution_state_view import (
    render_core_spine_state_hint,
    render_execution_state_packets_block,
    render_execution_state_packets_details,
    render_structure_state_block,
)
from historical_refill_view import render_historical_refill_context_block
from level_state_view import render_level_state_block
from line_authority_view import render_line_authority_block
from post_apple_rotation_view import render_post_apple_rotation_block
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


def _page_refresh_seconds() -> int:
    raw = os.environ.get("COCKPIT_PAGE_REFRESH_SECONDS") or os.environ.get(
        "COCKPIT_REFRESH_SECONDS"
    )
    try:
        return max(int(raw or "10"), 1)
    except ValueError:
        return 10


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


def render_price_feed_lag_line(pa: dict[str, Any]) -> str:
    authority = pa.get("price_authority") or {}
    warnings = []
    if authority.get("price_feed_stale") is True:
        lag = authority.get("price_feed_lag_minutes")
        max_age = authority.get("price_feed_max_age_minutes") or 15
        source = pa.get("spot_source") or "price source"
        stamp = (
            authority.get("display_time_utc")
            or authority.get("last_bar_utc")
            or "unknown"
        )
        lag_text = f"{lag:.1f} min" if isinstance(lag, (int, float)) else "unknown age"
        warnings.append(
            f"PRICE FEED LAG · {_esc(source)} is {lag_text} old "
            f"(max {max_age} min) · last price stamp {_esc(stamp)}"
        )
    if authority.get("analysis_bar_stale") is True:
        lag = authority.get("analysis_bar_lag_minutes")
        max_age = authority.get("analysis_bar_max_age_minutes") or 15
        stamp = authority.get("last_bar_utc") or "unknown"
        lag_text = f"{lag:.1f} min" if isinstance(lag, (int, float)) else "unknown age"
        warnings.append(
            f"ANALYTICS BAR LAG · Yahoo 1m bars are {lag_text} old "
            f"(max {max_age} min) · VWAP/momentum/volume may be stale · "
            f"last bar {_esc(stamp)}"
        )
    if not warnings:
        return ""
    body = "<br>".join(warnings)
    return (
        f'<div style="border:2px solid {RED};background:#2d1111;color:{RED};'
        f'padding:8px;margin:4px 0 8px;border-radius:6px;font-size:13px;font-weight:bold;{WRAP}">'
        f"{body}<br>confirm against broker/live chart before using cockpit levels.</div>"
    )


def render_price_context_line(pa: dict[str, Any]) -> str:
    """Render non-authoritative quote context under the cockpit headline price."""
    authority = pa.get("price_authority") or {}
    bid = authority.get("cboe_bid")
    ask = authority.get("cboe_ask")
    if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
        return ""
    if bid <= 0 or ask <= 0 or bid > ask:
        return ""
    midpoint = (bid + ask) / 2.0
    display = pa.get("display_spot", pa.get("spot"))
    delta_text = ""
    if isinstance(display, (int, float)):
        delta = midpoint - display
        delta_text = f" &middot; Δ vs display {delta:+.2f}"
    trade_time = authority.get("cboe_last_trade_time_raw") or "delayed feed"
    return (
        f'<div style="color:{MUTE};font-size:11px;margin:-2px 0 8px;{WRAP}">'
        f"CBOE delayed options quote: bid/ask {_fmt_price(bid)} / {_fmt_price(ask)} "
        f"&middot; mid {_fmt_price(midpoint)}{delta_text} "
        f"&middot; context only, not top-price authority "
        f"&middot; {_esc(trade_time)}</div>"
    )


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
        f'<div style="color:{MUTE};font-size:10px;margin-top:5px;{WRAP}">Gamma/dealer language here is a gamma/OI proxy context read; execution authority still wins.</div>'
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
    audit = permission.get("authority_self_audit") or {}
    headline = str(audit.get("display_headline") or "EXECUTION READ")
    authority_note = str(
        audit.get("display_note")
        or "Score spine is a cockpit read; approval_decision is final authority."
    )
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
        state_hint = render_core_spine_state_hint(name, permission)
        phase_reason_html = (
            f'<div style="color:{MUTE};font-size:10px;margin-top:3px">{_esc(phase_reason)}</div>'
            if phase_reason
            else ""
        )
        rows.append(
            f'<tr><td style="padding:3px 8px;color:{MUTE}">{label}</td>'
            f'<td style="padding:3px 8px;color:{FG};text-align:right">{item["score"]}</td>'
            f'<td style="padding:3px 8px;text-align:center">{phase_badge}</td>'
            f'<td style="padding:3px 8px;color:#adbac7;{WRAP}">{_esc(item["reason"])}{state_hint}{phase_reason_html}</td></tr>'
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
        f"Diagnostic/supporting surfaces, not authority scores: {' • '.join(_esc(bit) for bit in supporting_bits)}</div>"
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
    packet_details = render_execution_state_packets_details(permission)
    unified_lane = " • ".join(
        part_label(name) for name in CORE_EXECUTION_SPINE_PART_NAMES
    )
    return (
        f"{setup_block}"
        f'<div style="border:2px solid {color};background:{SURFACE};padding:12px;margin:8px 0;border-radius:8px">'
        f'<div style="color:{color};font-weight:bold;font-size:18px">{_esc(headline)}: {gate} / {score} / {bias}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:4px;{WRAP}">{_esc(authority_note)}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px;{WRAP}">Authority inputs: {_esc(unified_lane)}</div>'
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
        f'<th style="padding:3px 8px;text-align:left;font-weight:normal">Authority Input</th>'
        f'<th style="padding:3px 8px;text-align:right;font-weight:normal">Diagnostic Score</th>'
        f'<th style="padding:3px 8px;text-align:center;font-weight:normal">Phase</th>'
        f'<th style="padding:3px 8px;text-align:left;font-weight:normal">Reason</th></tr>'
        f"{''.join(rows)}</table>"
        f"{supporting_summary}"
        f"{packet_details}</div>"
    )


# Directional lean per canonical auction bucket (Lane B inheritance).
_AUCTION_BULLISH = {"FAILED_BREAKDOWN", "CLEAN_BREAKOUT"}
_AUCTION_BEARISH = {"FAILED_BREAKOUT", "CLEAN_BREAKDOWN"}


def render_auction_context_block(auction_context: dict[str, Any] | None) -> str:
    """Headline the INHERITED auction bucket (prior completed session -> today).

    This is the real signal.auction_context computed by the canonical classifier.
    It sits at the top of the Live Read so the inherited auction is the primary
    context; day-type reads like STICKY/WHEE are demoted to texture below.
    """
    ctx = auction_context or {}
    if not ctx.get("available"):
        return ""
    bucket = str(ctx.get("bucket") or "UNCLASSIFIED")
    conf = int(ctx.get("confidence") or 0)
    ratio = ctx.get("range_atr_ratio")
    inherited_from = ctx.get("inherited_from_session")
    stale = ctx.get("calendar_days_stale")
    story = str(ctx.get("story") or "")

    if bucket in _AUCTION_BULLISH:
        color = GREEN
    elif bucket in _AUCTION_BEARISH:
        color = RED
    elif bucket == "RANGE_COMPRESSION":
        color = AMBER
    else:
        color = MUTE

    ratio_txt = f"{float(ratio):.2f}" if isinstance(ratio, (int, float)) else "n/a"
    proof = (
        f"inherited from {inherited_from}" if inherited_from else "no completed session"
    )
    if isinstance(stale, int) and stale > 5:
        proof += f" · {stale}d STALE"
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">INHERITED AUCTION (prior session &rarr; today)</div>'
        f'<div style="color:{color};font-size:22px;font-weight:bold;margin-top:4px">{_esc(bucket)}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:2px">confidence {conf} &middot; range/ATR {ratio_txt} &middot; {_esc(proof)}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        "</div>"
    )


_OPEN_RES_BULLISH = {
    "FAILED_BREAKDOWN_OPEN",
    "ACCEPTED_BREAKOUT_OPEN_2BAR",
    "ACCEPTED_BREAKOUT_OPEN_STRONG",
}
_OPEN_RES_BEARISH = {
    "FAILED_BREAKOUT_OPEN",
    "ACCEPTED_BREAKDOWN_OPEN_2BAR",
    "ACCEPTED_BREAKDOWN_OPEN_STRONG",
}


def render_open_resolution_block(open_resolution: dict[str, Any] | None) -> str:
    """Show how the OPEN resolved against yesterday's key levels (live).

    Real signal.open_resolution from the canonical open-resolution classifier.
    Sits directly under the inherited auction block: inherited context on top,
    today's open behavior right beneath it.
    """
    ctx = open_resolution or {}
    if not ctx.get("available"):
        return ""
    label = str(ctx.get("open_regime_label") or "NO_SETUP")
    if label == "NO_SETUP":
        return ""  # quiet morning: don't clutter the cockpit
    conf = int(ctx.get("confidence") or 0)
    setup_dir = str(ctx.get("setup_dir") or "NONE")
    key_source = str(ctx.get("key_source") or "")
    break_level = ctx.get("break_level")
    phase = str(ctx.get("phase") or "forming")
    story = str(ctx.get("story") or "")

    if label in _OPEN_RES_BULLISH:
        color = GREEN
    elif label in _OPEN_RES_BEARISH:
        color = RED
    elif phase == "forming":
        color = AMBER
    else:
        color = MUTE

    lvl_txt = (
        f"{float(break_level):.2f}" if isinstance(break_level, (int, float)) else "n/a"
    )
    meta = f"{setup_dir} setup &middot; {key_source} ${lvl_txt} &middot; conf {conf}"
    tag = "FORMING" if phase == "forming" else "RESOLVED"
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">OPEN RESOLUTION ({tag})</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(label)}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:2px">{meta}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        "</div>"
    )


_DEALER_BULLISH = {"SHORT_GAMMA", "CHASE"}
_DEALER_BEARISH = {"DEFENSIVE", "UNWIND_RISK"}


def render_dealer_positioning_block(dealer_positioning: dict[str, Any] | None) -> str:
    """Aggregate dealer gamma/OI proxy across the 0-1 DTE window.

    Real signal.dealer_positioning from the canonical dealer-positioning module.
    Surfaces gamma flip + dealer-state proxy that the single-expiry gamma card lacks.
    """
    ctx = dealer_positioning or {}
    if not ctx.get("available"):
        return ""
    state = str(ctx.get("dealer_state") or "NEUTRAL")
    flip = ctx.get("gamma_flip_strike")
    proxy = ctx.get("gamma_proxy")
    pcr_oi = ctx.get("pcr_oi")
    window = ctx.get("dte_window") or [0, 1]
    story = str(ctx.get("story") or "")
    max_oi = ctx.get("actionable_oi_wall_strike") or ctx.get("max_total_oi_strike")

    if state == "PINNED":
        color = AMBER
    elif state in _DEALER_BULLISH:
        color = GREEN
    elif state in _DEALER_BEARISH:
        color = RED
    else:
        color = BLUE

    flip_txt = f"${float(flip):.2f}" if isinstance(flip, (int, float)) else "n/a"
    proxy_txt = f"{float(proxy):+.2f}" if isinstance(proxy, (int, float)) else "n/a"
    pcr_txt = f"{float(pcr_oi):.2f}" if isinstance(pcr_oi, (int, float)) else "n/a"
    oi_txt = f"${float(max_oi):.2f}" if isinstance(max_oi, (int, float)) else "n/a"
    meta = (
        f"flip proxy {flip_txt} (spot {proxy_txt}) &middot; near OI wall {oi_txt} &middot; "
        f"PCR-OI {pcr_txt} &middot; {int(window[0])}-{int(window[1])} DTE"
    )
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">DEALER GAMMA/OI PROXY (0-1 DTE)</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(state)}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:2px">{meta}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        "</div>"
    )


def render_event_radar_block(event_radar: dict[str, Any] | None) -> str:
    """Macro-catalyst radar: FOMC / Jobs (NFP) / Treasury refunding.

    Real signal.event_radar from the canonical event calendar. Rendered at the
    very top because a scheduled catalyst changes the whole day's playbook
    (compression into the release, then an expansion/volatility window).
    """
    ctx = event_radar or {}
    if not ctx.get("available"):
        return ""
    headline = str(ctx.get("headline") or "")
    risk_window = bool(ctx.get("risk_window"))
    story = str(ctx.get("story") or "")
    upcoming = ctx.get("upcoming") or []
    social = ctx.get("social_catalyst") or {}
    latest_social = social.get("latest_relevant") or {}
    social_status = (social.get("source_status") or {}).get("status")

    next_event = ctx.get("next_event") or {}
    days_to = next_event.get("days_to")
    has_social_context = bool(latest_social) or social_status in {
        "ok",
        "source_blocked",
        "request_failed",
        "manual_override",
    }
    if not ctx.get("events_today") and next_event == {} and not has_social_context:
        return ""  # nothing scheduled/social in the whole window

    # Escalate color as the catalyst approaches.
    if risk_window:
        color = RED
    elif isinstance(days_to, int) and days_to <= 7:
        color = AMBER
    else:
        color = BLUE
    chips = " ".join(
        f'<span style="display:inline-block;padding:2px 8px;border:1px solid {MUTE};'
        f'border-radius:999px;color:#adbac7;font-size:11px;margin:2px 6px 2px 0">'
        f"{_esc(e.get('label'))} +{int(e.get('days_to', 0))}d</span>"
        for e in upcoming[:4]
    )
    social_html = ""
    if latest_social:
        impact = str(latest_social.get("impact") or "event").upper()
        terms = ", ".join(latest_social.get("matched_terms") or [])
        created = str(latest_social.get("created_at") or "")
        social_text = _esc(str(latest_social.get("text") or ""))
        terms_text = f" &middot; terms {_esc(terms)}" if terms else ""
        created_text = f" &middot; {_esc(created)}" if created else ""
        social_html = (
            f'<div style="border-top:1px solid #30363d;margin-top:8px;padding-top:8px">'
            f'<div style="color:{AMBER};font-size:11px;letter-spacing:.08em;font-weight:bold">TRUTH SOCIAL EVENT WATCH</div>'
            f'<div style="color:#e6edf3;font-size:12px;margin-top:4px;{WRAP}">{social_text}</div>'
            f'<div style="color:{MUTE};font-size:11px;margin-top:4px">impact {impact}'
            f"{terms_text}{created_text}</div>"
            "</div>"
        )
    elif social_status == "ok":
        count = int(social.get("status_count") or 0)
        social_html = (
            f'<div style="color:{MUTE};font-size:11px;margin-top:8px">'
            f"Truth Social scanner: no market-relevant Trump Truth in latest {count} posts."
            "</div>"
        )
    elif social_status in {"source_blocked", "request_failed"}:
        social_html = (
            f'<div style="color:{MUTE};font-size:11px;margin-top:8px">'
            f"Truth Social scanner: {_esc(str(social_status))}; manual override supported."
            "</div>"
        )
    return (
        f'<div style="border:2px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">EVENT RADAR (macro catalysts)</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(headline)}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        f'<div style="margin-top:8px">{chips}</div>'
        f"{social_html}"
        "</div>"
    )


def render_macro_overlay_block(macro_overlay: dict[str, Any] | None) -> str:
    """Live macro regime: VIX level, VIX term structure, 10Y rate impulse.

    Real signal.macro_overlay computed live from Yahoo ^VIX/^VIX3M/^TNX (no FRED
    key, no EOD lag). Term structure is the headline: contango = calm,
    backwardation = acute fear.
    """
    ctx = macro_overlay or {}
    if not ctx.get("available"):
        return ""
    state = str(ctx.get("macro_state") or "NEUTRAL")
    regime = str(ctx.get("term_regime") or "UNKNOWN")
    vix = ctx.get("vix")
    vix3m = ctx.get("vix3m")
    term = ctx.get("vix_term")
    story = str(ctx.get("story") or "")

    if state == "RISK_OFF" or regime == "BACKWARDATION":
        color = RED
    elif state == "RISK_ON_CALM":
        color = GREEN
    elif regime == "FLATTENING":
        color = AMBER
    else:
        color = BLUE

    vix_txt = f"{float(vix):.2f}" if isinstance(vix, (int, float)) else "n/a"
    vix3m_txt = f"{float(vix3m):.2f}" if isinstance(vix3m, (int, float)) else "n/a"
    term_txt = f"{float(term):.3f}" if isinstance(term, (int, float)) else "n/a"
    meta = f"{regime} &middot; VIX {vix_txt} / 3M {vix3m_txt} &middot; term {term_txt}"
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">MACRO OVERLAY (live VIX term / rates)</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(state)}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:2px">{meta}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        "</div>"
    )


def render_regime_read_block(regime_read: dict[str, Any] | None) -> str:
    """Daily structural regime backdrop from regime_daily.

    Real signal.regime_read: the last batch-computed regime_id / regime_label
    plus the day-over-day transition flag. This is an END-OF-DAY structural
    state (not a live recompute), so it is always stamped with its date /
    staleness. A transition day is highlighted because regime shifts often
    precede a volatility expansion.
    """
    ctx = regime_read or {}
    if not ctx.get("available"):
        return ""
    headline = str(ctx.get("headline") or "")
    story = str(ctx.get("story") or "")
    regime_date = str(ctx.get("date") or "")
    transition = bool(ctx.get("transition_flag"))
    stale_days = ctx.get("stale_days")

    # Transition day = amber highlight; stale (>5d) = muted; else blue.
    if transition:
        color = AMBER
    elif isinstance(stale_days, int) and stale_days > 5:
        color = MUTE
    else:
        color = BLUE

    date_txt = f"as of {regime_date}" if regime_date else ""
    stale = isinstance(stale_days, int) and stale_days > 5
    title = "REGIME BACKDROP ONLY" if stale else "REGIME (daily structural backdrop)"
    stale_note = (
        f'<div style="color:{AMBER};font-size:11px;margin-top:6px;{WRAP}">'
        "STALE BATCH CONTEXT: do not use this as today's live day classifier. "
        "Use Today's Live Battlefield above for the current session read.</div>"
        if stale
        else ""
    )
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">{title} {_esc(date_txt)}</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(headline)}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        f"{stale_note}"
        "</div>"
    )


def render_gap_fill_edge_block(gap_fill_edge: dict[str, Any] | None) -> str:
    """Live GAP FILL EDGE: today's gap vs historical conditional fill outcomes.

    Real signal.gap_fill_edge aggregated from auction_expectancy_events over
    ONLY pre-known conditions (gap direction + regime), so it is causal, not
    look-ahead. Color scales with historical fill rate; low-sample buckets are
    muted so a thin prior never looks like a strong edge.
    """
    ctx = gap_fill_edge or {}
    if not ctx.get("available"):
        return ""
    headline = str(ctx.get("headline") or "")
    story = str(ctx.get("story") or "")
    path_mix_text = str(ctx.get("path_mix_text") or "")
    fill_rate = ctx.get("fill_rate")
    quality = str(ctx.get("sample_quality") or "")

    if quality == "LOW_SAMPLE":
        color = MUTE
    elif isinstance(fill_rate, (int, float)) and fill_rate >= 0.70:
        color = GREEN
    elif isinstance(fill_rate, (int, float)) and fill_rate <= 0.40:
        color = RED
    else:
        color = AMBER

    path_line = ""
    if path_mix_text:
        path_line = (
            f'<div style="color:#adbac7;font-size:12px;margin-top:8px;{WRAP}">'
            f'<span style="color:{MUTE};font-weight:bold">PATH MIX:</span> '
            f"{_esc(path_mix_text)}</div>"
        )

    return (
        f'<div style="border:2px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">GAP FILL EDGE (historical, causal)</div>'
        f'<div style="color:{color};font-size:20px;font-weight:bold;margin-top:4px">{_esc(headline)}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(story)}</div>'
        f"{path_line}"
        "</div>"
    )


def render_chart(chart_svg_inline: str = "") -> str:
    """Inline the cockpit chart SVG directly into the page.

    Inlining (instead of <img src=\"cockpit_chart.svg\">) guarantees the chart
    renders regardless of how the HTML is opened -- localhost server, file://,
    or a cached page -- because the SVG travels inside the HTML itself. Falls
    back to the external file reference if no inline markup is supplied.
    """
    svg = (chart_svg_inline or "").strip()
    if svg.startswith("<svg"):
        # Make the fixed-size SVG responsive; keep viewBox for aspect ratio.
        svg = svg.replace('width="1000" height="576"', "", 1)
        svg = svg.replace(
            "<svg ",
            '<svg style="display:block;width:100%;height:auto" ',
            1,
        )
        return (
            '<div style="display:flex;gap:8px;align-items:center;margin:6px 0 4px 0">'
            '<span style="color:#8b949e;font-size:12px;font-weight:bold;letter-spacing:.08em">CHART ZOOM</span>'
            '<button onclick="sharpedgeChartZoom(-0.25)" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:4px 10px">−</button>'
            '<button onclick="sharpedgeChartZoom(0.25)" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:4px 10px">+</button>'
            '<button onclick="sharpedgeChartZoom(0,true)" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:4px 10px">Reset</button>'
            '<span id="chart-zoom-label" style="color:#8b949e;font-size:12px">100%</span>'
            "</div>"
            '<div id="chart-viewport" style="width:100%;border:1px solid #21262d;border-radius:8px;'
            f'overflow:auto;margin:6px 0"><div id="chart-scale" style="min-width:100%">{svg}</div></div>'
            "<script>"
            "(function(){"
            "function clamp(v){return Math.max(0.75,Math.min(3,Number(v)||1));}"
            "window.sharpedgeChartZoom=function(delta,reset){"
            'var scale=reset?1:clamp((Number(localStorage.getItem("sharpedgeChartZoom"))||1)+delta);'
            'localStorage.setItem("sharpedgeChartZoom",String(scale));'
            'var el=document.getElementById("chart-scale");'
            'var label=document.getElementById("chart-zoom-label");'
            'if(el){el.style.width=(scale*100)+"%";}'
            'if(label){label.textContent=Math.round(scale*100)+"%";}'
            "};"
            "window.sharpedgeChartZoom(0,false);"
            "})();"
            "</script>"
        )
    return (
        '<img src="cockpit_chart.svg" '
        'style="display:block;width:100%;border:1px solid #21262d;border-radius:8px">'
    )


def render_volume_weighted_rsi_block(pa: dict[str, Any] | None = None) -> str:
    packet = (pa or {}).get("volume_weighted_rsi") or {}
    if not packet:
        return ""
    active = bool(packet.get("active"))
    state = str(packet.get("state") or "inactive").upper()
    bias = str(packet.get("bias") or "NEUTRAL")
    score = packet.get("score", 0)
    value = packet.get("value")
    slope = packet.get("slope")
    quality = str(packet.get("volume_quality") or "unknown")
    reason = str(packet.get("reason") or "")
    color = MUTE
    if active and bias == "CALLS":
        color = GREEN
    elif active and bias == "PUTS":
        color = RED
    elif active:
        color = BLUE
    value_text = f"{float(value):.1f}" if isinstance(value, (int, float)) else "n/a"
    slope_text = f"{float(slope):+.1f}" if isinstance(slope, (int, float)) else "n/a"
    active_text = "ACTIVE" if active else "INACTIVE"
    return (
        f'<div style="border:1px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">VOLUME-WEIGHTED RSI (advisory)</div>'
        f'<div style="color:{color};font-size:18px;font-weight:bold;margin-top:4px">{_esc(active_text)} / {_esc(state)} / {score}</div>'
        f'<div style="color:#adbac7;font-size:12px;margin-top:2px">value {value_text} &middot; slope {slope_text} &middot; bias {_esc(bias)} &middot; volume {quality}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(reason)}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px">Advisory only: confirms/fades momentum quality; it does not override the execution spine.</div>'
        "</div>"
    )


def render_confluence_zones_block(confluence_zones: dict[str, Any] | None = None) -> str:
    """Compact confluence bounce/rejection zone strip for the cockpit body."""
    cz = confluence_zones or {}
    zones = cz.get("zones") or []
    if not zones:
        return ""
    reject = next((z for z in zones if z.get("side") == "resistance"), None)
    bounce = next((z for z in zones if z.get("side") == "support"), None)
    cards = []
    for zone in (reject, bounce):
        if not zone:
            continue
        edge = "#f85149" if zone.get("side") == "resistance" else "#3fb950"
        band = str(zone.get("conviction_band", ""))
        factors = " + ".join(_esc(str(f.get("name", ""))) for f in zone.get("contributing_factors", []))
        cards.append(
            f'<div style="border-left:3px solid {edge};padding:6px 10px;margin-top:6px;background:#0d1117">'
            f'<div style="font-size:12px"><b style="color:{edge}">{_esc(str(zone.get("stance", "")).upper())}</b> '
            f'<span style="color:{FG}">${zone.get("zone_lo")}–${zone.get("zone_hi")}</span> '
            f'<span style="color:#7d8590">conv {zone.get("conviction")} ({_esc(band)}) · '
            f'gate {_esc(str((zone.get("regime_gate") or {}).get("applied", "")))}</span></div>'
            f'<div style="color:#7d8590;font-size:10px;margin-top:2px;{WRAP}">{factors} · {zone.get("factor_count", 0)} stacked</div>'
            f'<div style="color:#adbac7;font-size:11px;margin-top:2px;{WRAP}">→ {_esc(str(zone.get("trigger", "")))}</div></div>'
        )
    if not cards:
        return ""
    return (
        f'<div style="margin-top:10px;padding:10px;border:1px solid #30363d;border-radius:6px;background:#0d1117">'
        f'<div style="color:{MUTE};font-size:11px;margin-bottom:2px">CONFLUENCE ZONES • stacked levels · regime-gated · advisory</div>'
        f'{"".join(cards)}</div>'
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
    decision_receipt: dict[str, Any] | None = None,
    edge_token_position: dict[str, Any] | None = None,
    regime_refinement: dict[str, Any] | None = None,
    weekly_context: dict[str, Any] | None = None,
    monthly_context: dict[str, Any] | None = None,
    stamp: str = "",
    level_states: dict[str, Any] | None = None,
    line_authority: dict[str, Any] | None = None,
    timeframe_agreement: dict[str, Any] | None = None,
    transition_pressure: dict[str, Any] | None = None,
    auction_context: dict[str, Any] | None = None,
    open_resolution: dict[str, Any] | None = None,
    dealer_positioning: dict[str, Any] | None = None,
    macro_overlay: dict[str, Any] | None = None,
    event_radar: dict[str, Any] | None = None,
    post_apple_rotation: dict[str, Any] | None = None,
    regime_read: dict[str, Any] | None = None,
    gap_fill_edge: dict[str, Any] | None = None,
    historical_refill_context: dict[str, Any] | None = None,
    chart_svg_inline: str = "",
    candle_coach: dict[str, Any] | None = None,
    confluence_zones: dict[str, Any] | None = None,
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
    page_refresh_seconds = _page_refresh_seconds()
    return f"""<!DOCTYPE html><html><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<meta http-equiv=\"refresh\" content=\"{page_refresh_seconds}\">
<meta http-equiv=\"Cache-Control\" content=\"no-store, no-cache, must-revalidate, max-age=0\">
<meta http-equiv=\"Pragma\" content=\"no-cache\">
<script>
const scrollKey = 'sharpedge.cockpit.scrollY';
window.addEventListener('load', () => {{
  const saved = sessionStorage.getItem(scrollKey);
  if (saved !== null) {{
    window.scrollTo(0, Number(saved) || 0);
  }}
}});
window.addEventListener('scroll', () => {{
  sessionStorage.setItem(scrollKey, String(window.scrollY || 0));
}}, {{ passive: true }});
setTimeout(() => {{
  sessionStorage.setItem(scrollKey, String(window.scrollY || 0));
  const next = new URL(window.location.href);
  next.searchParams.set('v', Date.now().toString());
  window.location.replace(next.toString());
}}, {page_refresh_seconds * 1000});
</script>
<title>SharpEdge Cockpit</title></head>
<body style=\"margin:0;min-height:100vh;overflow-x:auto;overflow-y:auto;-webkit-overflow-scrolling:touch;touch-action:pan-y;overscroll-behavior-y:contain;background:#0d1117;color:#e6edf3;font-family:monospace\">
<main style=\"padding:12px 12px 28px;max-width:100%;box-sizing:border-box;overflow-wrap:anywhere\">
<div style=\"display:flex;justify-content:space-between;align-items:baseline\">
<h2 style=\"margin:0;font-size:18px\">SharpEdge Live Read - SPY</h2>
<span style=\"color:#7d8590;font-size:12px\">updated {stamp} | auto {page_refresh_seconds}s</span>
</div>
<div style=\"font-size:26px;font-weight:bold;margin:6px 0\">${pa["spot"]:.2f}
<span style=\"font-size:16px;color:{day_color}\">{sign}{pa["day_chg"]:.2f}% today</span>
<span style=\"font-size:11px;color:#39c5cf\">{_esc(pa.get("spot_source") or "price")}</span></div>
{render_price_feed_lag_line(pa)}
{render_price_context_line(pa)}
{render_confluence_zones_block(confluence_zones)}
{render_chart(chart_svg_inline)}\n{render_candle_coach_block(candle_coach)}
{render_event_radar_block(event_radar)}
{render_post_apple_rotation_block(post_apple_rotation)}
{render_auction_context_block(auction_context)}
{render_open_resolution_block(open_resolution)}
{render_permission_overview(permission or {}, permission_trend or {})}
<h3 style=\"font-size:14px;color:#e6edf3;margin:14px 0 4px\">TODAY'S LIVE BATTLEFIELD + EXECUTION SPINE
(primary authority lane)</h3>
{render_permission_section(permission or {}, pa, op, micro or {}, magnitude or {}, gp or {}, setups, permission_trend or {})}
{render_under_hood_audit_block(permission=permission or {}, permission_trend=permission_trend or {}, decision_receipt=decision_receipt or {}, transition_pressure=transition_pressure, auction_context=auction_context, open_resolution=open_resolution, dealer_positioning=dealer_positioning, regime_read=regime_read, gap_fill_edge=gap_fill_edge)}
{render_dealer_positioning_block(dealer_positioning)}
{render_macro_overlay_block(macro_overlay)}
{render_regime_read_block(regime_read)}
{render_gap_fill_edge_block(gap_fill_edge)}
{render_historical_refill_context_block(historical_refill_context)}
{active_setup_level_badge(setups)}
{render_transition_pressure_block(transition_pressure)}
{render_volume_weighted_rsi_block(pa)}
{render_timeframe_agreement_block(timeframe_agreement)}
{render_level_state_block(level_states or {})}
{render_line_authority_block(line_authority or {})}
{weekly_context_section(weekly_context)}
{monthly_context_section(monthly_context)}
<h3 style=\"font-size:14px;color:#7d8590;margin:14px 0 4px\">SETUPS / DAY-TYPE TEXTURE
(failed breaks + exhaustion + compression; day-type is evidence under the inherited auction above)</h3>
{setup_section(setups)}
<h3 style=\"font-size:14px;color:#7d8590;margin:14px 0 4px\">THE READ
(context)</h3>
{"".join(cards)}
<p style=\"color:#484f58;font-size:11px;margin-top:14px\">Free data (Yahoo 1m + CBOE delayed options). Decision support only - you own every trade.</p>
</main></body></html>"""


__all__ = [
    "infer_target",
    "reachability_context",
    "render_candle_coach_block",
    "render_execution_expansion_potential_block",
    "render_execution_state_packets_block",
    "render_execution_vector_interactions_block",
    "render_live_read_html",
    "render_location_strip",
    "render_permission_overview",
    "render_permission_score_trend",
    "render_permission_section",
    "render_regime_refinement_block",
    "render_setup_conviction_block",
    "render_structure_state_block",
    "render_under_hood_audit_block",
    "render_volume_weighted_rsi_block",
    "summarize_permission_scores",
]
