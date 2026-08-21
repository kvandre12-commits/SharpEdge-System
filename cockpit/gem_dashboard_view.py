"""Gem-first dashboard for SharpEdge.

This page centers the product on the user's real gem:
permission score, permission trend, exhaustion markers, traverse targets,
and fair value gaps. Secondary machinery is intentionally demoted.
"""

from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#8b949e"
BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_price(value: Any) -> str:
    return f"${float(value):.2f}" if isinstance(value, (int, float)) else "n/a"


def _score_color(score: Any) -> str:
    try:
        value = int(score)
    except (TypeError, ValueError):
        return MUTE
    if value >= 75:
        return GREEN
    if value >= 60:
        return BLUE
    if value >= 45:
        return AMBER
    return RED


def _bias_color(bias: str) -> str:
    label = str(bias or "").upper()
    if label == "CALLS":
        return GREEN
    if label == "PUTS":
        return RED
    return MUTE


def _chip(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:4px 8px;border:1px solid {color};'
        f'border-radius:999px;color:{color};font-size:11px;font-weight:bold;margin:2px 6px 2px 0">{_esc(text)}</span>'
    )


def _card(title: str, body: str, *, accent: str = BORDER) -> str:
    return (
        f'<div style="background:{SURFACE};border:1px solid {accent};border-radius:14px;padding:14px">'
        f'<div style="color:{MUTE};font-size:11px;font-weight:bold;letter-spacing:.08em">{_esc(title)}</div>'
        f'<div style="margin-top:8px">{body}</div></div>'
    )


def _fvg_label(gap: dict[str, Any]) -> str:
    if not gap:
        return "no gap"
    direction = str(gap.get("direction") or "gap").upper()
    low = gap.get("gap_low")
    high = gap.get("gap_high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return f"{direction} FVG {low:.2f}-{high:.2f}"
    return f"{direction} FVG"


def _permission_trend_svg(permission_trend: dict[str, Any]) -> str:
    points = permission_trend.get("points") or []
    if not points:
        return f'<div style="color:{MUTE};font-size:12px">No permission trend history yet.</div>'

    values = [int(point.get("score") or 0) for point in points]
    width, height, pad = 420, 120, 12
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1)

    def x(index: int) -> float:
        return pad + index / max(len(values) - 1, 1) * (width - pad * 2)

    def y(value: int) -> float:
        return pad + (1 - ((value - min_v) / span)) * (height - pad * 2)

    poly = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" fill="{_score_color(value)}"/>'
        f'<text x="{x(i):.1f}" y="{height - 6}" fill="{MUTE}" font-size="10" text-anchor="middle">{_esc(points[i].get("time") or "?")}</text>'
        for i, value in enumerate(values)
    )
    latest_markers = points[-1].get("event_markers") or []
    direction = str(permission_trend.get("direction") or "new").upper()
    delta = permission_trend.get("delta")
    delta_text = f"{int(delta):+d}" if isinstance(delta, int) else "n/a"
    marker_html = "".join(_chip(marker, CYAN) for marker in latest_markers[:4])
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">'
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#0d1117;border:1px solid {BORDER};border-radius:10px">'
        f'<polyline points="{poly}" fill="none" stroke="{CYAN}" stroke-width="3"/>{dots}</svg>'
        f'<div style="flex:1;min-width:220px">'
        f'<div style="color:{FG};font-size:15px;font-weight:bold">{_esc(direction)} • {delta_text}</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:6px">Setup and exhaustion events moving the permission read.</div>'
        f'<div style="margin-top:10px">{marker_html or _chip("no fresh marker", MUTE)}</div>'
        "</div></div>"
    )


def _calls_puts_lane(signal: dict[str, Any], side: str) -> dict[str, str]:
    permission = signal.get("trade_permission") or {}
    trend = signal.get("permission_score_trend") or {}
    target = signal.get("target_plan") or {}
    setup_conviction = permission.get("setup_conviction") or {}
    active = str(permission.get("bias") or "").upper() == side
    score = int(permission.get("trade_permission_score") or 0)
    delta = trend.get("delta")
    delta_text = f"{int(delta):+d}" if isinstance(delta, int) else "n/a"
    target_text = _fmt_price(target.get("price")) if active else "wait"
    objective = (
        str(target.get("objective") or "watch")
        if active
        else "opposite side not in control"
    )
    headline = f"{'ACTIVE' if active else 'STANDBY'} • {score}/100"
    detail = (
        f"trend {delta_text} • {objective} • target {target_text}"
        if active
        else f"trend {delta_text} • wait for exhaustion to flip permission"
    )
    note = str(
        setup_conviction.get("reason") or signal.get("entry_setup_tag") or "watch"
    )
    return {
        "title": f"{side} LANE",
        "headline": headline,
        "detail": detail,
        "note": note,
        "accent": _bias_color(side) if active else BORDER,
    }


def _master_state(signal: dict[str, Any]) -> dict[str, str]:
    permission = signal.get("trade_permission") or {}
    setup_conviction = permission.get("setup_conviction") or {}
    entry_gate = signal.get("entry_gate") or {}
    trend = signal.get("permission_score_trend") or {}
    score = int(permission.get("trade_permission_score") or 0)
    setup_gate = str(setup_conviction.get("setup_gate") or "WATCH").upper()
    trade_gate = str(permission.get("trade_gate") or "WATCH").upper()
    delta = trend.get("delta")
    has_trigger = bool(entry_gate.get("actionable")) and isinstance(
        entry_gate.get("trigger_price"), (int, float)
    )
    if (
        trade_gate in {"DENY", "BLOCK", "STAND_DOWN"}
        or setup_gate != "ACTIONABLE"
        or score < 60
    ):
        return {
            "state": "DO NOTHING",
            "detail": f"{setup_gate} setup / {trade_gate} gate / {score}/100 permission.",
            "note": "Context can be true without being tradable. Save ammo for the screamers.",
            "accent": RED if trade_gate in {"DENY", "BLOCK", "STAND_DOWN"} else AMBER,
        }
    if not has_trigger or score < 70 or (isinstance(delta, int) and delta < 0):
        detail = f"{trade_gate} thesis is alive, but the entry is not fully armed yet."
        if isinstance(delta, int) and delta < 0:
            detail = f"Permission is weakening ({delta:+d}) — don’t force it."
        return {
            "state": "WAIT",
            "detail": detail,
            "note": "Let price come to the trigger zone. No trigger, no trade.",
            "accent": BLUE if score >= 70 else AMBER,
        }
    trigger = _fmt_price(entry_gate.get("trigger_price"))
    fail = _fmt_price(entry_gate.get("level_price"))
    return {
        "state": "LIVE",
        "detail": f"Trigger armed at {trigger}; fail level {fail}; permission {score}/100.",
        "note": "This is where a small number of real trades should come from.",
        "accent": GREEN,
    }


def _master_state_card(signal: dict[str, Any]) -> str:
    state = _master_state(signal)
    body = (
        f'<div style="color:{state["accent"]};font-size:34px;font-weight:900;letter-spacing:.04em">{_esc(state["state"])}</div>'
        f'<div style="color:{FG};font-size:13px;margin-top:8px">{_esc(state["detail"])}</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:8px">{_esc(state["note"])}</div>'
    )
    return _card("MASTER STATE", body, accent=state["accent"])


def _screamer_filter_card(signal: dict[str, Any]) -> dict[str, str]:
    state = _master_state(signal)
    if state["state"] == "DO NOTHING":
        return {
            "title": "SCREAMER FILTER",
            "headline": "NOT A SCREAMER",
            "detail": state["detail"],
            "note": "Wait for a truly actionable trigger before taking risk.",
            "accent": AMBER,
        }
    if state["state"] == "WAIT":
        return {
            "title": "SCREAMER FILTER",
            "headline": "WAIT FOR TRIGGER",
            "detail": state["detail"],
            "note": "No trigger, no trade. Revolutionary concept.",
            "accent": BLUE,
        }
    return {
        "title": "SCREAMER FILTER",
        "headline": "LIVE SCREAMER",
        "detail": state["detail"],
        "note": state["note"],
        "accent": GREEN,
    }


def _execution_plan_card(signal: dict[str, Any]) -> dict[str, str]:
    entry_gate = signal.get("entry_gate") or {}
    target = signal.get("target_plan") or {}
    trigger_price = entry_gate.get("trigger_price")
    fail_price = entry_gate.get("level_price")
    reachable = target.get("reachable_today") or {}
    exit_price = reachable.get("price") or target.get("price")
    bars_ago = entry_gate.get("bars_ago")
    headline_parts = []
    if isinstance(trigger_price, (int, float)):
        headline_parts.append(f"entry {_fmt_price(trigger_price)}")
    if isinstance(exit_price, (int, float)):
        headline_parts.append(f"exit {_fmt_price(exit_price)}")
    headline = " • ".join(headline_parts) or "wait"
    detail_parts = []
    if isinstance(fail_price, (int, float)):
        level_name = entry_gate.get("level_name") or "fail level"
        detail_parts.append(f"kill-switch {_esc(level_name)} {_fmt_price(fail_price)}")
    if target.get("likely_travel"):
        detail_parts.append(str(target.get("likely_travel")))
    detail = " • ".join(detail_parts) or "No armed entry/exit yet. Good. Less nonsense."
    note_parts = [str(target.get("reason") or signal.get("entry_setup_tag") or "")]
    if isinstance(bars_ago, int):
        note_parts.append(f"trigger candle {bars_ago} bars ago")
    note = " • ".join(part for part in note_parts if part)
    return {
        "title": "EXECUTION PLAN",
        "headline": headline.upper(),
        "detail": detail,
        "note": note,
        "accent": PURPLE,
    }


def _exhaustion_card(signal: dict[str, Any]) -> dict[str, str]:
    permission = signal.get("trade_permission") or {}
    decision_receipt = signal.get("decision_receipt") or {}
    primary_event = decision_receipt.get("primary_setup_event") or {}
    exhaustion = (permission.get("scores") or {}).get("exhaustion_score") or {}
    detail = str(exhaustion.get("reason") or "no exhaustion read")
    if primary_event.get("event_type"):
        level = primary_event.get("level") or {}
        suffix = f" @ {level.get('name')}" if level.get("name") else ""
        detail = f"{primary_event.get('event_type')} {str(primary_event.get('status') or '').upper()}{suffix}"
    return {
        "title": "EXHAUSTION MARKER",
        "headline": f"score {exhaustion.get('score', 'n/a')}",
        "detail": detail,
        "note": str(signal.get("entry_setup_tag") or "watch"),
        "accent": _score_color(exhaustion.get("score")),
    }


def _exit_card(signal: dict[str, Any]) -> dict[str, str]:
    permission = signal.get("trade_permission") or {}
    trend = signal.get("permission_score_trend") or {}
    decision_receipt = signal.get("decision_receipt") or {}
    transitions = decision_receipt.get("setup_event_transitions") or []
    latest_transition = transitions[-1] if transitions else {}
    nearest_gap = (signal.get("fair_value_gaps") or {}).get("nearest_open_gap") or {}
    detail = "permission still alive"
    if latest_transition and str(latest_transition.get("status") or "").lower() in {
        "invalidated",
        "expired",
    }:
        detail = (
            f"{latest_transition.get('event_type')} {latest_transition.get('status')}"
        )
    elif isinstance(trend.get("delta"), int) and int(trend.get("delta")) < 0:
        detail = "permission trend weakening — protect convexity"
    elif nearest_gap:
        detail = f"watch {_fvg_label(nearest_gap)} fill {nearest_gap.get('fill_state')}"
    return {
        "title": "EXIT / INVALIDATION WATCH",
        "headline": str(permission.get("trade_gate") or "WATCH"),
        "detail": detail,
        "note": str((signal.get("target_plan") or {}).get("likely_travel") or ""),
        "accent": AMBER,
    }


def _target_fvg_card(signal: dict[str, Any]) -> dict[str, str]:
    target = signal.get("target_plan") or {}
    fvg = signal.get("fair_value_gaps") or {}
    nearest = fvg.get("nearest_open_gap") or {}
    detail = str(target.get("reason") or "")
    if target.get("price") is not None:
        detail = f"{target.get('label')} {_fmt_price(target.get('price'))} • {detail}"
    note = f"nearest gap: {_fvg_label(nearest)}" if nearest else "no nearby open gap"
    return {
        "title": "TRAVERSE TARGET",
        "headline": str(target.get("objective") or "watch"),
        "detail": detail,
        "note": note,
        "accent": CYAN,
    }


def _cards_grid(signal: dict[str, Any]) -> str:
    cards = [
        _calls_puts_lane(signal, "CALLS"),
        _calls_puts_lane(signal, "PUTS"),
        _screamer_filter_card(signal),
        _execution_plan_card(signal),
        _exhaustion_card(signal),
        _exit_card(signal),
        _target_fvg_card(signal),
    ]
    return "".join(
        _card(
            card["title"],
            f'<div style="color:{card["accent"]};font-size:18px;font-weight:bold">{_esc(card["headline"])}</div>'
            f'<div style="color:{FG};font-size:12px;margin-top:8px">{_esc(card["detail"])}</div>'
            f'<div style="color:{MUTE};font-size:11px;margin-top:8px">{_esc(card["note"])}</div>',
            accent=card["accent"],
        )
        for card in cards
    )


def _fvg_section(fvg: dict[str, Any]) -> str:
    if not fvg:
        return f'<div style="color:{MUTE};font-size:12px">No fair value gap read yet.</div>'
    sections = []
    for label, gap in (
        ("nearest", fvg.get("nearest_open_gap") or {}),
        ("above", fvg.get("nearest_open_gap_above") or {}),
        ("below", fvg.get("nearest_open_gap_below") or {}),
    ):
        if not gap:
            continue
        color = GREEN if str(gap.get("direction")) == "bullish" else RED
        sections.append(
            f'<div style="padding:10px 0;border-top:1px solid {BORDER}">'
            f'<div style="color:{color};font-weight:bold">{_esc(label.upper())}: {_esc(_fvg_label(gap))}</div>'
            f'<div style="color:{FG};font-size:12px;margin-top:4px">fill {gap.get("fill_state")} {gap.get("fill_pct")}% • {gap.get("age_bars")} bars old • {gap.get("position_vs_spot")} spot</div>'
            f'<div style="color:{MUTE};font-size:11px;margin-top:4px">distance from spot: {_esc(gap.get("distance_from_spot"))} • fill comes {gap.get("fill_direction")}</div>'
            "</div>"
        )
    return (
        "".join(sections)
        or f'<div style="color:{MUTE};font-size:12px">Recent gaps are already filled. Cute.</div>'
    )


def _score_ladder(permission: dict[str, Any]) -> str:
    rows = []
    for name in (
        "structure_score",
        "acceptance_score",
        "trend_score",
        "volume_score",
        "exhaustion_score",
        "trap_score",
        "dealer_gamma_score",
        "regime_score",
    ):
        item = (permission.get("scores") or {}).get(name) or {}
        if not item:
            continue
        rows.append(
            f'<div style="display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-top:1px solid {BORDER}">'
            f'<span style="color:{MUTE};font-size:12px">{_esc(name.replace("_", " "))}</span>'
            f'<span style="color:{_score_color(item.get("score"))};font-weight:bold">{_esc(item.get("score"))}</span>'
            "</div>"
        )
    return (
        "".join(rows)
        or f'<div style="color:{MUTE};font-size:12px">No score ladder.</div>'
    )


def _hero(signal: dict[str, Any]) -> str:
    permission = signal.get("trade_permission") or {}
    trend = signal.get("permission_score_trend") or {}
    target = signal.get("target_plan") or {}
    spot = signal.get("spot")
    score = permission.get("trade_permission_score")
    bias = str(permission.get("bias") or "NEUTRAL")
    direction = str(trend.get("direction") or "new").upper()
    delta = trend.get("delta")
    delta_text = f"{int(delta):+d}" if isinstance(delta, int) else "n/a"
    target_text = (
        f"{target.get('label')} {_fmt_price(target.get('price'))}"
        if target.get("label")
        else "No target"
    )
    left_body = (
        f'<div style="color:{FG};font-size:40px;font-weight:900">SPY {_fmt_price(spot)}</div>'
        f'<div style="margin-top:8px;color:{_score_color(score)};font-size:58px;font-weight:900;line-height:1">{_esc(score)}</div>'
        f'<div style="margin-top:8px;color:{_bias_color(bias)};font-size:18px;font-weight:bold">{_esc(bias)} • {_esc(permission.get("trade_gate") or "WATCH")}</div>'
        f'<div style="margin-top:10px;color:{MUTE};font-size:13px">permission trend {direction} ({delta_text}) • target {_esc(target_text)}</div>'
    )
    right_body = (
        f'<div style="color:{FG};font-size:16px;font-weight:bold">{_esc(signal.get("gamma_regime") or "unknown")} gamma</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:6px">VWAP {signal.get("vs_vwap", "n/a")}% • rng {signal.get("rng_pos", "n/a")}% • vol {signal.get("vol_mult", "n/a")}x</div>'
        f'<div style="margin-top:10px">{_chip(str(signal.get("entry_setup_tag") or "no setup"), CYAN)}{_chip(str(target.get("objective") or "watch"), PURPLE)}</div>'
    )
    return (
        '<div style="display:grid;grid-template-columns:1.3fr .7fr;gap:16px">'
        + _card("GEM", left_body, accent=_score_color(score))
        + _card("MARKET FRAME", right_body, accent=BORDER)
        + "</div>"
    )


def render_gem_dashboard_html(signal: dict[str, Any], stamp: str) -> str:
    permission = signal.get("trade_permission") or {}
    trend = signal.get("permission_score_trend") or {}
    chart_html = (
        '<img src="gem_chart.svg" alt="SharpEdge gem chart" '
        'style="width:100%;display:block;border:1px solid #30363d;border-radius:14px;background:#0d1117"/>'
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SharpEdge Gem Dashboard</title>
  <meta http-equiv=\"refresh\" content=\"2\" />
  <meta http-equiv=\"Cache-Control\" content=\"no-store, no-cache, must-revalidate, max-age=0\" />
  <meta http-equiv=\"Pragma\" content=\"no-cache\" />
  <style>
    body {{ background:{BG}; color:{FG}; font-family:Inter,system-ui,-apple-system,sans-serif; margin:0; }}
    .wrap {{ max-width:1400px; margin:0 auto; padding:20px; }}
    .grid {{ display:grid; grid-template-columns:1.15fr .85fr; gap:16px; }}
    .stack {{ display:grid; gap:16px; }}
    .markers {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    @media (max-width: 1180px) {{ .markers {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width: 980px) {{ .grid, .markers {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div style=\"display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:14px\">
      <div>
        <div style=\"color:{PURPLE};font-size:13px;font-weight:bold;letter-spacing:.12em\">SHARPEDGE • GEM FIRST</div>
        <div style=\"color:{MUTE};font-size:13px;margin-top:4px\">Permission score, permission trend, exhaustion, traverse, invalidation. The rest can sit down.</div>
      </div>
      <div style=\"color:{MUTE};font-size:12px\">Updated {_esc(stamp)}</div>
    </div>
    <div class=\"stack\">
      {_hero(signal)}
      {_master_state_card(signal)}
      <div class=\"grid\">
        {_card("GEM GRAPH", chart_html, accent=BLUE)}
        {_card("PERMISSION TREND", _permission_trend_svg(trend), accent=CYAN)}
      </div>
      <div class=\"markers\">{_cards_grid(signal)}</div>
      <div class=\"grid\">
        {_card("FAIR VALUE GAP MAP", _fvg_section(signal.get("fair_value_gaps") or {}), accent=PURPLE)}
        {_card("SCORE LADDER", _score_ladder(permission), accent=BORDER)}
      </div>
    </div>
  </div>
</body>
</html>
"""


__all__ = ["render_gem_dashboard_html"]
