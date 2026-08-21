"""HTML renderer for the educational Candle Coach cockpit block."""

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
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _bias_color(value: Any) -> str:
    text = str(value or "").lower()
    if "bull" in text or "demand" in text:
        return GREEN
    if "bear" in text or "supply" in text:
        return RED
    if "compression" in text or "pause" in text or "indecision" in text:
        return AMBER
    return BLUE


def _candle_visual(candles: list[dict[str, Any]] | None) -> str:
    bars = [c for c in candles or [] if isinstance(c, dict)]
    if not bars:
        return ""
    highs = [float(c.get("high", 0)) for c in bars]
    lows = [float(c.get("low", 0)) for c in bars]
    lo = min(lows)
    hi = max(highs)
    span = max(hi - lo, 1e-9)
    width = 230
    height = 92
    pad_x = 18
    pad_y = 10
    gap = (width - pad_x * 2) / max(len(bars), 1)

    def y(price: float) -> float:
        return pad_y + (1 - (price - lo) / span) * (height - pad_y * 2)

    parts = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" style="display:block;margin:8px 0;background:#090d13;border:1px solid #21262d;border-radius:8px">',
        f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" stroke="#21262d"/>',
    ]
    for idx, candle in enumerate(bars):
        cx = pad_x + gap * idx + gap / 2
        open_y = y(float(candle.get("open", 0)))
        close_y = y(float(candle.get("close", 0)))
        high_y = y(float(candle.get("high", 0)))
        low_y = y(float(candle.get("low", 0)))
        body_top = min(open_y, close_y)
        body_h = max(abs(close_y - open_y), 3)
        direction = str(candle.get("direction") or "flat")
        color = GREEN if direction == "bull" else RED if direction == "bear" else AMBER
        body_w = min(22, max(13, gap * 0.42))
        parts.extend(
            [
                f'<line x1="{cx:.1f}" y1="{high_y:.1f}" x2="{cx:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="2"/>',
                f'<rect x="{cx - body_w / 2:.1f}" y="{body_top:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" rx="2" fill="{color}" opacity="0.82"/>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _framework_block(framework: dict[str, Any]) -> str:
    if not framework:
        return ""
    gates = framework.get("gates") or []
    items = "".join(
        f'<li style="margin:4px 0;{WRAP}"><b style="color:{CYAN}">{_esc(gate.get("label", "gate"))}</b> '
        f'<span style="color:{AMBER}">[{_esc(gate.get("status", "unknown"))}]</span> '
        f'<span style="color:{MUTE}">{_esc(gate.get("message", ""))}</span></li>'
        for gate in gates
        if isinstance(gate, dict)
    )
    return (
        f'<div style="border:1px solid #30363d;background:#0d1117;padding:10px;border-radius:8px;margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap">'
        f'<div style="color:{PURPLE};font-size:12px;font-weight:bold;letter-spacing:.08em">CONDITIONAL TRADEABILITY GATES</div>'
        f'<div style="color:{AMBER};font-size:12px;font-weight:bold">Output: {_esc(framework.get("output", "Watch"))}</div></div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:5px">{_esc(framework.get("expected_value_formula", ""))}</div>'
        f'<ol style="margin:8px 0 0 20px;padding:0;color:{FG};font-size:12px">{items}</ol>'
        "</div>"
    )


def _auction_box_block(box: dict[str, Any]) -> str:
    if not box:
        return ""
    premise = box.get("premise") or {}
    location = box.get("location") or {}
    acceptance = box.get("acceptance") or {}
    participation = box.get("participation") or {}
    permission = box.get("permission_context") or {}
    options_proxy = box.get("options_flow_proxy") or {}
    flow = options_proxy.get("flow_pressure") or {}
    spreads = options_proxy.get("spread_proxy") or {}
    source = options_proxy.get("source") or {}
    freshness = options_proxy.get("freshness") or {}
    stale_badge = (
        f'<span style="color:{RED};font-weight:bold">STALE</span> '
        if options_proxy.get("stale")
        else ""
    )
    options_html = ""
    if options_proxy:
        options_html = (
            f'<div style="border:1px solid #30363d;background:#0d1117;padding:8px;border-radius:7px;margin-top:8px">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap">'
            f'<div style="color:{AMBER};font-size:11px;font-weight:bold;letter-spacing:.08em">CBOE DELAYED OPTIONS PROXY</div>'
            f'<div style="color:{MUTE};font-size:10px">{stale_badge}{_esc(options_proxy.get("authority", "proxy"))}</div></div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:6px;color:{FG};font-size:11px">'
            f'<div>flow<br><span style="color:{CYAN}">{_esc(flow.get("state") or "unknown")}</span></div>'
            f'<div>PCVR<br><span style="color:{CYAN}">{_esc(flow.get("pcvr") or "n/a")}</span></div>'
            f'<div>call spread<br><span style="color:{CYAN}">{_esc(spreads.get("call_quality") or "unknown")}</span></div>'
            f'<div>put spread<br><span style="color:{CYAN}">{_esc(spreads.get("put_quality") or "unknown")}</span></div>'
            f"</div>"
            f'<div style="color:{MUTE};font-size:10px;margin-top:6px;{WRAP}">{_esc(options_proxy.get("summary", ""))}</div>'
            f'<div style="color:{RED};font-size:10px;margin-top:4px;{WRAP}">{_esc(freshness.get("reason", ""))}</div>'
            f'<div style="color:{RED};font-size:10px;margin-top:4px;{WRAP}">{_esc(source.get("delay_note", ""))}</div>'
            "</div>"
        )
    facts = [item for item in (box.get("facts") or []) if isinstance(item, dict)]
    event_stack = " / ".join(str(item) for item in premise.get("event_stack") or [])
    compact_rows = "".join(
        f'<div style="display:grid;grid-template-columns:130px 92px 1fr;gap:8px;padding:5px 0;border-top:1px solid #21262d">'
        f'<div style="color:{FG};font-size:11px;font-weight:bold">{_esc(item.get("label", "fact"))}</div>'
        f'<div style="color:{AMBER};font-size:11px">{_esc(item.get("status", "unknown"))}</div>'
        f'<div style="font-size:11px;{WRAP}"><span style="color:{BLUE}">{_esc(item.get("value") if item.get("value") is not None else "")}</span><br><span style="color:{MUTE}">{_esc(item.get("read", ""))}</span></div>'
        f"</div>"
        for item in facts
    )
    missing = ", ".join(str(item) for item in box.get("missing_microstructure") or [])
    return (
        f'<div style="border:1px solid {CYAN};background:#09131a;padding:11px;border-radius:8px;margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap">'
        f'<div style="color:{CYAN};font-size:12px;font-weight:bold;letter-spacing:.08em">AUCTION EXECUTION BOX</div>'
        f'<div style="color:{MUTE};font-size:11px">human-in-the-loop context &middot; not permission</div></div>'
        f'<div style="color:{FG};font-size:13px;font-weight:bold;margin-top:6px;{WRAP}">'
        f"Premise: {_esc(event_stack or premise.get('candle_context', 'candle event'))}</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;margin-top:8px">'
        f'<div><span style="color:{MUTE};font-size:10px">LOCATION</span><br><span style="color:{BLUE};font-size:12px">{_esc(location.get("state") or "unknown")}</span><div style="color:{MUTE};font-size:11px;{WRAP}">{_esc(location.get("reason") or "")}</div></div>'
        f'<div><span style="color:{MUTE};font-size:10px">ACCEPTANCE</span><br><span style="color:{AMBER};font-size:12px">{_esc(acceptance.get("state") or "unknown")}</span><div style="color:{MUTE};font-size:11px;{WRAP}">{_esc(acceptance.get("reason") or "")}</div></div>'
        f'<div><span style="color:{MUTE};font-size:10px">PARTICIPATION</span><br><span style="color:{PURPLE};font-size:12px">{_esc(participation.get("state") or "unknown")}</span><div style="color:{MUTE};font-size:11px;{WRAP}">{_esc(participation.get("reason") or "")}</div></div>'
        f'<div><span style="color:{MUTE};font-size:10px">SPINE CONTEXT</span><br><span style="color:{FG};font-size:12px">{_esc(permission.get("gate") or "UNKNOWN")} {_esc(permission.get("score") or "")}</span><div style="color:{MUTE};font-size:11px">bias {_esc(permission.get("bias") or "NEUTRAL")}</div></div>'
        f"</div>{options_html}"
        f'<div style="margin-top:8px">{compact_rows}</div>'
        f'<div style="color:{RED};font-size:11px;margin-top:8px;{WRAP}"><b>Missing tape/depth:</b> {_esc(missing)}</div>'
        f'<div style="color:{CYAN};font-size:11px;margin-top:8px;{WRAP}">{_esc(box.get("doctrine", ""))}</div>'
        "</div>"
    )


def _pattern_card(pattern: dict[str, Any], title: str) -> str:
    if not pattern:
        return ""
    color = _bias_color(pattern.get("bias_hint"))
    anatomy = pattern.get("anatomy") or {}
    anatomy_html = ""
    if anatomy:
        anatomy_html = (
            f'<div style="color:{MUTE};font-size:11px;margin-top:6px">'
            f"body {float(anatomy.get('body_pct', 0)) * 100:.0f}% &middot; "
            f"upper wick {float(anatomy.get('upper_wick_pct', 0)) * 100:.0f}% &middot; "
            f"lower wick {float(anatomy.get('lower_wick_pct', 0)) * 100:.0f}% &middot; "
            f"{_esc(anatomy.get('direction', ''))}</div>"
        )
    return (
        f'<div style="border:1px solid #30363d;background:#0d1117;padding:10px;border-radius:8px;min-width:240px;flex:1">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
        f'<div style="color:{MUTE};font-size:11px;font-weight:bold;letter-spacing:.08em">{_esc(title)}</div>'
        f'<div style="color:{MUTE};font-size:10px">{_esc(pattern.get("clock", ""))}</div></div>'
        f'<div style="color:{color};font-size:17px;font-weight:bold;margin-top:4px">{_esc(pattern.get("name", "unknown"))}</div>'
        f"{_candle_visual(pattern.get('candles') or [])}"
        f'<div style="color:{FG};font-size:12px;margin-top:4px;{WRAP}">{_esc(pattern.get("meaning", ""))}</div>'
        f'<div style="color:{CYAN};font-size:12px;margin-top:6px;{WRAP}"><b>Watch next:</b> {_esc(pattern.get("watch_next", ""))}</div>'
        f"{anatomy_html}"
        "</div>"
    )


def _rate(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _expectancy_block(expectancy: dict[str, Any]) -> str:
    if not expectancy:
        return ""
    ctx = expectancy.get("live_context") or {}
    row = expectancy.get("match") or {}
    if not expectancy.get("available"):
        return (
            f'<div style="border:1px solid #30363d;background:#0d1117;padding:9px;border-radius:8px;margin-top:10px">'
            f'<div style="color:{AMBER};font-size:12px;font-weight:bold;letter-spacing:.08em">HISTORICAL CANDLE EXPECTANCY</div>'
            f'<div style="color:{MUTE};font-size:11px;margin-top:5px;{WRAP}">'
            f"No matrix row attached. status={_esc(expectancy.get('status', 'unknown'))}; "
            f"live event={_esc(ctx.get('event_name', 'unmapped'))} / {_esc(ctx.get('event_direction', 'n/a'))}. "
            f"This remains education-only and cannot supply empirical EV.</div></div>"
        )
    ready = "yes" if row.get("deployment_ready") else "no"
    avg_r = row.get("avg_realized_R")
    try:
        avg_r_text = f"{float(avg_r):+.2f}R"
    except (TypeError, ValueError):
        avg_r_text = "n/a"
    notes = row.get("confidence_notes") or ""
    return (
        f'<div style="border:1px solid {CYAN};background:#071419;padding:10px;border-radius:8px;margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:baseline">'
        f'<div style="color:{CYAN};font-size:12px;font-weight:bold;letter-spacing:.08em">HISTORICAL CANDLE EXPECTANCY</div>'
        f'<div style="color:{MUTE};font-size:10px">education only &middot; no permission override</div></div>'
        f'<div style="color:{FG};font-size:12px;margin-top:6px;{WRAP}">'
        f"{_esc(ctx.get('event_name'))} / {_esc(ctx.get('event_direction'))} matched {_esc(expectancy.get('match_tier'))}</div>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:8px;margin-top:7px;color:{FG};font-size:11px">'
        f'<div>sample<br><span style="color:{CYAN}">n={_esc(row.get("n", "n/a"))}</span></div>'
        f'<div>confidence<br><span style="color:{CYAN}">{_esc(row.get("confidence_label", "n/a"))} {_esc(row.get("confidence_score", ""))}</span></div>'
        f'<div>avg R<br><span style="color:{CYAN}">{_esc(avg_r_text)}</span></div>'
        f'<div>target first<br><span style="color:{CYAN}">{_rate(row.get("target_before_stop_rate"))}</span></div>'
        f'<div>stop first<br><span style="color:{CYAN}">{_rate(row.get("stop_before_target_rate"))}</span></div>'
        f'<div>deployment<br><span style="color:{AMBER}">{_esc(row.get("deployment_tier", "n/a"))} ready={ready}</span></div>'
        f'</div><div style="color:{MUTE};font-size:11px;margin-top:7px;{WRAP}">{_esc(expectancy.get("interpretation", ""))}</div>'
        f'<div style="color:{AMBER};font-size:11px;margin-top:5px;{WRAP}">{_esc(notes)}</div>'
        "</div>"
    )


def _candle_vector_lesson_block(lesson: dict[str, Any]) -> str:
    if not lesson:
        return ""
    graph = lesson.get("graph_bridge") or {}
    rows = [row for row in lesson.get("vector_rows") or [] if isinstance(row, dict)]
    row_html = "".join(
        f'<div style="display:grid;grid-template-columns:125px 70px 1fr;gap:8px;padding:7px 0;border-top:1px solid #21262d">'
        f'<div><span style="color:{CYAN};font-size:11px;font-weight:bold">{_esc(row.get("part", "vector"))}</span><br>'
        f'<span style="color:{MUTE};font-size:10px">{_esc(row.get("correlation_family", ""))}</span></div>'
        f'<div><span style="color:{_bias_color(row.get("bias"))};font-size:11px;font-weight:bold">{_esc(row.get("bias", "NEUTRAL"))}</span><br>'
        f'<span style="color:{AMBER};font-size:10px">{_esc(row.get("score", "n/a"))}</span></div>'
        f'<div style="font-size:11px;{WRAP}"><span style="color:{FG}">{_esc(row.get("teaching_question", ""))}</span><br>'
        f'<span style="color:{MUTE}">{_esc(row.get("reason", ""))}</span><br>'
        f'<span style="color:{PURPLE}">graph relation: {_esc(row.get("graph_relation", "unknown"))}</span></div>'
        "</div>"
        for row in rows
    )
    patterns = " / ".join(str(item) for item in lesson.get("pattern_stack") or [])
    return (
        f'<div style="border:1px solid {PURPLE};background:#100d18;padding:10px;border-radius:8px;margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:baseline">'
        f'<div style="color:{PURPLE};font-size:12px;font-weight:bold;letter-spacing:.08em">CANDLE → VECTOR → GRAPH LESSON</div>'
        f'<div style="color:{MUTE};font-size:10px">education only &middot; graph canon aware</div></div>'
        f'<div style="color:{FG};font-size:12px;margin-top:6px;{WRAP}"><b>Candle stack:</b> {_esc(patterns or "no candle stack")}</div>'
        f'<div style="color:{CYAN};font-size:12px;margin-top:5px;{WRAP}">{_esc(lesson.get("pattern_lesson", ""))}</div>'
        f'<div style="border:1px solid #30363d;background:#090d13;padding:8px;border-radius:7px;margin-top:8px;color:{MUTE};font-size:11px;{WRAP}">'
        f'<b style="color:{BLUE}">Graph teaches:</b> {_esc(graph.get("graph_bias", "NEUTRAL"))} — {_esc(graph.get("graph_reason", ""))}<br>'
        f'<span style="color:{MUTE}">{_esc(graph.get("teaching", ""))}</span></div>'
        f'<div style="margin-top:6px">{row_html}</div>'
        f'<div style="color:{AMBER};font-size:11px;margin-top:7px;{WRAP}">{_esc(lesson.get("doctrine", ""))}</div>'
        "</div>"
    )


def _encyclopedia_block(encyclopedia: dict[str, Any]) -> str:
    entries = [
        item for item in encyclopedia.get("entries") or [] if isinstance(item, dict)
    ]
    if not entries:
        return ""
    cards = "".join(
        f'<div style="border:1px solid #30363d;background:#090d13;padding:9px;border-radius:8px;margin:7px 0">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:baseline">'
        f'<div style="color:{_bias_color(entry.get("bias_hint"))};font-size:13px;font-weight:bold">{_esc(entry.get("name", "pattern"))}</div>'
        f'<div style="color:{MUTE};font-size:10px">{_esc(entry.get("window", ""))} &middot; {_esc(entry.get("family", ""))}</div></div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}"><b>Anatomy:</b> {_esc(entry.get("anatomy", ""))}</div>'
        f'<div style="color:{FG};font-size:11px;margin-top:4px;{WRAP}"><b>Auction read:</b> {_esc(entry.get("auction_read", ""))}</div>'
        f'<div style="color:{BLUE};font-size:11px;margin-top:4px;{WRAP}"><b>Where it matters:</b> {_esc(entry.get("where_it_matters", ""))}</div>'
        f'<div style="color:{CYAN};font-size:11px;margin-top:4px;{WRAP}"><b>Confirm:</b> {_esc(entry.get("confirmation", ""))}</div>'
        f'<div style="color:{AMBER};font-size:11px;margin-top:4px;{WRAP}"><b>Invalid:</b> {_esc(entry.get("invalidation", ""))}</div>'
        f'<div style="color:{RED};font-size:11px;margin-top:4px;{WRAP}"><b>False positive:</b> {_esc(entry.get("false_positive", ""))}</div>'
        "</div>"
        for entry in entries
    )
    doctrine = encyclopedia.get("doctrine") or "Candles are context, not permission."
    return (
        f'<details style="margin-top:10px"><summary style="color:{PURPLE};font-size:12px;font-weight:bold;cursor:pointer">'
        f"Deep candlestick encyclopedia ({len(entries)} entries)</summary>"
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px;{WRAP}">{_esc(doctrine)}</div>'
        f'<div style="max-height:420px;overflow:auto;padding-right:3px;margin-top:6px">{cards}</div></details>'
    )


def render_candle_coach_block(packet: dict[str, Any] | None) -> str:
    coach = packet or {}
    if not coach.get("available"):
        return ""
    recent = coach.get("recent_notable") or []
    library = [str(item) for item in coach.get("pattern_library") or []]
    library_items = "".join(
        f'<span style="display:inline-block;border:1px solid #30363d;border-radius:999px;padding:2px 7px;margin:2px;color:{MUTE};font-size:10px">{_esc(item)}</span>'
        for item in library
    )
    library_html = (
        f'<details style="margin-top:10px"><summary style="color:{CYAN};font-size:12px;font-weight:bold;cursor:pointer">Pattern encyclopedia covered ({len(library)})</summary>'
        f'<div style="margin-top:6px">{library_items}</div></details>'
        if library_items
        else ""
    )
    recent_items = "".join(
        f'<li style="margin:3px 0"><span style="color:{_bias_color(item.get("bias_hint"))};font-weight:bold">'
        f"{_esc(item.get('name', 'pattern'))}</span> "
        f'<span style="color:{MUTE}">({_esc(item.get("window", ""))} {_esc(item.get("clock", ""))})</span></li>'
        for item in recent
    )
    recent_html = (
        f'<div style="margin-top:10px;color:{FG};font-size:12px">'
        f'<div style="color:{MUTE};font-size:11px;font-weight:bold;letter-spacing:.08em">LAST NOTABLE CONFIGURATIONS</div>'
        f'<ul style="margin:6px 0 0 18px;padding:0">{recent_items}</ul></div>'
        if recent_items
        else ""
    )
    expectancy_html = _expectancy_block(coach.get("candle_expectancy") or {})
    vector_lesson_html = _candle_vector_lesson_block(
        coach.get("candle_vector_lesson") or {}
    )
    encyclopedia_html = _encyclopedia_block(coach.get("pattern_encyclopedia") or {})
    return (
        f'<div style="border:2px solid {PURPLE};background:{SURFACE};padding:12px;margin:10px 0;border-radius:10px">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap">'
        f'<div style="color:{PURPLE};font-size:13px;font-weight:bold;letter-spacing:.10em">CANDLE COACH — EVENT DETECTOR</div>'
        f'<div style="color:{MUTE};font-size:11px">education only &middot; not execution authority</div></div>'
        f'<div style="color:{FG};font-size:18px;font-weight:bold;margin-top:5px">{_esc(coach.get("headline", ""))}</div>'
        f'<div style="color:{MUTE};font-size:12px;margin-top:3px">Context: {_esc(coach.get("context", "unknown"))}</div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
        f"{_pattern_card(coach.get('latest_single') or {}, 'LATEST 1-CANDLE EVENT')}"
        f"{_pattern_card(coach.get('latest_pair') or {}, 'LATEST 2-CANDLE CONFIGURATION')}"
        f"{_pattern_card(coach.get('latest_three') or {}, 'LATEST 3-CANDLE PATTERN')}"
        f"{_pattern_card(coach.get('latest_structure') or {}, 'LARGER CANDLE STRUCTURE')}"
        f"</div>{expectancy_html}{vector_lesson_html}{_auction_box_block((coach.get('execution_framework') or {}).get('auction_execution_box') or {})}"
        f"{_framework_block(coach.get('execution_framework') or {})}{recent_html}{library_html}{encyclopedia_html}"
        f'<div style="color:{AMBER};font-size:11px;margin-top:10px;{WRAP}">'
        f"{_esc(coach.get('lesson', 'Candles teach auction behavior; always demand confirmation.'))}</div>"
        "</div>"
    )


__all__ = ["render_candle_coach_block"]
