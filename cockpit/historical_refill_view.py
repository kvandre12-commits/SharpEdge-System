"""HTML renderer for the historical refill surface cockpit card."""

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
CYAN = "#39c5cf"
WRAP = "overflow-wrap:anywhere;word-break:break-word"
SIDE_ORDER = ("above_ema200", "near_ema200", "below_ema200", "unknown")
DISTANCE_ORDER = (
    "above_5pct_plus",
    "above_2_to_5pct",
    "above_0_5_to_2pct",
    "near_ema200",
    "below_0_5_to_2pct",
    "below_2_to_5pct",
    "below_5pct_plus",
    "unknown",
)


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        if value in (None, ""):
            return "n/a"
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _mode_fill_text(est: dict[str, Any]) -> str:
    base = f"{_fmt(est.get('mode_trading_days'), 1)} trading days"
    count = est.get("mode_count")
    rate = _fmt(est.get("mode_rate_pct"), 1, "%")
    if count is None or rate == "n/a":
        return base
    return f"{base} ({count}× / {rate})"


def _timing_fill_text(est: dict[str, Any], key: str) -> str:
    base = f"{_fmt(est.get(f'{key}_trading_days'), 1)} trading days"
    window = est.get(f"{key}_horizon_days")
    rate = _fmt(est.get(f"{key}_horizon_rate_pct"), 1, "%")
    count = est.get(f"{key}_horizon_count")
    if window is None or rate == "n/a":
        return base
    count_text = f"{int(count)}× / " if isinstance(count, (int, float)) else ""
    return f"{base} ({count_text}≤{_fmt(window, 0)}d / {rate})"


def _ema_stats_text(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    count = stats.get("event_count")
    fill = _fmt(stats.get("fill_rate_pct"), 1, "%")
    median = _fmt(stats.get("median_trading_days"), 1)
    return f"{count} events · {fill} fill · med {median}d"


def _ema_context_rows(ctx: dict[str, Any]) -> list[tuple[str, str]]:
    ema = ctx.get("ema200_context") or {}
    if not ema:
        return []
    distance = _fmt(ema.get("distance_pct"), 2, "%")
    return [
        ("EMA200 side", ema.get("side") or "unknown"),
        ("EMA200 dist", distance),
        ("EMA side stats", _ema_stats_text(ema.get("side_stats"))),
        ("EMA bucket", ema.get("distance_bucket") or "unknown"),
        ("EMA bucket stats", _ema_stats_text(ema.get("distance_bucket_stats"))),
    ]


def _ordered_stats_items(
    stats: dict[str, Any], order: tuple[str, ...]
) -> list[tuple[str, Any]]:
    seen = set(order)
    items = [(key, stats.get(key)) for key in order if key in stats]
    items.extend(
        (key, value) for key, value in sorted(stats.items()) if key not in seen
    )
    return items


def _render_ema_surface_matrix(ctx: dict[str, Any]) -> str:
    ema = ctx.get("ema200_context") or {}
    side_stats = ema.get("all_side_stats") or {}
    bucket_stats = ema.get("all_distance_bucket_stats") or {}
    if not side_stats and not bucket_stats:
        return ""

    def rows(
        stats: dict[str, Any],
        order: tuple[str, ...],
        active_label: str | None,
    ) -> str:
        rendered = []
        for label, value in _ordered_stats_items(stats, order):
            active = label == active_label
            bg = "background:#1f2d1f;" if active else ""
            border = (
                f"border-left:3px solid {GREEN};padding-left:5px;" if active else ""
            )
            label_color = GREEN if active else MUTE
            marker = " ← current" if active else ""
            rendered.append(
                f'<div style="display:flex;justify-content:space-between;gap:8px;border-top:1px solid #30363d;padding:4px 0;{bg}{border}">'
                f'<span style="color:{label_color};font-weight:{"bold" if active else "normal"}">{_esc(label)}{marker}</span>'
                f'<span style="color:{FG};font-weight:bold;text-align:right">{_esc(_ema_stats_text(value))}</span>'
                "</div>"
            )
        return "".join(rendered)

    side_block = rows(side_stats, SIDE_ORDER, ema.get("side"))
    bucket_block = rows(bucket_stats, DISTANCE_ORDER, ema.get("distance_bucket"))
    return (
        f'<div style="margin-top:10px;color:{MUTE};font-size:11px;{WRAP}">'
        f'<div style="color:{AMBER};font-weight:bold">EMA200 REFILL BUCKET SURFACE</div>'
        f'<div style="color:{CYAN};margin-top:4px">Side buckets</div>{side_block}'
        f'<div style="color:{CYAN};margin-top:8px">Distance buckets</div>{bucket_block}'
        "</div>"
    )


def _stack_surface_stats_text(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    count = stats.get("reference_count")
    fill = _fmt(stats.get("new_event_fill_rate_pct"), 1, "%")
    median_days = _fmt(stats.get("new_event_median_trading_days"), 1)
    max_days = _fmt(stats.get("new_event_max_trading_days"), 1)
    full_stack = _fmt(stats.get("full_stack_resolved_rate_pct"), 1, "%")
    stack_median = _fmt(
        stats.get("full_stack_resolution_calendar_median_days"),
        1,
    )
    return (
        f"{count} refs · latest fill {fill} · med {median_days}d · "
        f"max {max_days}d · full stack {full_stack} · stack med {stack_median} cal-d"
    )


def _stack_label_text(label: Any) -> str:
    return str(label or "single_active_dip").replace("_", " ").upper()


def _stack_summary_text(ctx: dict[str, Any]) -> str:
    stack = ctx.get("active_refill_stack") or {}
    count = stack.get("active_count") or 1
    label = _stack_label_text(stack.get("stack_label"))
    high = _fmt(stack.get("highest_target"), 2)
    near = _fmt(stack.get("nearest_target"), 2)
    return f"{label} · {count} open · targets ${near}–${high}"


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _preferred_stack_stats(
    history: dict[str, Any],
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    for label, key in (
        ("exact stack", "exact_signature_stats"),
        ("last bucket pair", "last_pair_distance_bucket_stats"),
        ("new EMA bucket", "new_ema_distance_bucket_stats"),
        ("new EMA side", "new_ema_side_stats"),
        ("all stacks", "overall"),
    ):
        stats = history.get(key)
        if stats:
            return label, stats
    return None, None


def _stack_read_label(stats: dict[str, Any]) -> str:
    latest = _num(stats.get("new_event_fill_rate_pct"))
    full_stack = _num(stats.get("full_stack_resolved_rate_pct"))
    if latest is None:
        return "STACK CONTEXT ONLY"
    if full_stack is not None and latest >= 90 and full_stack <= latest - 20:
        return "FAST LATEST REFILL / OLDER TARGET MAGNET"
    if full_stack is not None and latest >= 80 and full_stack >= 70:
        return "STACK RESOLUTION EDGE"
    if latest >= 80:
        return "LATEST REFILL EDGE"
    if latest >= 60:
        return "MIXED STACK REFILL"
    return "THIN/WEAK STACK CONTEXT"


def _render_stack_read(ctx: dict[str, Any]) -> str:
    stack = ctx.get("active_refill_stack") or {}
    if (stack.get("active_count") or 1) <= 1:
        return ""
    history = stack.get("historical_stack_surface") or {}
    source, stats = _preferred_stack_stats(history)
    if not source or not stats:
        return ""

    label = _stack_read_label(stats)
    count = stats.get("reference_count")
    latest = _num(stats.get("new_event_fill_rate_pct"))
    latest_score = "n/a" if latest is None else f"{round(latest):.0f}/100"
    latest_med = _fmt(stats.get("new_event_median_trading_days"), 1)
    full_stack = _fmt(stats.get("full_stack_resolved_rate_pct"), 1, "%")
    stack_med = _fmt(stats.get("full_stack_resolution_calendar_median_days"), 1)
    return (
        f'<div style="border:1px solid #30363d;background:#0d1117;'
        f'padding:8px;margin-top:10px;border-radius:6px;{WRAP}">'
        f'<div style="color:{MUTE};font-size:10px;letter-spacing:.08em;'
        f'font-weight:bold">STACK READ · diagnostic only</div>'
        f'<div style="color:{GREEN};font-size:14px;font-weight:bold;margin-top:3px">'
        f"{_esc(label)} · latest refill score {_esc(latest_score)}</div>"
        f'<div style="color:#adbac7;font-size:12px;margin-top:3px">'
        f"{_esc(source)} · {count} refs · latest med {latest_med}d · "
        f"full stack {full_stack} · stack med {stack_med} cal-d</div>"
        "</div>"
    )


def _render_stack_history(ctx: dict[str, Any]) -> str:
    stack = ctx.get("active_refill_stack") or {}
    history = stack.get("historical_stack_surface") or {}
    if not history:
        return ""
    rows = [
        (
            "exact stack",
            _stack_surface_stats_text(history.get("exact_signature_stats")),
        ),
        (
            "last bucket pair",
            _stack_surface_stats_text(history.get("last_pair_distance_bucket_stats")),
        ),
        (
            "new EMA bucket",
            _stack_surface_stats_text(history.get("new_ema_distance_bucket_stats")),
        ),
        ("new EMA side", _stack_surface_stats_text(history.get("new_ema_side_stats"))),
        ("all stacks", _stack_surface_stats_text(history.get("overall"))),
    ]
    rendered = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:8px;border-top:1px solid #30363d;padding:4px 0">'
        f'<span style="color:{MUTE}">{_esc(label)}</span>'
        f'<span style="color:{FG};font-weight:bold;text-align:right">{_esc(value)}</span>'
        "</div>"
        for label, value in rows
    )
    excluded = history.get("latest_event_date_excluded_from_stats")
    note = "latest event date excluded" if excluded else "latest event date included"
    return (
        f'<div style="margin-top:10px;color:{MUTE};font-size:11px;{WRAP}">'
        f'<div style="color:{AMBER};font-weight:bold">HISTORICAL STACK SURFACE</div>'
        f"{rendered}"
        f'<div style="margin-top:5px">stats basis: {_esc(note)}</div>'
        "</div>"
    )


def _render_stack_items(ctx: dict[str, Any]) -> str:
    stack = ctx.get("active_refill_stack") or {}
    items = stack.get("items") or []
    if len(items) <= 1:
        return ""
    rows = "".join(
        f'<div style="border-top:1px solid #30363d;padding-top:6px;margin-top:6px">'
        f'<span style="color:{FG};font-weight:bold">{_esc(item.get("event_date"))}</span> '
        f'<span style="color:{MUTE}">{_esc(item.get("mode"))} ≥ {_fmt(item.get("threshold_pct"), 1, "%")}</span><br>'
        f'<span style="color:{CYAN}">target ${_fmt(item.get("target"), 2)} · '
        f"{_fmt(item.get('move_pct'), 2, '%')} move · "
        f"{item.get('elapsed_trading_days')}d open · "
        f"{_esc(item.get('ema200_distance_bucket') or 'unknown')}</span>"
        "</div>"
        for item in items[-4:]
    )
    signature = _esc(stack.get("interaction_signature") or "")
    return (
        f'<div style="margin-top:10px;color:{MUTE};font-size:11px;{WRAP}">'
        f'<div style="color:{AMBER};font-weight:bold">ACTIVE REFILL STACK</div>'
        f"{rows}"
        f'<div style="margin-top:6px">signature: {signature}</div>'
        "</div>"
    )


def _phase_color(phase: str) -> str:
    return {
        "inside_median_window": GREEN,
        "late_but_inside_observed_window": AMBER,
        "beyond_observed_window": RED,
        "unknown": MUTE,
    }.get(phase, BLUE)


def render_historical_refill_context_block(
    historical_refill_context: dict[str, Any] | None,
) -> str:
    """Render active refill-window diagnostics.

    This is explicitly context, not authority. It must stay visually separate
    from execution permission / approval gates.
    """
    ctx = historical_refill_context or {}
    if not ctx.get("available"):
        return ""

    est = ctx.get("estimated") or {}
    phase = str(est.get("phase") or "unknown")
    color = _phase_color(phase)
    remaining = ctx.get("remaining_points_to_fill")
    remaining_text = (
        f"{float(remaining):+.2f} pts" if isinstance(remaining, (int, float)) else "n/a"
    )
    sample = ctx.get("event_count")
    fill_rate = _fmt(ctx.get("fill_rate_pct"), suffix="%")
    within_20 = _fmt(est.get("fill_within_20d_rate_pct"), suffix="%")
    within_60 = _fmt(est.get("fill_within_60d_rate_pct"), suffix="%")
    rows = [
        ("bucket", f"{ctx.get('mode')} ≥ {ctx.get('threshold_pct')}%"),
        ("stack", _stack_summary_text(ctx)),
        ("event date", ctx.get("event_date")),
        ("target", f"${_fmt(ctx.get('gap_fill_target'), 2)}"),
        ("spot to target", remaining_text),
        ("elapsed", f"{ctx.get('elapsed_trading_days')} trading days"),
        ("median fill", _timing_fill_text(est, "median")),
        ("mode fill", _mode_fill_text(est)),
        ("mean fill", _timing_fill_text(est, "mean")),
        ("max observed", f"{_fmt(est.get('max_trading_days'), 1)} trading days"),
        ("sample", f"{sample} events" if sample is not None else "n/a"),
        ("fill rate", fill_rate),
        ("≤20d", within_20),
        ("≤60d", within_60),
        *_ema_context_rows(ctx),
    ]
    grid = "".join(
        f'<div><div style="color:{MUTE};font-size:10px;text-transform:uppercase">{_esc(label)}</div>'
        f'<div style="color:{FG};font-weight:bold;font-size:13px">{_esc(value)}</div></div>'
        for label, value in rows
    )
    return (
        f'<div style="border:2px solid {color};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">HISTORICAL REFILL SURFACE (active window / diagnostic)</div>'
        f'<div style="color:{color};font-size:19px;font-weight:bold;margin-top:4px;{WRAP}">{_esc(ctx.get("headline") or "ACTIVE REFILL WINDOW")}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(ctx.get("story") or "")}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:10px">{grid}</div>'
        f'<div style="color:{CYAN};font-size:12px;margin-top:8px;{WRAP}">{_esc(est.get("story") or "")}</div>'
        f"{_render_stack_read(ctx)}"
        f"{_render_stack_items(ctx)}"
        f"{_render_stack_history(ctx)}"
        f"{_render_ema_surface_matrix(ctx)}"
        f'<div style="color:{MUTE};font-size:11px;margin-top:6px;{WRAP}">{_esc(ctx.get("caveat") or "Diagnostic only; no authority change.")}</div>'
        "</div>"
    )


__all__ = ["render_historical_refill_context_block"]
