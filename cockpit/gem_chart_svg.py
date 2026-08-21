"""Dedicated gem-first chart for SharpEdge.

This chart emphasizes the user's core read:
price, channel, fair value gaps, VWAP, current execution levels,
and the near/strategic exit path.
"""

from __future__ import annotations

from typing import Any

W, H = 1080, 460
PAD_L, PAD_R, PAD_T, PAD_B = 46, 86, 18, 34
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B
BG = "#0d1117"
GRID = "#1f2630"
MUTE = "#8b949e"
VWAP = "#ffd33d"
TARGET = "#39c5cf"
CHANNEL = "#bc8cff"
BULL_FVG = "#3fb950"
BEAR_FVG = "#f85149"
ENTRY = "#58a6ff"
FAIL = "#d29922"


def _channel_overlay(
    volatility_structure: dict[str, Any] | None,
) -> tuple[float, float] | None:
    state = volatility_structure or {}
    channel_high = state.get("channel_high")
    channel_low = state.get("channel_low")
    if not isinstance(channel_high, (int, float)) or not isinstance(
        channel_low, (int, float)
    ):
        return None
    if float(channel_high) <= float(channel_low):
        return None
    return float(channel_low), float(channel_high)


def _display_gaps(fair_value_gaps: dict[str, Any] | None) -> list[dict[str, Any]]:
    packet = fair_value_gaps or {}
    candidates: list[dict[str, Any]] = []
    seen = set()
    for gap in (
        packet.get("nearest_open_gap_above") or {},
        packet.get("nearest_open_gap_below") or {},
        packet.get("nearest_open_gap") or {},
    ):
        if not gap:
            continue
        signature = (
            gap.get("direction"),
            gap.get("gap_low"),
            gap.get("gap_high"),
            gap.get("created_index"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(gap)
    return candidates[:2]


def _line_markers(
    target_plan: dict[str, Any] | None,
    entry_gate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    target = target_plan or {}
    gate = entry_gate or {}
    markers: list[dict[str, Any]] = []

    trigger_price = gate.get("trigger_price")
    if isinstance(trigger_price, (int, float)):
        markers.append(
            {
                "label": "ENTRY TRIGGER",
                "price": float(trigger_price),
                "color": ENTRY,
                "dash": "4 3",
            }
        )

    level_price = gate.get("level_price")
    if isinstance(level_price, (int, float)):
        level_name = str(gate.get("level_name") or "LEVEL").upper()
        markers.append(
            {
                "label": f"FAIL {level_name}",
                "price": float(level_price),
                "color": FAIL,
                "dash": "2 4",
            }
        )

    reachable = target.get("reachable_today") or {}
    reachable_price = reachable.get("price")
    if isinstance(reachable_price, (int, float)):
        reachable_label = str(reachable.get("label") or "REACHABLE EXIT").upper()
        markers.append(
            {
                "label": f"EXIT {reachable_label}",
                "price": float(reachable_price),
                "color": TARGET,
                "dash": "8 4",
            }
        )

    target_price = target.get("price")
    if isinstance(target_price, (int, float)):
        target_label = str(target.get("label") or "STRATEGIC EXIT").upper()
        markers.append(
            {
                "label": f"STRATEGIC {target_label}",
                "price": float(target_price),
                "color": TARGET,
                "dash": "10 4",
            }
        )
    return markers


def _entry_zone_callout(
    rows: list[tuple[Any, ...]] | list[list[Any]],
    entry_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    gate = entry_gate or {}
    bars_ago = gate.get("bars_ago")
    trigger_price = gate.get("trigger_price")
    level_price = gate.get("level_price")
    if not gate.get("actionable"):
        return {}
    if not isinstance(bars_ago, int):
        return {}
    prices = [
        float(value)
        for value in (trigger_price, level_price)
        if isinstance(value, (int, float))
    ]
    if not prices:
        return {}
    index = max(0, min(len(rows) - 1, len(rows) - 1 - int(bars_ago)))
    return {
        "index": index,
        "zone_low": min(prices),
        "zone_high": max(prices),
        "trigger_price": (
            float(trigger_price) if isinstance(trigger_price, (int, float)) else None
        ),
    }


def render_gem_chart_svg(
    rows: list[tuple[Any, ...]] | list[list[Any]],
    pa: dict[str, Any],
    target_plan: dict[str, Any] | None = None,
    volatility_structure: dict[str, Any] | None = None,
    fair_value_gaps: dict[str, Any] | None = None,
    entry_gate: dict[str, Any] | None = None,
) -> str:
    closes = [float(bar[4]) for bar in rows]
    highs = [float(bar[2]) for bar in rows]
    lows = [float(bar[3]) for bar in rows]
    n = len(closes)
    channel = _channel_overlay(volatility_structure)
    display_gaps = _display_gaps(fair_value_gaps)
    line_markers = _line_markers(target_plan, entry_gate)
    entry_zone = _entry_zone_callout(rows, entry_gate)

    values = [*highs, *lows]
    if isinstance(pa.get("vwap"), (int, float)):
        values.append(float(pa["vwap"]))
    if channel:
        values.extend(channel)
    for gap in display_gaps:
        if isinstance(gap.get("gap_low"), (int, float)):
            values.append(float(gap["gap_low"]))
        if isinstance(gap.get("gap_high"), (int, float)):
            values.append(float(gap["gap_high"]))
    for marker in line_markers:
        values.append(float(marker["price"]))
    if entry_zone:
        values.extend([entry_zone["zone_low"], entry_zone["zone_high"]])

    lo = min(values)
    hi = max(values)
    span = (hi - lo) or 1.0
    pad = span * 0.08
    lo -= pad
    hi += pad
    span = hi - lo

    def x(index: int) -> float:
        return PAD_L + index / max(n - 1, 1) * PLOT_W

    def y(price: float) -> float:
        return PAD_T + (1 - (price - lo) / span) * PLOT_H

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
    ]

    for step in range(5):
        price = lo + span * step / 4
        gy = y(price)
        out.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L + PLOT_W}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{PAD_L - 8}" y="{gy + 4:.1f}" fill="{MUTE}" font-size="11" text-anchor="end">${price:.2f}</text>'
        )

    if channel:
        channel_low, channel_high = channel
        top = y(channel_high)
        height = max(y(channel_low) - top, 1.0)
        out.append(
            f'<rect x="{PAD_L}" y="{top:.1f}" width="{PLOT_W:.1f}" height="{height:.1f}" fill="{CHANNEL}" opacity="0.10" rx="8"/>'
        )
        for label, price in (("CHANNEL HI", channel_high), ("CHANNEL LO", channel_low)):
            anchor_y = y(price)
            out.append(
                f'<line x1="{PAD_L}" y1="{anchor_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{anchor_y:.1f}" stroke="{CHANNEL}" stroke-width="1.3" stroke-dasharray="7 4"/>'
            )
            out.append(
                f'<text x="{PAD_L + 8}" y="{anchor_y - 6:.1f}" fill="{CHANNEL}" font-size="11" font-weight="bold">{label} {price:.2f}</text>'
            )

    for gap in display_gaps:
        gap_low = gap.get("gap_low")
        gap_high = gap.get("gap_high")
        if not isinstance(gap_low, (int, float)) or not isinstance(
            gap_high, (int, float)
        ):
            continue
        color = BULL_FVG if str(gap.get("direction")) == "bullish" else BEAR_FVG
        top = y(float(gap_high))
        height = max(y(float(gap_low)) - top, 1.0)
        out.append(
            f'<rect x="{PAD_L}" y="{top:.1f}" width="{PLOT_W:.1f}" height="{height:.1f}" fill="{color}" opacity="0.08"/>'
        )
        out.append(
            f'<line x1="{PAD_L}" y1="{top:.1f}" x2="{PAD_L + PLOT_W}" y2="{top:.1f}" stroke="{color}" stroke-width="1" opacity="0.85"/>'
        )
        out.append(
            f'<line x1="{PAD_L}" y1="{y(float(gap_low)):.1f}" x2="{PAD_L + PLOT_W}" y2="{y(float(gap_low)):.1f}" stroke="{color}" stroke-width="1" opacity="0.85"/>'
        )
        out.append(
            f'<text x="{PAD_L + PLOT_W - 8}" y="{top - 6:.1f}" fill="{color}" font-size="11" font-weight="bold" text-anchor="end">{str(gap.get("direction") or "gap").upper()} FVG {gap_low:.2f}-{gap_high:.2f}</text>'
        )

    vwap = pa.get("vwap")
    if isinstance(vwap, (int, float)):
        vy = y(float(vwap))
        out.append(
            f'<line x1="{PAD_L}" y1="{vy:.1f}" x2="{PAD_L + PLOT_W}" y2="{vy:.1f}" stroke="{VWAP}" stroke-width="1.4" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{PAD_L + PLOT_W + 6}" y="{vy + 4:.1f}" fill="{VWAP}" font-size="12" font-weight="bold">VWAP {float(vwap):.2f}</text>'
        )

    if entry_zone:
        anchor_x = x(int(entry_zone["index"]))
        zone_top = y(float(entry_zone["zone_high"]))
        zone_height = max(y(float(entry_zone["zone_low"])) - zone_top, 10.0)
        zone_left = max(PAD_L, anchor_x - 18)
        out.append(
            f'<line x1="{anchor_x:.1f}" y1="{PAD_T}" x2="{anchor_x:.1f}" y2="{PAD_T + PLOT_H}" stroke="{ENTRY}" stroke-width="1" stroke-dasharray="2 5" opacity="0.75"/>'
        )
        out.append(
            f'<rect x="{zone_left:.1f}" y="{zone_top:.1f}" width="36" height="{zone_height:.1f}" rx="8" fill="{ENTRY}" opacity="0.18" stroke="{ENTRY}" stroke-width="1.2"/>'
        )
        out.append(
            f'<text x="{anchor_x:.1f}" y="{max(PAD_T + 12, zone_top - 8):.1f}" fill="{ENTRY}" font-size="11" font-weight="bold" text-anchor="middle">ENTRY ZONE</text>'
        )
        out.append(
            f'<text x="{anchor_x:.1f}" y="{min(PAD_T + PLOT_H - 6, zone_top + zone_height + 14):.1f}" fill="{ENTRY}" font-size="10" font-weight="bold" text-anchor="middle">TRIGGER CANDLE</text>'
        )
        trigger_price = entry_zone.get("trigger_price")
        if isinstance(trigger_price, (int, float)):
            out.append(
                f'<circle cx="{anchor_x:.1f}" cy="{y(float(trigger_price)):.1f}" r="4.2" fill="{ENTRY}"/>'
            )

    for marker in line_markers:
        price = float(marker["price"])
        anchor_y = y(price)
        out.append(
            f'<line x1="{PAD_L}" y1="{anchor_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{anchor_y:.1f}" stroke="{marker["color"]}" stroke-width="1.5" stroke-dasharray="{marker["dash"]}" opacity="0.95"/>'
        )
        out.append(
            f'<text x="{PAD_L + PLOT_W + 6}" y="{anchor_y + 4:.1f}" fill="{marker["color"]}" font-size="12" font-weight="bold">{marker["label"]} {price:.2f}</text>'
        )

    line_color = BULL_FVG if float(pa.get("vs_vwap") or 0.0) >= 0 else BEAR_FVG
    points = " ".join(f"{x(i):.1f},{y(closes[i]):.1f}" for i in range(n))
    out.append(
        f'<polyline points="{points}" fill="none" stroke="{line_color}" stroke-width="2.8"/>'
    )
    out.append(
        f'<circle cx="{x(n - 1):.1f}" cy="{y(closes[-1]):.1f}" r="4.6" fill="#58a6ff"/>'
    )
    out.append(
        f'<text x="{PAD_L + PLOT_W + 6}" y="{y(closes[-1]) - 8:.1f}" fill="#58a6ff" font-size="13" font-weight="bold">SPOT {closes[-1]:.2f}</text>'
    )
    out.append(
        f'<text x="{PAD_L}" y="{H - 10}" fill="{MUTE}" font-size="11">price + channels + trigger zone + fail level + exits + FVG zones</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


__all__ = ["render_gem_chart_svg"]
