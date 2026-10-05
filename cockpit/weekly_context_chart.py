from __future__ import annotations

from collections import defaultdict
from typing import Any

from context_chart_focus import build_focus_window
from execution_vector_context import swing_points
from market_data_sources import fetch_yahoo_regular_session_chart_rows

W, H = 1000, 360
PAD_L, PAD_R, PAD_T, PAD_B = 48, 80, 30, 28
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B
CARRY_LEVEL_META = {
    "H1": {
        "color": "#f85149",
        "dash": "7 4",
        "short_label": "H1 peak",
        "long_label": "H1 session peak",
        "chart_label": "H1",
    },
    "LH1": {
        "color": "#d29922",
        "dash": "4 4",
        "short_label": "LH1 lower high",
        "long_label": "LH1 lower high",
        "chart_label": "LH1",
    },
    "HL1": {
        "color": "#39c5cf",
        "dash": "4 4",
        "short_label": "HL1 higher low",
        "long_label": "HL1 higher low",
        "chart_label": "HL1",
    },
    "L1": {
        "color": "#26a641",
        "dash": "7 4",
        "short_label": "L1 washout low",
        "long_label": "L1 washout low",
        "chart_label": "L1",
    },
}


def fetch_weekly_context_rows(
    symbol: str = "SPY",
    *,
    interval: str = "5m",
    range_: str = "14d",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, source = fetch_yahoo_regular_session_chart_rows(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    return rows, source


def derive_today_carry_levels(
    session_rows: list[tuple[int, float, float, float, float, int]],
) -> list[dict[str, Any]]:
    if not session_rows:
        return []
    highs, lows = swing_points(session_rows, window=2)
    h1 = _primary_high(highs, session_rows)
    l1 = _primary_low(lows, session_rows)
    lh1 = _recent_lower_high(highs, h1)
    hl1 = _recent_higher_low(lows, l1)

    levels = [
        _carry_level("H1", h1),
        _carry_level("LH1", lh1),
        _carry_level("HL1", hl1),
        _carry_level("L1", l1),
    ]
    return [level for level in levels if level is not None]


def summarize_weekly_context(
    recent_rows: list[dict[str, Any]],
    carry_levels: list[dict[str, Any]],
    *,
    spot: float,
    symbol: str = "SPY",
    lookback_days: int = 5,
) -> dict[str, Any]:
    grouped = _group_recent_days(recent_rows, lookback_days=lookback_days)
    ordered_days = list(grouped.items())
    closes = [float(row["close"]) for _day, rows in ordered_days for row in rows]
    if not closes:
        return {
            "symbol": symbol,
            "lookback_days": 0,
            "panel_note": "Top chart = live execution tape. Middle chart = weekly carry map zoomed on active trade neighborhood. Distant rails stay in the chips/text.",
            "headline": "Weekly carry map unavailable",
            "detail": "No multi-day context data came back.",
            "kind": "info",
            "legend": [],
        }

    range_low = min(closes)
    range_high = max(closes)
    range_span = max(range_high - range_low, 1e-9)
    range_position_pct = round((spot - range_low) / range_span * 100)
    levels_by_name = {level["name"]: float(level["price"]) for level in carry_levels}
    nearest = _nearest_carry_level(spot, carry_levels)
    headline, detail, kind = _structure_read(
        spot,
        levels_by_name,
        nearest,
        range_low=range_low,
        range_high=range_high,
        range_position_pct=range_position_pct,
    )
    legend = [
        {
            "name": name,
            "label": CARRY_LEVEL_META[name]["short_label"],
            "price": levels_by_name[name],
            "color": CARRY_LEVEL_META[name]["color"],
        }
        for name in ["H1", "LH1", "HL1", "L1"]
        if name in levels_by_name
    ]
    return {
        "symbol": symbol,
        "lookback_days": len(ordered_days),
        "panel_note": (
            f"Middle chart = {len(ordered_days)}-day carry map zoomed on active trade neighborhood. Distant rails stay in the chips/text."
        ),
        "headline": headline,
        "detail": detail,
        "kind": kind,
        "legend": legend,
        "range_position_pct": range_position_pct,
        "range_low": round(range_low, 2),
        "range_high": round(range_high, 2),
    }


def build_weekly_context_svg(
    recent_rows: list[dict[str, Any]],
    carry_levels: list[dict[str, Any]],
    *,
    symbol: str = "SPY",
    lookback_days: int = 5,
) -> str:
    if not recent_rows:
        return _empty_weekly_context_svg(symbol)

    grouped = _group_recent_days(recent_rows, lookback_days=lookback_days)
    ordered_days = list(grouped.items())
    flat_rows = [row for _day, rows in ordered_days for row in rows]
    closes = [float(row["close"]) for row in flat_rows]
    latest_rows = ordered_days[-1][1]
    recent_focus = [float(row["close"]) for row in latest_rows]
    lo, hi, visible_level_names = build_focus_window(
        closes,
        carry_levels,
        anchor=closes[-1],
        recent_values=recent_focus,
        min_span_pct=0.9,
        nearby_level_pct=0.55,
        outer_level_pct=1.1,
        pad_ratio=0.12,
    )
    span = hi - lo

    def px(index: float) -> float:
        return PAD_L + (index / max(len(flat_rows) - 1, 1)) * PLOT_W

    def py(price: float) -> float:
        return PAD_T + (1 - (price - lo) / span) * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
        f'<text x="{PAD_L}" y="22" fill="#e6edf3" font-size="15" font-weight="bold">'
        f"{symbol} 5-day carry map</text>",
        f'<text x="{PAD_L + PLOT_W}" y="22" fill="#7d8590" font-size="11" text-anchor="end">'
        f"today = bright blue • older days fade</text>",
    ]

    latest_start = len(flat_rows) - len(latest_rows)
    latest_x = px(latest_start)
    latest_w = px(len(flat_rows) - 1) - latest_x
    parts.append(
        f'<rect x="{latest_x:.1f}" y="{PAD_T}" width="{latest_w:.1f}" height="{PLOT_H:.1f}" '
        f'fill="#58a6ff" opacity="0.06" rx="6"/>'
    )
    parts.append(
        f'<text x="{latest_x + 8:.1f}" y="{PAD_T + 14:.1f}" fill="#58a6ff" font-size="10" font-weight="bold">TODAY</text>'
    )

    for step in range(6):
        value = lo + span * step / 5
        grid_y = py(value)
        parts.append(
            f'<line x1="{PAD_L}" y1="{grid_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{grid_y:.1f}" '
            f'stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 6}" y="{grid_y + 4:.1f}" fill="#7d8590" font-size="10" text-anchor="end">'
            f"${value:.2f}</text>"
        )

    start = 0
    for day, day_rows in ordered_days:
        end = start + len(day_rows) - 1
        day_x = px(start)
        parts.append(
            f'<line x1="{day_x:.1f}" y1="{PAD_T}" x2="{day_x:.1f}" y2="{PAD_T + PLOT_H}" '
            f'stroke="#161b22" stroke-width="1"/>'
        )
        mid = px((start + end) / 2)
        parts.append(
            f'<text x="{mid:.1f}" y="{H - 10}" fill="#7d8590" font-size="10" text-anchor="middle">'
            f"{day[5:]}</text>"
        )
        start = end + 1

    for level in carry_levels:
        if level["name"] not in visible_level_names:
            continue
        meta = CARRY_LEVEL_META.get(level["name"], {})
        color = meta.get("color", "#7d8590")
        dash = meta.get("dash", "3 5")
        chart_label = meta.get("chart_label", level["name"])
        level_y = py(float(level["price"]))
        parts.append(
            f'<line x1="{PAD_L}" y1="{level_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{level_y:.1f}" '
            f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}" opacity="0.95"/>'
        )
        parts.append(
            f'<text x="{PAD_L + PLOT_W + 6}" y="{level_y + 4:.1f}" fill="{color}" font-size="10" font-weight="bold">'
            f"{chart_label} {float(level['price']):.2f}</text>"
        )

    start = 0
    total_days = len(ordered_days)
    for index, (_day, day_rows) in enumerate(ordered_days):
        pts = " ".join(
            f"{px(start + offset):.1f},{py(float(row['close'])):.1f}"
            for offset, row in enumerate(day_rows)
        )
        age_from_latest = total_days - 1 - index
        if age_from_latest == 0:
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="5.8" opacity="0.16"/>'
            )
            color = "#58a6ff"
            width = 2.9
            opacity = 1.0
        elif age_from_latest == 1:
            color = "#8bb8ff"
            width = 2.2
            opacity = 0.74
        else:
            fade_rank = index / max(total_days - 2, 1)
            shade = int(88 + 34 * fade_rank)
            color = f"rgb({shade // 2},{shade - 4},{min(255, shade + 46)})"
            width = 1.35
            opacity = 0.18 + 0.14 * fade_rank
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity:.2f}"/>'
        )
        start += len(day_rows)

    last_close = closes[-1]
    last_y = py(last_close)
    parts.append(
        f'<circle cx="{px(len(flat_rows) - 1):.1f}" cy="{last_y:.1f}" r="3.8" fill="#ffd33d"/>'
    )
    parts.append(
        f'<text x="{px(len(flat_rows) - 1) + 8:.1f}" y="{last_y + 4:.1f}" fill="#ffd33d" '
        f'font-size="11" font-weight="bold">TODAY ${last_close:.2f}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _group_recent_days(
    recent_rows: list[dict[str, Any]],
    *,
    lookback_days: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recent_rows:
        grouped[str(row["date"])].append(row)
    ordered_dates = sorted(grouped)[-lookback_days:]
    return {date: grouped[date] for date in ordered_dates}


def _structure_read(
    spot: float,
    levels_by_name: dict[str, float],
    nearest: dict[str, Any] | None,
    *,
    range_low: float,
    range_high: float,
    range_position_pct: int,
) -> tuple[str, str, str]:
    h1 = levels_by_name.get("H1")
    lh1 = levels_by_name.get("LH1")
    hl1 = levels_by_name.get("HL1")
    l1 = levels_by_name.get("L1")
    nearest_text = _nearest_text(nearest)
    range_text = f"5-day range ${range_low:.2f} -> ${range_high:.2f} ({range_position_pct}% up the range)."

    if h1 is not None and spot >= h1:
        return (
            "Pressing through H1 at the top of the 5-day map",
            f"Spot ${spot:.2f} is through H1 ${h1:.2f}. {nearest_text} {range_text}",
            "ok",
        )
    if lh1 is not None and h1 is not None and lh1 <= spot < h1:
        return (
            "Holding the upper carry shelf beneath H1",
            f"Spot ${spot:.2f} is between LH1 ${lh1:.2f} and H1 ${h1:.2f}. {nearest_text} {range_text}",
            "ok",
        )
    if hl1 is not None and lh1 is not None and hl1 <= spot < lh1:
        return (
            "Inside today’s carry box, not yet pressing the highs",
            f"Spot ${spot:.2f} is between HL1 ${hl1:.2f} and LH1 ${lh1:.2f}. {nearest_text} {range_text}",
            "info",
        )
    if l1 is not None and hl1 is not None and l1 < spot < hl1:
        return (
            "Leaning on the lower carry shelf",
            f"Spot ${spot:.2f} is below HL1 ${hl1:.2f} but still above L1 ${l1:.2f}. {nearest_text} {range_text}",
            "warn",
        )
    if l1 is not None and spot <= l1:
        return (
            "Back at today’s L1 washout low",
            f"Spot ${spot:.2f} is testing or through L1 ${l1:.2f}. {nearest_text} {range_text}",
            "bad",
        )
    return (
        "Reading today’s pivots against the 5-day tape",
        f"{nearest_text} {range_text}",
        "info",
    )


def _nearest_carry_level(
    spot: float,
    carry_levels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not carry_levels:
        return None
    nearest = min(carry_levels, key=lambda level: abs(float(level["price"]) - spot))
    meta = CARRY_LEVEL_META.get(nearest["name"], {})
    distance_pct = abs(float(nearest["price"]) - spot) / max(spot, 1e-9) * 100
    return {
        "name": nearest["name"],
        "label": meta.get("long_label", nearest["name"]),
        "price": float(nearest["price"]),
        "distance_pct": distance_pct,
    }


def _nearest_text(nearest: dict[str, Any] | None) -> str:
    if not nearest:
        return "No carry pivot was isolated cleanly."
    return (
        f"Nearest carry pivot: {nearest['label']} ${nearest['price']:.2f} "
        f"({nearest['distance_pct']:.2f}% away)."
    )


def _primary_high(
    highs: list[tuple[int, float]],
    session_rows: list[tuple[int, float, float, float, float, int]],
) -> tuple[int, float]:
    if highs:
        return max(highs, key=lambda item: (item[1], item[0]))
    fallback_index, fallback_bar = max(
        enumerate(session_rows), key=lambda item: item[1][2]
    )
    return fallback_index, float(fallback_bar[2])


def _primary_low(
    lows: list[tuple[int, float]],
    session_rows: list[tuple[int, float, float, float, float, int]],
) -> tuple[int, float]:
    if lows:
        return min(lows, key=lambda item: (item[1], -item[0]))
    fallback_index, fallback_bar = min(
        enumerate(session_rows), key=lambda item: item[1][3]
    )
    return fallback_index, float(fallback_bar[3])


def _recent_lower_high(
    highs: list[tuple[int, float]],
    h1: tuple[int, float],
) -> tuple[int, float] | None:
    return _most_recent_point(
        highs,
        min_price=None,
        max_price=h1[1] - 1e-9,
        after_index=h1[0],
    )


def _recent_higher_low(
    lows: list[tuple[int, float]],
    l1: tuple[int, float],
) -> tuple[int, float] | None:
    return _most_recent_point(
        lows,
        min_price=l1[1] + 1e-9,
        max_price=None,
        after_index=l1[0],
    )


def _most_recent_point(
    points: list[tuple[int, float]],
    *,
    min_price: float | None,
    max_price: float | None,
    after_index: int,
) -> tuple[int, float] | None:
    def matches(point: tuple[int, float]) -> bool:
        _index, price = point
        if min_price is not None and price < min_price:
            return False
        if max_price is not None and price > max_price:
            return False
        return True

    after_candidates = [
        point for point in points if point[0] > after_index and matches(point)
    ]
    if after_candidates:
        return max(after_candidates, key=lambda item: item[0])
    fallback_candidates = [point for point in points if matches(point)]
    if fallback_candidates:
        return max(fallback_candidates, key=lambda item: item[0])
    return None


def _carry_level(name: str, point: tuple[int, float] | None) -> dict[str, Any] | None:
    if point is None:
        return None
    index, price = point
    return {"name": name, "price": round(float(price), 2), "session_index": int(index)}


def _empty_weekly_context_svg(symbol: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="monospace">'
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>'
        f'<text x="{PAD_L}" y="36" fill="#e6edf3" font-size="15" font-weight="bold">{symbol} weekly context</text>'
        f'<text x="{PAD_L}" y="72" fill="#7d8590" font-size="12">No multi-day context data available.</text>'
        "</svg>"
    )


__all__ = [
    "build_weekly_context_svg",
    "derive_today_carry_levels",
    "fetch_weekly_context_rows",
    "summarize_weekly_context",
]
