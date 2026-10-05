from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

from context_chart_focus import build_focus_window
from market_data_sources import fetch_yahoo_daily_bars

W, H = 1000, 340
PAD_L, PAD_R, PAD_T, PAD_B = 48, 86, 34, 32
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B
MONTHLY_LEVEL_META = {
    "PMH": {
        "color": "#f85149",
        "dash": "7 4",
        "short_label": "Prior month high",
        "long_label": "prior month high",
        "chart_label": "PMH",
    },
    "MOPEN": {
        "color": "#58a6ff",
        "dash": "1 0",
        "short_label": "Month open",
        "long_label": "current month open",
        "chart_label": "MOPEN",
    },
    "PMC": {
        "color": "#bc8cff",
        "dash": "4 4",
        "short_label": "Prior month close",
        "long_label": "prior month close",
        "chart_label": "PMC",
    },
    "PML": {
        "color": "#26a641",
        "dash": "7 4",
        "short_label": "Prior month low",
        "long_label": "prior month low",
        "chart_label": "PML",
    },
}


def fetch_monthly_context_rows(
    symbol: str = "SPY",
    *,
    interval: str = "1d",
    range_: str = "2y",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, source = fetch_yahoo_daily_bars(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    return rows, source


def derive_monthly_levels(
    daily_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _group_recent_months(daily_rows, lookback_months=2)
    ordered_months = list(grouped.items())
    if not ordered_months:
        return []

    levels = []
    current_month_key, current_month_rows = ordered_months[-1]
    if current_month_rows:
        levels.append(
            _monthly_level(
                "MOPEN", float(current_month_rows[0]["open"]), current_month_key
            )
        )

    if len(ordered_months) >= 2:
        previous_month_key, previous_month_rows = ordered_months[-2]
        highs = [float(row["high"]) for row in previous_month_rows]
        lows = [float(row["low"]) for row in previous_month_rows]
        previous_close = float(previous_month_rows[-1]["close"])
        levels.extend(
            [
                _monthly_level("PMH", max(highs), previous_month_key),
                _monthly_level("PMC", previous_close, previous_month_key),
                _monthly_level("PML", min(lows), previous_month_key),
            ]
        )

    return levels


def summarize_monthly_context(
    daily_rows: list[dict[str, Any]],
    monthly_levels: list[dict[str, Any]],
    *,
    spot: float,
    symbol: str = "SPY",
    lookback_months: int = 6,
) -> dict[str, Any]:
    grouped = _group_recent_months(daily_rows, lookback_months=lookback_months)
    ordered_months = list(grouped.items())
    closes = [float(row["close"]) for _month, rows in ordered_months for row in rows]
    if not closes:
        return {
            "symbol": symbol,
            "lookback_months": 0,
            "panel_note": "Bottom chart = monthly structure map zoomed on the active month. Distant rails stay in the chips/text.",
            "headline": "Monthly structure map unavailable",
            "detail": "No monthly context data came back.",
            "kind": "info",
            "legend": [],
        }

    range_low = min(closes)
    range_high = max(closes)
    range_span = max(range_high - range_low, 1e-9)
    range_position_pct = max(0, min(100, round((spot - range_low) / range_span * 100)))
    levels_by_name = {
        level["name"]: float(level["price"])
        for level in monthly_levels
        if isinstance(level.get("price"), (int, float))
    }
    nearest = _nearest_monthly_level(spot, monthly_levels)
    headline, detail, kind = _monthly_structure_read(
        spot,
        levels_by_name,
        nearest,
        range_low=range_low,
        range_high=range_high,
        range_position_pct=range_position_pct,
        month_count=len(ordered_months),
    )
    legend = [
        {
            "name": name,
            "label": MONTHLY_LEVEL_META[name]["short_label"],
            "price": levels_by_name[name],
            "color": MONTHLY_LEVEL_META[name]["color"],
        }
        for name in ["PMH", "MOPEN", "PMC", "PML"]
        if name in levels_by_name
    ]
    return {
        "symbol": symbol,
        "lookback_months": len(ordered_months),
        "panel_note": (
            f"Bottom chart = {len(ordered_months)}-month structure map built from prior month rails + current month open, zoomed on the active month. Distant rails stay in the chips/text."
        ),
        "headline": headline,
        "detail": detail,
        "kind": kind,
        "legend": legend,
        "range_position_pct": range_position_pct,
        "range_low": round(range_low, 2),
        "range_high": round(range_high, 2),
    }


def build_monthly_context_svg(
    daily_rows: list[dict[str, Any]],
    monthly_levels: list[dict[str, Any]],
    *,
    symbol: str = "SPY",
    lookback_months: int = 6,
) -> str:
    if not daily_rows:
        return _empty_monthly_context_svg(symbol)

    grouped = _group_recent_months(daily_rows, lookback_months=lookback_months)
    ordered_months = list(grouped.items())
    flat_rows = [row for _month, rows in ordered_months for row in rows]
    closes = [float(row["close"]) for row in flat_rows]
    current_month_rows = ordered_months[-1][1]
    prior_month_rows = ordered_months[-2][1] if len(ordered_months) >= 2 else []
    recent_focus = [
        *[float(row["close"]) for row in prior_month_rows[-8:]],
        *[float(row["close"]) for row in current_month_rows],
    ]
    lo, hi, visible_level_names = build_focus_window(
        closes,
        monthly_levels,
        anchor=closes[-1],
        recent_values=recent_focus,
        min_span_pct=2.2,
        nearby_level_pct=1.1,
        outer_level_pct=1.9,
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
        f'<text x="{PAD_L}" y="22" fill="#e6edf3" font-size="15" font-weight="bold">{symbol} 6-month structure</text>',
        f'<text x="{PAD_L + PLOT_W}" y="22" fill="#7d8590" font-size="11" text-anchor="end">current month = bright blue • older months fade</text>',
    ]

    current_month_start = len(flat_rows) - len(current_month_rows)
    current_month_x = px(current_month_start)
    current_month_w = px(len(flat_rows) - 1) - current_month_x
    parts.append(
        f'<rect x="{current_month_x:.1f}" y="{PAD_T}" width="{current_month_w:.1f}" height="{PLOT_H:.1f}" fill="#58a6ff" opacity="0.05" rx="6"/>'
    )
    parts.append(
        f'<text x="{current_month_x + 8:.1f}" y="{PAD_T + 14:.1f}" fill="#58a6ff" font-size="10" font-weight="bold">THIS MONTH</text>'
    )

    for step in range(5):
        value = lo + span * step / 4
        grid_y = py(value)
        parts.append(
            f'<line x1="{PAD_L}" y1="{grid_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{grid_y:.1f}" stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 6}" y="{grid_y + 4:.1f}" fill="#7d8590" font-size="10" text-anchor="end">${value:.2f}</text>'
        )

    start = 0
    total_months = len(ordered_months)
    for index, (month_key, month_rows) in enumerate(ordered_months):
        end = start + len(month_rows) - 1
        month_x = px(start)
        parts.append(
            f'<line x1="{month_x:.1f}" y1="{PAD_T}" x2="{month_x:.1f}" y2="{PAD_T + PLOT_H}" stroke="#161b22" stroke-width="1"/>'
        )
        mid = px((start + end) / 2)
        parts.append(
            f'<text x="{mid:.1f}" y="{H - 10}" fill="#7d8590" font-size="10" text-anchor="middle">{_short_month_label(month_key)}</text>'
        )
        pts = " ".join(
            f"{px(start + offset):.1f},{py(float(row['close'])):.1f}"
            for offset, row in enumerate(month_rows)
        )
        age_from_latest = total_months - 1 - index
        if age_from_latest == 0:
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="#58a6ff" stroke-width="5.2" opacity="0.14"/>'
            )
            color = "#58a6ff"
            width = 2.6
            opacity = 1.0
        elif age_from_latest == 1:
            color = "#8bb8ff"
            width = 1.9
            opacity = 0.72
        else:
            fade_rank = index / max(total_months - 2, 1)
            shade = int(86 + 40 * fade_rank)
            color = f"rgb({shade // 2},{shade - 8},{min(255, shade + 40)})"
            width = 1.15
            opacity = 0.16 + 0.14 * fade_rank
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity:.2f}"/>'
        )
        start = end + 1

    for level in monthly_levels:
        if level["name"] not in visible_level_names:
            continue
        meta = MONTHLY_LEVEL_META.get(level["name"], {})
        color = meta.get("color", "#7d8590")
        dash = meta.get("dash", "3 5")
        chart_label = meta.get("chart_label", level["name"])
        level_y = py(float(level["price"]))
        parts.append(
            f'<line x1="{PAD_L}" y1="{level_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{level_y:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}" opacity="0.95"/>'
        )
        parts.append(
            f'<text x="{PAD_L + PLOT_W + 6}" y="{level_y + 4:.1f}" fill="{color}" font-size="10" font-weight="bold">{chart_label} {float(level["price"]):.2f}</text>'
        )

    last_close = closes[-1]
    last_y = py(last_close)
    parts.append(
        f'<circle cx="{px(len(flat_rows) - 1):.1f}" cy="{last_y:.1f}" r="3.5" fill="#ffd33d"/>'
    )
    parts.append(
        f'<text x="{px(len(flat_rows) - 1) + 8:.1f}" y="{last_y + 4:.1f}" fill="#ffd33d" font-size="11" font-weight="bold">NOW ${last_close:.2f}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _group_recent_months(
    daily_rows: list[dict[str, Any]],
    *,
    lookback_months: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        month_key = str(row["date"])[:7]
        grouped[month_key].append(row)
    ordered_keys = sorted(grouped)[-lookback_months:]
    return {key: grouped[key] for key in ordered_keys}


def _monthly_structure_read(
    spot: float,
    levels_by_name: dict[str, float],
    nearest: dict[str, Any] | None,
    *,
    range_low: float,
    range_high: float,
    range_position_pct: int,
    month_count: int,
) -> tuple[str, str, str]:
    pmh = levels_by_name.get("PMH")
    mopen = levels_by_name.get("MOPEN")
    pmc = levels_by_name.get("PMC")
    pml = levels_by_name.get("PML")
    nearest_text = _nearest_text(nearest)
    range_text = (
        f"{month_count}-month range ${range_low:.2f} -> ${range_high:.2f} "
        f"({range_position_pct}% up the range)."
    )

    if pmh is not None and spot >= pmh:
        return (
            "Pressing through the prior month high",
            f"Spot ${spot:.2f} is through PMH ${pmh:.2f}. {nearest_text} {range_text}",
            "ok",
        )
    if mopen is not None and pmc is not None and spot >= max(mopen, pmc):
        return (
            "Holding above monthly value inside the upper month band",
            f"Spot ${spot:.2f} is above MOPEN ${mopen:.2f} and PMC ${pmc:.2f}. {nearest_text} {range_text}",
            "ok",
        )
    if (
        mopen is not None
        and pmc is not None
        and min(mopen, pmc) <= spot < max(mopen, pmc)
    ):
        return (
            "Chopping around the monthly value pocket",
            f"Spot ${spot:.2f} is between MOPEN ${mopen:.2f} and PMC ${pmc:.2f}. {nearest_text} {range_text}",
            "info",
        )
    if pml is not None and spot <= pml:
        return (
            "Back at the prior month low",
            f"Spot ${spot:.2f} is testing or through PML ${pml:.2f}. {nearest_text} {range_text}",
            "bad",
        )
    if pml is not None:
        return (
            "Below monthly value, but still above the prior month low",
            f"Spot ${spot:.2f} is below the value pocket and above PML ${pml:.2f}. {nearest_text} {range_text}",
            "warn",
        )
    return (
        "Reading spot against prior month rails",
        f"{nearest_text} {range_text}",
        "info",
    )


def _nearest_monthly_level(
    spot: float,
    monthly_levels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not monthly_levels:
        return None
    nearest = min(monthly_levels, key=lambda level: abs(float(level["price"]) - spot))
    meta = MONTHLY_LEVEL_META.get(nearest["name"], {})
    distance_pct = abs(float(nearest["price"]) - spot) / max(spot, 1e-9) * 100
    return {
        "name": nearest["name"],
        "label": meta.get("long_label", nearest["name"]),
        "price": float(nearest["price"]),
        "distance_pct": distance_pct,
    }


def _nearest_text(nearest: dict[str, Any] | None) -> str:
    if not nearest:
        return "No monthly rail was isolated cleanly."
    return (
        f"Nearest monthly rail: {nearest['label']} ${nearest['price']:.2f} "
        f"({nearest['distance_pct']:.2f}% away)."
    )


def _monthly_level(name: str, price: float, month_key: str) -> dict[str, Any]:
    return {
        "name": name,
        "price": round(float(price), 2),
        "month": month_key,
    }


def _short_month_label(month_key: str) -> str:
    month = dt.date.fromisoformat(f"{month_key}-01")
    return month.strftime("%b")


def _empty_monthly_context_svg(symbol: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="monospace">'
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>'
        f'<text x="{PAD_L}" y="36" fill="#e6edf3" font-size="15" font-weight="bold">{symbol} monthly context</text>'
        f'<text x="{PAD_L}" y="64" fill="#7d8590" font-size="12">No monthly context data available.</text>'
        "</svg>"
    )


__all__ = [
    "build_monthly_context_svg",
    "derive_monthly_levels",
    "fetch_monthly_context_rows",
    "summarize_monthly_context",
]
