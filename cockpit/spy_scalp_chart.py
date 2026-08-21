"""SVG entry map for the focused SPY scalp dashboard."""

from __future__ import annotations

import html
from collections.abc import Iterable

Row = tuple[int, float, float, float, float, int]

WIDTH = 1120
HEIGHT = 520
PAD_L = 58
PAD_R = 128
PAD_T = 34
PAD_B = 54
PLOT_W = WIDTH - PAD_L - PAD_R
PLOT_H = HEIGHT - PAD_T - PAD_B


class Scale:
    def __init__(self, rows: list[Row], levels: list[float]) -> None:
        prices = [price for row in rows for price in (row[2], row[3])] + levels
        lo = min(prices) if prices else 0.0
        hi = max(prices) if prices else 1.0
        pad = max((hi - lo) * 0.10, 0.20)
        self.lo = lo - pad
        self.hi = hi + pad
        self.count = max(len(rows) - 1, 1)

    def x(self, index: int) -> float:
        return PAD_L + (index / self.count) * PLOT_W

    def y(self, price: float) -> float:
        span = max(self.hi - self.lo, 1e-9)
        return PAD_T + (self.hi - price) / span * PLOT_H


def render_spy_scalp_chart_svg(rows: list[Row], packet: dict) -> str:
    """Render a compact scalp-entry map with levels/channels."""
    visible = _regular_rows(rows)[-180:]
    if len(visible) < 2:
        return _empty_svg("Need more regular-session bars for scalp chart")

    or_range = packet.get("opening_range") or {}
    indicators = packet.get("indicators") or {}
    levels = _chart_levels(or_range, indicators, packet)
    scale = Scale(visible, levels)
    closes = [row[4] for row in visible]
    ema9 = _ema_series(closes, 9)
    ema20 = _ema_series(closes, 20)
    channel_hi, channel_lo = _rolling_channel(visible, lookback=20)
    tight_hi, tight_lo = _rolling_channel(visible, lookback=8)

    parts = [_svg_open(), _background(), _grid(scale)]
    parts.append(
        _channel_area(scale, channel_hi, channel_lo, "#38bdf830", "20-bar channel")
    )
    parts.append(_channel_line(scale, channel_hi, "#38bdf8", "20ch high"))
    parts.append(_channel_line(scale, channel_lo, "#38bdf8", "20ch low"))
    parts.append(_channel_line(scale, tight_hi, "#fbbf24", "8ch high"))
    parts.append(_channel_line(scale, tight_lo, "#fbbf24", "8ch low"))
    parts.append(_candles(scale, visible))
    parts.append(_series_line(scale, ema20, "#a78bfa", "EMA20"))
    parts.append(_series_line(scale, ema9, "#f97316", "EMA9"))
    parts.extend(_level_lines(scale, or_range, indicators, packet))
    parts.append(_spot_badge(scale, packet))
    parts.append(_trigger_annotation(scale, packet, visible))
    parts.append(_legend(packet))
    parts.append("</svg>")
    return "\n".join(part for part in parts if part)


def _regular_rows(rows: Iterable[Row]) -> list[Row]:
    return [row for row in rows if 0 <= int(row[0]) < 390]


def _chart_levels(or_range: dict, indicators: dict, packet: dict) -> list[float]:
    levels: list[float] = [float(packet.get("spot") or 0)]
    for key in ("high", "low"):
        value = or_range.get(key)
        if isinstance(value, (int, float)):
            levels.append(float(value))
    vwap = indicators.get("vwap")
    if isinstance(vwap, (int, float)):
        levels.append(float(vwap))
    return [level for level in levels if level > 0]


def _ema_series(values: list[float], period: int) -> list[float | None]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    ema = values[0]
    out: list[float | None] = []
    for index, value in enumerate(values):
        ema = value * alpha + ema * (1 - alpha)
        out.append(ema if index >= period - 1 else None)
    return out


def _rolling_channel(
    rows: list[Row], lookback: int
) -> tuple[list[float | None], list[float | None]]:
    highs: list[float | None] = []
    lows: list[float | None] = []
    for index in range(len(rows)):
        if index < lookback - 1:
            highs.append(None)
            lows.append(None)
            continue
        window = rows[index - lookback + 1 : index + 1]
        highs.append(max(row[2] for row in window))
        lows.append(min(row[3] for row in window))
    return highs, lows


def _svg_open() -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        'role="img" aria-label="SPY scalp entry map">'
    )


def _background() -> str:
    return f"""
<rect width="{WIDTH}" height="{HEIGHT}" rx="18" fill="#07101a"/>
<rect x="{PAD_L}" y="{PAD_T}" width="{PLOT_W}" height="{PLOT_H}" rx="12" fill="#0b1320" stroke="#243244"/>
<text x="{PAD_L}" y="23" fill="#e5f2ff" font-size="18" font-weight="800">SPY scalp entry map</text>
<text x="{WIDTH - PAD_R}" y="23" fill="#8ea0b5" font-size="12">OR15 + VWAP + EMA + channels</text>
"""


def _grid(scale: Scale) -> str:
    lines = []
    for step in range(5):
        y = PAD_T + step * PLOT_H / 4
        price = scale.hi - step * (scale.hi - scale.lo) / 4
        lines.append(
            f'<line x1="{PAD_L}" x2="{PAD_L + PLOT_W}" y1="{y:.1f}" y2="{y:.1f}" stroke="#182536"/>'
        )
        lines.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#7b8ba1" font-size="11">{price:.2f}</text>'
        )
    return "\n".join(lines)


def _candles(scale: Scale, rows: list[Row]) -> str:
    width = max(min(PLOT_W / len(rows) * 0.65, 5.0), 1.2)
    parts = []
    for index, row in enumerate(rows):
        _minute, open_, high, low, close, _volume = row
        color = "#34d399" if close >= open_ else "#fb7185"
        x = scale.x(index)
        high_y = scale.y(high)
        low_y = scale.y(low)
        open_y = scale.y(open_)
        close_y = scale.y(close)
        top = min(open_y, close_y)
        body_h = max(abs(close_y - open_y), 1.2)
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{high_y:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="1" opacity="0.75"/>'
        )
        parts.append(
            f'<rect x="{x - width / 2:.1f}" y="{top:.1f}" width="{width:.1f}" height="{body_h:.1f}" rx="1" fill="{color}" opacity="0.88"/>'
        )
    return "\n".join(parts)


def _series_line(
    scale: Scale, values: list[float | None], color: str, label: str
) -> str:
    path = _path_from_values(scale, values)
    if not path:
        return ""
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/><text x="{PAD_L + 8}" y="{scale.y(_last(values)) - 6:.1f}" fill="{color}" font-size="11">{label}</text>'


def _channel_line(
    scale: Scale, values: list[float | None], color: str, label: str
) -> str:
    path = _path_from_values(scale, values)
    if not path:
        return ""
    y = scale.y(_last(values))
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4" stroke-dasharray="5 5" opacity="0.82"/><text x="{PAD_L + PLOT_W + 8}" y="{y + 4:.1f}" fill="{color}" font-size="11">{label}</text>'


def _channel_area(
    scale: Scale,
    highs: list[float | None],
    lows: list[float | None],
    fill: str,
    _label: str,
) -> str:
    points_hi = [(i, v) for i, v in enumerate(highs) if v is not None]
    points_lo = [(i, v) for i, v in enumerate(lows) if v is not None]
    if not points_hi or not points_lo:
        return ""
    forward = " ".join(f"{scale.x(i):.1f},{scale.y(v):.1f}" for i, v in points_hi)
    backward = " ".join(
        f"{scale.x(i):.1f},{scale.y(v):.1f}" for i, v in reversed(points_lo)
    )
    return f'<polygon points="{forward} {backward}" fill="{fill}" stroke="none"/>'


def _level_lines(
    scale: Scale, or_range: dict, indicators: dict, packet: dict
) -> list[str]:
    out: list[str] = []
    if isinstance(or_range.get("high"), (int, float)):
        out.append(_price_line(scale, float(or_range["high"]), "ORH", "#22c55e", "8 4"))
    if isinstance(or_range.get("low"), (int, float)):
        out.append(_price_line(scale, float(or_range["low"]), "ORL", "#ef4444", "8 4"))
    if isinstance(indicators.get("vwap"), (int, float)):
        out.append(_price_line(scale, float(indicators["vwap"]), "VWAP", "#eab308", ""))
    spot = packet.get("spot")
    if isinstance(spot, (int, float)):
        out.append(_price_line(scale, float(spot), "SPOT", "#e5f2ff", "2 5"))
    return out


def _price_line(scale: Scale, price: float, label: str, color: str, dash: str) -> str:
    y = scale.y(price)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{PAD_L}" x2="{PAD_L + PLOT_W}" y1="{y:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="1.8"{dash_attr}/>'
        f'<text x="{PAD_L + PLOT_W + 8}" y="{y + 4:.1f}" fill="{color}" font-size="12" font-weight="700">{label} {price:.2f}</text>'
    )


def _spot_badge(scale: Scale, packet: dict) -> str:
    spot = packet.get("spot")
    if not isinstance(spot, (int, float)):
        return ""
    y = scale.y(float(spot))
    label = html.escape(str(packet.get("bias") or "NEUTRAL"))
    status = html.escape(str(packet.get("status") or ""))
    return f"""
<rect x="{PAD_L + PLOT_W - 188}" y="{y - 28:.1f}" width="180" height="24" rx="12" fill="#111827" stroke="#e5f2ff"/>
<text x="{PAD_L + PLOT_W - 98}" y="{y - 11:.1f}" text-anchor="middle" fill="#e5f2ff" font-size="11" font-weight="800">{label} | {status}</text>
"""


def _trigger_annotation(scale: Scale, packet: dict, rows: list[Row]) -> str:
    trigger = packet.get("trigger") or {}
    or_range = packet.get("opening_range") or {}
    bias = trigger.get("bias") or packet.get("bias")
    level_key = "high" if bias == "CALLS" else "low" if bias == "PUTS" else "high"
    level = or_range.get(level_key)
    if not isinstance(level, (int, float)):
        return ""
    x = scale.x(len(rows) - 1)
    y = scale.y(float(level))
    color = "#22c55e" if bias == "CALLS" else "#ef4444" if bias == "PUTS" else "#94a3b8"
    text = (
        "entry needs hold above ORH"
        if bias == "CALLS"
        else "entry needs hold below ORL"
    )
    if trigger.get("state") == "armed":
        text = "trigger armed: use pullback/limit only"
    return f"""
<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#07101a" stroke-width="2"/>
<path d="M {x - 120:.1f} {y - 34:.1f} L {x - 12:.1f} {y - 6:.1f}" stroke="{color}" stroke-width="1.5" fill="none"/>
<rect x="{x - 286:.1f}" y="{y - 56:.1f}" width="164" height="28" rx="10" fill="#111827" stroke="{color}"/>
<text x="{x - 204:.1f}" y="{y - 38:.1f}" text-anchor="middle" fill="{color}" font-size="11" font-weight="800">{html.escape(text)}</text>
"""


def _legend(packet: dict) -> str:
    trend = html.escape(str((packet.get("trend") or {}).get("reason", "")))
    trigger = html.escape(str((packet.get("trigger") or {}).get("reason", "")))
    return f"""
<rect x="{PAD_L}" y="{HEIGHT - 40}" width="{PLOT_W}" height="28" rx="10" fill="#0f172a" stroke="#263244"/>
<text x="{PAD_L + 12}" y="{HEIGHT - 22}" fill="#cbd5e1" font-size="12">Trend: {trend}</text>
<text x="{PAD_L + 480}" y="{HEIGHT - 22}" fill="#cbd5e1" font-size="12">Trigger: {trigger}</text>
"""


def _path_from_values(scale: Scale, values: list[float | None]) -> str:
    commands: list[str] = []
    started = False
    for index, value in enumerate(values):
        if value is None:
            started = False
            continue
        cmd = "M" if not started else "L"
        commands.append(f"{cmd} {scale.x(index):.1f} {scale.y(value):.1f}")
        started = True
    return " ".join(commands)


def _last(values: list[float | None]) -> float:
    for value in reversed(values):
        if value is not None:
            return value
    return 0.0


def _empty_svg(message: str) -> str:
    safe = html.escape(message)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="{WIDTH}" height="{HEIGHT}" fill="#07101a"/>
<text x="40" y="80" fill="#e5f2ff" font-size="22">{safe}</text>
</svg>"""
