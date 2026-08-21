from __future__ import annotations

import datetime as dt
import html

from gate_workflows import primary_trade_setup

W, H = 1000, 576
PAD_L, PAD_R, PAD_T, PAD_B = 42, 64, 18, 190
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B
CHANNEL_COLOR = "#bc8cff"
RIGHT_LABEL_X = W - 8
RIGHT_LABEL_FONT_SIZE = 9
LEVEL_STATE_ORDER = ("ORH", "ORL", "PDH", "PDL", "PDC")


def _channel_overlay(volatility_structure):
    state = volatility_structure or {}
    channel_high = state.get("channel_high")
    channel_low = state.get("channel_low")
    if not isinstance(channel_high, (int, float)) or not isinstance(
        channel_low, (int, float)
    ):
        return None
    if channel_high <= channel_low:
        return None
    return float(channel_low), float(channel_high)


def _channel_logic(pa, volatility_structure, spot):
    channel = _channel_overlay(volatility_structure)
    if not channel:
        return None
    channel_low, channel_high = channel
    width = channel_high - channel_low
    if width <= 0:
        return None
    position = (float(spot) - channel_low) / width
    channel_pct = volatility_structure.get("channel_pct")
    slope = volatility_structure.get("channel_slope_pct")
    if spot > channel_high:
        state = "BREAKING ABOVE CHANNEL"
        color = "#26a641"
    elif spot < channel_low:
        state = "BREAKING BELOW CHANNEL"
        color = "#f85149"
    elif position >= 0.75:
        state = "PRESSING CHANNEL HIGH"
        color = "#d29922"
    elif position <= 0.25:
        state = "PRESSING CHANNEL LOW"
        color = "#d29922"
    else:
        state = "INSIDE CHANNEL"
        color = CHANNEL_COLOR
    return {
        "state": state,
        "color": color,
        "low": channel_low,
        "high": channel_high,
        "mid": (channel_high + channel_low) / 2,
        "position_pct": max(0.0, min(100.0, position * 100.0)),
        "width_pct": channel_pct if isinstance(channel_pct, (int, float)) else None,
        "slope_pct": slope if isinstance(slope, (int, float)) else None,
        "structure": str(volatility_structure.get("structure_state") or "unknown"),
        "volatility": str(volatility_structure.get("volatility_state") or "unknown"),
        "vs_vwap": pa.get("vs_vwap"),
    }


def _render_channel_logic_background(logic):
    if not logic:
        return []
    x = PAD_L + PLOT_W - 330
    y = PAD_T + 18
    width_txt = (
        f"width {logic['width_pct']:.3f}%"
        if isinstance(logic.get("width_pct"), (int, float))
        else "width n/a"
    )
    slope_txt = (
        f"slope {logic['slope_pct']:+.3f}%"
        if isinstance(logic.get("slope_pct"), (int, float))
        else "slope n/a"
    )
    return [
        '<g opacity="0.30">',
        f'<rect x="{x}" y="{y}" width="320" height="66" fill="#0d1117" opacity="0.28" rx="10" stroke="{logic["color"]}" stroke-opacity="0.55"/>',
        f'<text x="{x + 12}" y="{y + 17}" fill="{CHANNEL_COLOR}" font-size="11" font-weight="bold">CHANNEL LOGIC</text>',
        f'<text x="{x + 12}" y="{y + 35}" fill="{logic["color"]}" font-size="14" font-weight="bold">{html.escape(logic["state"])}</text>',
        f'<text x="{x + 12}" y="{y + 51}" fill="#adbac7" font-size="10">pos {logic["position_pct"]:.0f}% • {width_txt} • {slope_txt}</text>',
        f'<text x="{x + 12}" y="{y + 63}" fill="#7d8590" font-size="9">{html.escape(logic["structure"])} / {html.escape(logic["volatility"])} • {logic["low"]:.2f}-{logic["high"]:.2f}</text>',
        "</g>",
    ]


def _level_state_accent(event_state):
    if event_state in {"failed_break_reclaimed", "holding_above_support"}:
        return "#26a641"
    if event_state in {
        "failed_break_rejected",
        "holding_below_resistance",
        "lost_support",
    }:
        return "#f85149"
    if str(event_state).startswith("testing"):
        return "#d29922"
    if event_state in {"accepted_above_resistance", "accepted_above_reference"}:
        return "#58a6ff"
    return "#7d8590"


def _level_state_label(event_state):
    labels = {
        "failed_break_reclaimed": "FAIL RECLAIM",
        "failed_break_rejected": "FAIL REJECT",
        "holding_above_support": "HOLD SUPPORT",
        "holding_below_resistance": "HOLD RESIST",
        "accepted_above_resistance": "ACCEPT > R",
        "accepted_above_reference": "ACCEPT > REF",
        "accepted_below_reference": "ACCEPT < REF",
        "testing_support": "TEST SUPPORT",
        "testing_resistance": "TEST RESIST",
        "testing_reference": "TEST REF",
        "lost_support": "LOST SUPPORT",
    }
    return labels.get(
        str(event_state), str(event_state or "UNKNOWN").replace("_", " ").upper()
    )


def _level_state_strip(level_states):
    states = level_states or {}
    if not states:
        return []
    x = PAD_L + 12
    y = PAD_T + 8
    row_h = 24
    out = [
        f'<rect x="{x - 8}" y="{y - 18}" width="196" height="{len(LEVEL_STATE_ORDER) * row_h + 24}" fill="#0d1117" opacity="0.82" rx="8"/>',
        f'<text x="{x}" y="{y - 4}" fill="#bc8cff" font-size="11" font-weight="bold">LEVEL STATES</text>',
    ]
    for idx, name in enumerate(LEVEL_STATE_ORDER):
        state = states.get(name)
        if not isinstance(state, dict):
            continue
        event_state = str(state.get("event_state") or "unknown")
        accent = _level_state_accent(event_state)
        row_y = y + idx * row_h
        out.append(
            f'<rect x="{x}" y="{row_y}" width="180" height="18" rx="5" fill="#161b22" stroke="#30363d"/>'
        )
        out.append(
            f'<text x="{x + 6}" y="{row_y + 12}" fill="{accent}" font-size="10" font-weight="bold">{name}</text>'
        )
        out.append(
            f'<text x="{x + 52}" y="{row_y + 12}" fill="#adbac7" font-size="10">{_level_state_label(event_state)}</text>'
        )
    return out


def _parse_timestamp(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _marker_session_minute(marker_ts):
    parsed = _parse_timestamp(marker_ts)
    if parsed is None:
        return None
    minute_of_day = parsed.hour * 60 + parsed.minute
    return minute_of_day - 570  # 09:30 ET regular-session open


def _row_session_minute(row):
    if not row:
        return None
    value = row[0]
    if isinstance(value, (int, float)):
        return float(value)
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return float(parsed.hour * 60 + parsed.minute - 570)


def _nearest_bar_index(rows, marker_ts):
    target = _marker_session_minute(marker_ts)
    if target is None:
        return None
    best_idx = None
    best_delta = None
    for idx, row in enumerate(rows):
        value = _row_session_minute(row)
        if value is None:
            continue
        delta = abs(value - target)
        if best_delta is None or delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx


def _short_marker_label(event_type, status="confirmed"):
    status_text = str(status or "confirmed").lower()
    if status_text == "confirmed":
        suffix = "CONFIRMED"
    elif status_text == "observed":
        suffix = "OBSERVED"
    elif status_text == "candidate":
        suffix = "CANDIDATE"
    else:
        suffix = status_text.upper()
    labels = {
        "DOWNSIDE EXHAUSTION": f"DOWNSIDE EXHAUSTION {suffix}",
        "UPSIDE EXHAUSTION": f"UPSIDE EXHAUSTION {suffix}",
        "FAILED BREAKDOWN": f"FAILED BREAKDOWN {suffix}",
        "FAILED BREAKOUT": f"FAILED BREAKOUT {suffix}",
        "EXHAUSTION -> RUNNER HANDOFF": f"RUNNER HANDOFF {suffix}",
    }
    return labels.get(str(event_type), f"{str(event_type or 'SETUP')} {suffix}")


def _quote_source_label(source):
    labels = {
        "cboe_bid_ask_midpoint": "CBOE MID",
        "cboe_current_price": "CBOE",
        "cnbc_last_price": "CNBC",
        "yahoo_extended_session_price": "YHOO EXT",
        "yahoo_regular_market_price": "YHOO",
        "yahoo_completed_bar_close": "BAR",
    }
    return labels.get(str(source or ""), "QUOTE")


def chart_svg(
    rows,
    pa,
    levels=None,
    setups=None,
    volatility_structure=None,
    level_states=None,
    setup_markers=None,
    show_signal_overlays=False,
):
    closes = [b[4] for b in rows]
    n = len(closes)
    display_spot = pa.get("display_spot") or pa.get("spot")
    quote_label = _quote_source_label(pa.get("spot_source"))
    has_display_spot = isinstance(display_spot, (int, float))
    levels = levels or {}
    setups = setups or []
    channel_overlay = _channel_overlay(volatility_structure)
    values = [*closes]
    if has_display_spot:
        values.append(float(display_spot))
    if channel_overlay:
        values.extend(channel_overlay)
    if isinstance(pa.get("vwap"), (int, float)):
        values.append(float(pa["vwap"]))
    values.extend(
        float(price) for price in levels.values() if isinstance(price, (int, float))
    )
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pad = span * 0.08
    lo -= pad
    hi += pad
    span = hi - lo

    def x(i):
        return PAD_L + i / max(n - 1, 1) * PLOT_W

    def y(p):
        return PAD_T + (1 - (p - lo) / span) * PLOT_H

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
    ]
    if show_signal_overlays:
        s.extend(
            _render_channel_logic_background(
                _channel_logic(
                    pa,
                    volatility_structure or {},
                    float(display_spot) if has_display_spot else closes[-1],
                )
            )
        )

    for step in range(5):
        value = lo + span * step / 4
        gy = y(value)
        s.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L + PLOT_W}" y2="{gy:.1f}" '
            f'stroke="#1f2630" stroke-width="1"/>'
        )
        s.append(
            f'<text x="{PAD_L - 6}" y="{gy + 4:.1f}" fill="#7d8590" font-size="11" text-anchor="end">'
            f"${value:.2f}</text>"
        )

    level_styles = {
        "ORH": ("#58a6ff", "4 3"),
        "ORL": ("#58a6ff", "4 3"),
        "PDH": ("#bc8cff", "2 4"),
        "PDL": ("#bc8cff", "2 4"),
        "PDC": ("#7d8590", "1 4"),
    }
    active_level_name = None
    active_level_price = None
    if show_signal_overlays:
        active_setup = primary_trade_setup(setups)
        active_level_name = active_setup.get("level_name")
        active_level_price = active_setup.get("level_price")
        if active_setup.get("tag") in {"FAILED BREAKDOWN", "FAILED BREAKOUT"}:
            trigger_price = active_setup.get("trigger_price")
            if isinstance(trigger_price, (int, float)):
                levels = {**levels, "TRIGGER": trigger_price}
                level_styles["TRIGGER"] = ("#f85149", "1 3")

    for name, price in levels.items():
        if not isinstance(price, (int, float)):
            continue
        color, dash = level_styles.get(name, ("#30363d", "2 6"))
        stroke_width = (
            2.2 if name == active_level_name or price == active_level_price else 1.35
        )
        ly = y(price)
        s.append(
            f'<line x1="{PAD_L}" y1="{ly:.1f}" x2="{PAD_L + PLOT_W}" y2="{ly:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="{dash}" opacity="0.95"/>'
        )
        s.append(
            f'<text x="{PAD_L + 6}" y="{ly - 6:.1f}" fill="{color}" '
            f'font-size="12" font-weight="bold">{name} {price:.2f}</text>'
        )

    if channel_overlay:
        channel_low, channel_high = channel_overlay
        channel_y = y(channel_high)
        channel_height = max(y(channel_low) - channel_y, 1.0)
        s.append(
            f'<rect x="{PAD_L}" y="{channel_y:.1f}" width="{PLOT_W:.1f}" '
            f'height="{channel_height:.1f}" fill="{CHANNEL_COLOR}" opacity="0.08"/>'
        )
        channel_mid = (channel_high + channel_low) / 2
        s.append(
            f'<line x1="{PAD_L}" y1="{y(channel_mid):.1f}" x2="{PAD_L + PLOT_W}" y2="{y(channel_mid):.1f}" '
            f'stroke="{CHANNEL_COLOR}" stroke-width="0.9" stroke-dasharray="2 5" opacity="0.55"/>'
        )
        s.append(
            f'<text x="{PAD_L + PLOT_W - 8}" y="{y(channel_mid) - 5:.1f}" fill="{CHANNEL_COLOR}" '
            f'font-size="10" text-anchor="end">CHANNEL MID {channel_mid:.2f}</text>'
        )
        for label, price, anchor_y in (
            ("CHANNEL HI", channel_high, y(channel_high)),
            ("CHANNEL LO", channel_low, y(channel_low)),
        ):
            s.append(
                f'<line x1="{PAD_L}" y1="{anchor_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{anchor_y:.1f}" '
                f'stroke="{CHANNEL_COLOR}" stroke-width="1.2" stroke-dasharray="6 4" opacity="0.95"/>'
            )
            s.append(
                f'<text x="{PAD_L + 6}" y="{anchor_y - 6:.1f}" fill="{CHANNEL_COLOR}" '
                f'font-size="12" font-weight="bold">{label} {price:.2f}</text>'
            )

    vy = y(pa["vwap"])
    s.append(
        f'<line x1="{PAD_L}" y1="{vy:.1f}" x2="{PAD_L + PLOT_W}" y2="{vy:.1f}" '
        f'stroke="#ffd33d" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    s.append(
        f'<text x="{RIGHT_LABEL_X}" y="{vy + 4:.1f}" fill="#ffd33d" '
        f'font-size="{RIGHT_LABEL_FONT_SIZE}" font-weight="bold" text-anchor="end">'
        f"VWAP {pa['vwap']:.2f}</text>"
    )

    col = "#26a641" if pa["vs_vwap"] >= 0 else "#f85149"
    pts = " ".join(f"{x(i):.1f},{y(closes[i]):.1f}" for i in range(n))
    s.append(
        f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>'
    )
    s.append(
        f'<circle cx="{x(n - 1):.1f}" cy="{y(closes[-1]):.1f}" r="4.4" fill="#58a6ff"/>'
    )
    s.append(
        f'<text x="{RIGHT_LABEL_X}" y="{y(closes[-1]) + 4:.1f}" fill="#58a6ff" '
        f'font-size="{RIGHT_LABEL_FONT_SIZE}" font-weight="bold" text-anchor="end">'
        f"BAR {closes[-1]:.2f}</text>"
    )
    if has_display_spot:
        quote_y = y(float(display_spot))
        quote_differs = abs(float(display_spot) - closes[-1]) >= 0.005
        quote_opacity = "0.95" if quote_differs else "0.55"
        s.append(
            f'<line x1="{PAD_L}" y1="{quote_y:.1f}" x2="{PAD_L + PLOT_W}" y2="{quote_y:.1f}" '
            f'stroke="#39c5cf" stroke-width="1.2" stroke-dasharray="2 4" opacity="{quote_opacity}"/>'
        )
        s.append(
            f'<circle cx="{PAD_L + PLOT_W:.1f}" cy="{quote_y:.1f}" r="5.2" fill="#39c5cf" stroke="#0d1117" stroke-width="2"/>'
        )
        s.append(
            f'<text x="{RIGHT_LABEL_X}" y="{quote_y - 8:.1f}" fill="#39c5cf" '
            f'font-size="{RIGHT_LABEL_FONT_SIZE}" font-weight="bold" text-anchor="end">'
            f"{quote_label} {float(display_spot):.2f}</text>"
        )

    if show_signal_overlays:
        s.extend(_level_state_strip(level_states))

    badge_width = 258
    badge_gap = 8
    badge_lanes = [
        PAD_T + PLOT_H + 28,
        PAD_T + PLOT_H + 61,
        PAD_T + PLOT_H + 94,
        PAD_T + PLOT_H + 127,
        PAD_T + PLOT_H + 160,
    ]
    badge_lane_ends = [PAD_L - badge_gap for _ in badge_lanes]
    renderable_markers = []

    for marker in setup_markers or [] if show_signal_overlays else []:
        if not isinstance(marker, dict):
            continue
        idx = _nearest_bar_index(rows, marker.get("ts"))
        price = marker.get("price")
        if idx is None or not isinstance(price, (int, float)):
            continue
        mx = x(idx)
        renderable_markers.append(
            {
                "mx": mx,
                "my": y(float(price)),
                "status": str(marker.get("status") or "confirmed").lower(),
                "color": str(marker.get("color") or "#d29922"),
                "label": html.escape(
                    _short_marker_label(marker.get("event_type"), marker.get("status"))
                ),
                "detail": html.escape(str(marker.get("detail") or ""))[:90],
            }
        )

    for marker in sorted(renderable_markers, key=lambda item: item["mx"]):
        mx = marker["mx"]
        my = marker["my"]
        color = marker["color"]
        label = marker["label"]
        detail = marker["detail"]
        preferred_badge_x = min(max(PAD_L, mx + 8), W - badge_width - 8)
        lane_choice = 0
        badge_x = preferred_badge_x
        for lane_idx in range(len(badge_lanes)):
            lane_x = max(preferred_badge_x, badge_lane_ends[lane_idx] + badge_gap)
            if lane_x + badge_width <= W - 8:
                lane_choice = lane_idx
                badge_x = lane_x
                break
        else:
            lane_choice = min(
                range(len(badge_lane_ends)), key=lambda idx: badge_lane_ends[idx]
            )
            badge_x = preferred_badge_x
        badge_y = badge_lanes[lane_choice]
        badge_lane_ends[lane_choice] = badge_x + badge_width
        s.append(
            f'<line x1="{mx:.1f}" y1="{PAD_T}" x2="{mx:.1f}" y2="{PAD_T + PLOT_H}" '
            f'stroke="{color}" stroke-width="1.4" stroke-dasharray="3 4" opacity="0.75"/>'
        )
        s.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="6.2" fill="{color}" stroke="#0d1117" stroke-width="2"/>'
        )
        s.append(
            f'<path d="M {mx - 5:.1f} {my + 10:.1f} L {mx:.1f} {my + 18:.1f} L {mx + 5:.1f} {my + 10:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="2"/>'
        )
        s.append(
            f'<line x1="{mx:.1f}" y1="{PAD_T + PLOT_H:.1f}" x2="{mx:.1f}" y2="{badge_y - 18:.1f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="2 5" opacity="0.45"/>'
        )
        s.append(
            f'<rect x="{badge_x:.1f}" y="{badge_y - 15:.1f}" width="{badge_width}" height="30" rx="7" '
            f'fill="#0d1117" opacity="0.9" stroke="{color}"/>'
        )
        s.append(
            f'<text x="{badge_x + 8:.1f}" y="{badge_y - 2:.1f}" fill="{color}" '
            f'font-size="10" font-weight="bold">{label}</text>'
        )
        if detail:
            s.append(
                f'<text x="{badge_x + 8:.1f}" y="{badge_y + 10:.1f}" fill="#adbac7" '
                f'font-size="9">{detail}</text>'
            )

    s.append("</svg>")
    return "\n".join(s)


__all__ = ["chart_svg"]
