from __future__ import annotations

from gate_workflows import primary_trade_setup

W, H = 1000, 420
PAD_L, PAD_R, PAD_T, PAD_B = 42, 64, 18, 34
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B
CHANNEL_COLOR = "#bc8cff"
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
    x = PAD_L + PLOT_W - 188
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


def chart_svg(
    rows, pa, levels=None, setups=None, volatility_structure=None, level_states=None
):
    closes = [b[4] for b in rows]
    n = len(closes)
    lo, hi = min(closes), max(closes)
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
    levels = levels or {}
    setups = setups or []
    channel_overlay = _channel_overlay(volatility_structure)

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
        f'<text x="{PAD_L + PLOT_W + 4}" y="{vy + 4:.1f}" fill="#ffd33d" '
        f'font-size="12" font-weight="bold">VWAP {pa["vwap"]:.2f}</text>'
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
        f'<text x="{PAD_L + PLOT_W + 4}" y="{y(closes[-1]) + 4:.1f}" fill="#58a6ff" '
        f'font-size="13" font-weight="bold">${closes[-1]:.2f}</text>'
    )
    s.extend(_level_state_strip(level_states))
    s.append("</svg>")
    return "\n".join(s)


__all__ = ["chart_svg"]
