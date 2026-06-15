"""Overlay the last ~2 weeks of intraday SPY trading days on one chart.

Lightweight + DRY: only stdlib + requests. We hand-render an SVG so there is
no matplotlib/pandas dependency to compile on a phone.

Each trading day's 5-minute curve is normalized to % change from that day's
first regular-session print, then drawn on a shared axis (x = minutes since
the 09:30 ET open). Older days fade; the latest day is bold.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

import requests

SYMBOL = "SPY"
URL = (
    f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
    "?interval=5m&range=14d"
)
OPEN_MIN = 570   # 09:30 ET in minutes-from-midnight
CLOSE_MIN = 960  # 16:00 ET
SESSION_LEN = CLOSE_MIN - OPEN_MIN  # 390 minutes

W, H = 1100, 680
PAD_L, PAD_R, PAD_T, PAD_B = 70, 180, 50, 60
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B


def fetch():
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    gmt = res["meta"]["gmtoffset"]  # seconds to add to UTC for exchange wall time
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    return gmt, ts, closes


def group_by_day(gmt, ts, closes):
    """Return {date: [(minute_of_day, close), ...]} for regular-session bars."""
    days = defaultdict(list)
    for t, c in zip(ts, closes):
        if c is None:
            continue
        local = dt.datetime.utcfromtimestamp(t + gmt)
        minute = local.hour * 60 + local.minute
        if OPEN_MIN <= minute <= CLOSE_MIN:
            days[local.date()].append((minute - OPEN_MIN, c))
    # keep only days that actually have data, sorted chronologically
    return {d: sorted(v) for d, v in sorted(days.items()) if v}


def to_pct(series):
    base = series[0][1]
    return [(x, (c / base - 1.0) * 100.0) for x, c in series]


def build_svg(days):
    pct_days = {d: to_pct(s) for d, s in days.items()}
    all_pct = [y for s in pct_days.values() for _, y in s]
    lo, hi = min(all_pct), max(all_pct)
    span = (hi - lo) or 1.0
    pad = span * 0.08
    lo -= pad
    hi += pad
    span = hi - lo

    def px(minute):
        return PAD_L + (minute / SESSION_LEN) * PLOT_W

    def py(pct):
        return PAD_T + (1 - (pct - lo) / span) * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
        f'<text x="{PAD_L}" y="30" fill="#e6edf3" font-size="20" '
        f'font-weight="bold">SPY intraday overlay — last {len(days)} trading '
        f'days (normalized % from open)</text>',
    ]

    # horizontal gridlines + y labels (% values)
    steps = 6
    for i in range(steps + 1):
        val = lo + span * i / steps
        y = py(val)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" '
            f'y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" fill="#7d8590" '
            f'font-size="11" text-anchor="end">{val:+.2f}%</text>'
        )

    # vertical gridlines every hour + x labels (clock time ET)
    for hr_off in range(0, SESSION_LEN + 1, 60):
        x = px(hr_off)
        clock = OPEN_MIN + hr_off
        label = f"{clock // 60:02d}:{clock % 60:02d}"
        parts.append(
            f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" '
            f'y2="{PAD_T + PLOT_H}" stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{PAD_T + PLOT_H + 20:.1f}" fill="#7d8590" '
            f'font-size="11" text-anchor="middle">{label}</text>'
        )

    # zero line
    zy = py(0.0)
    parts.append(
        f'<line x1="{PAD_L}" y1="{zy:.1f}" x2="{PAD_L + PLOT_W}" '
        f'y2="{zy:.1f}" stroke="#484f58" stroke-width="1.5" '
        f'stroke-dasharray="4 4"/>'
    )

    ordered = list(pct_days.items())  # chronological
    n = len(ordered)
    for idx, (d, series) in enumerate(ordered):
        latest = idx == n - 1
        # fade older days: recency 0 (oldest) -> 1 (latest)
        recency = idx / max(n - 1, 1)
        if latest:
            color, width, opacity = "#ff5252", 3.0, 1.0
        else:
            # blue-ish ramp, brighter & more opaque as it gets recent
            shade = int(90 + 120 * recency)
            color = f"rgb({shade // 2},{shade},{min(255, shade + 60)})"
            width = 1.4
            opacity = 0.30 + 0.45 * recency
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in series)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" opacity="{opacity:.2f}"/>'
        )
        # end-of-day label in the right margin
        last_y = py(series[-1][1])
        parts.append(
            f'<text x="{PAD_L + PLOT_W + 8}" y="{last_y + 4:.1f}" '
            f'fill="{color}" font-size="11" opacity="{max(opacity, 0.6):.2f}">'
            f'{d.strftime("%m/%d")} {series[-1][1]:+.2f}%</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    gmt, ts, closes = fetch()
    days = group_by_day(gmt, ts, closes)
    svg = build_svg(days)
    out = "/data/data/com.termux/files/home/spy_overlay/spy_overlay.svg"
    with open(out, "w") as f:
        f.write(svg)
    print(f"trading days plotted: {len(days)}")
    for d, s in days.items():
        chg = (s[-1][1] / s[0][1] - 1) * 100
        print(f"  {d}  bars={len(s):3d}  day_change={chg:+.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
