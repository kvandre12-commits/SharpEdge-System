"""SPY intraday PRICE + VOLUME chart for the last ~2 weeks.

Continuous timeline: trading days are concatenated back-to-back (overnight
gaps removed) so the tape reads like a real charting app. Top panel = price
in dollars; bottom panel = volume bars (green up-bar / red down-bar).

Lightweight + DRY: stdlib + requests only, hand-rendered SVG (no matplotlib).
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
OPEN_MIN = 570   # 09:30 ET
CLOSE_MIN = 960  # 16:00 ET

W, H = 1200, 760
PAD_L, PAD_R, PAD_T = 78, 95, 50
PAD_B = 60
GAP = 24                       # gap between price and volume panels
PRICE_H = 430                  # price panel height
VOL_H = H - PAD_T - PRICE_H - GAP - PAD_B  # volume panel height
PLOT_W = W - PAD_L - PAD_R
PRICE_TOP = PAD_T
PRICE_BOT = PAD_T + PRICE_H
VOL_TOP = PRICE_BOT + GAP
VOL_BOT = VOL_TOP + VOL_H


def fetch():
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    gmt = res["meta"]["gmtoffset"]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    return gmt, ts, q["close"], q["volume"]


def collect(gmt, ts, closes, vols):
    """Ordered list of regular-session bars: (date, close, volume)."""
    rows = []
    for t, c, v in zip(ts, closes, vols):
        if c is None:
            continue
        local = dt.datetime.utcfromtimestamp(t + gmt)
        minute = local.hour * 60 + local.minute
        if OPEN_MIN <= minute <= CLOSE_MIN:
            rows.append((local.date(), c, v or 0))
    return rows


def day_spans(rows):
    """Group consecutive rows into (date, start_index, end_index) spans."""
    spans = []
    start = 0
    cur = rows[0][0]
    for i, (d, _, _) in enumerate(rows):
        if d != cur:
            spans.append((cur, start, i - 1))
            cur = d
            start = i
    spans.append((cur, start, len(rows) - 1))
    return spans


def build_svg(rows):
    n = len(rows)
    prices = [c for _, c, _ in rows]
    vols = [v for _, _, v in rows]
    p_lo, p_hi = min(prices), max(prices)
    p_pad = (p_hi - p_lo) * 0.06 or 1.0
    p_lo -= p_pad
    p_hi += p_pad
    p_span = p_hi - p_lo
    v_max = max(vols) or 1

    def px(i):
        return PAD_L + (i / max(n - 1, 1)) * PLOT_W

    def py(price):
        return PRICE_TOP + (1 - (price - p_lo) / p_span) * PRICE_H

    def vy(vol):
        return VOL_BOT - (vol / v_max) * VOL_H

    spans = day_spans(rows)
    last_close = prices[-1]
    first_close = prices[0]
    net = (last_close / first_close - 1) * 100

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
        f'<text x="{PAD_L}" y="30" fill="#e6edf3" font-size="20" '
        f'font-weight="bold">SPY 5-min price &amp; volume - last {len(spans)} '
        f'trading days &#160;&#160;<tspan fill="#7d8590" font-size="14">'
        f'(${first_close:.2f} -&gt; ${last_close:.2f}, {net:+.2f}%)</tspan>'
        f'</text>',
    ]

    # ---- price panel gridlines + $ labels ----
    steps = 6
    for i in range(steps + 1):
        val = p_lo + p_span * i / steps
        y = py(val)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" '
            f'y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" fill="#7d8590" '
            f'font-size="11" text-anchor="end">${val:.2f}</text>'
        )

    # ---- day separators + date labels (both panels) ----
    for d, s, e in spans:
        x = px(s)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PRICE_TOP}" x2="{x:.1f}" '
            f'y2="{VOL_BOT}" stroke="#161b22" stroke-width="1"/>'
        )
        mid = px((s + e) / 2)
        parts.append(
            f'<text x="{mid:.1f}" y="{H - 22}" fill="#7d8590" '
            f'font-size="10" text-anchor="middle">{d.strftime("%m/%d")}</text>'
        )

    # ---- volume bars (green up / red down vs prior bar) ----
    bar_w = max(PLOT_W / n * 0.8, 1.0)
    for i, v in enumerate(vols):
        up = i == 0 or prices[i] >= prices[i - 1]
        color = "#26a641" if up else "#f85149"
        x = px(i) - bar_w / 2
        y = vy(v)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{VOL_BOT - y:.1f}" fill="{color}" opacity="0.65"/>'
        )

    # volume axis label
    parts.append(
        f'<text x="{PAD_L - 8}" y="{VOL_TOP + 12:.1f}" fill="#7d8590" '
        f'font-size="11" text-anchor="end">{v_max / 1e6:.0f}M</text>'
    )
    parts.append(
        f'<text x="{PAD_L - 8}" y="{VOL_BOT:.1f}" fill="#7d8590" '
        f'font-size="11" text-anchor="end">0</text>'
    )

    # ---- price line (drawn last, on top) ----
    pts = " ".join(f"{px(i):.1f},{py(prices[i]):.1f}" for i in range(n))
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="#58a6ff" '
        f'stroke-width="1.8"/>'
    )
    # last price dot + label
    ly = py(last_close)
    parts.append(
        f'<circle cx="{px(n - 1):.1f}" cy="{ly:.1f}" r="3.5" fill="#ffd33d"/>'
    )
    parts.append(
        f'<text x="{px(n - 1) + 8:.1f}" y="{ly + 4:.1f}" fill="#ffd33d" '
        f'font-size="12" font-weight="bold">${last_close:.2f}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    gmt, ts, closes, vols = fetch()
    rows = collect(gmt, ts, closes, vols)
    svg = build_svg(rows)
    out = "/data/data/com.termux/files/home/spy_overlay/spy_price_volume.svg"
    with open(out, "w") as f:
        f.write(svg)
    spans = day_spans(rows)
    print(f"bars={len(rows)}  days={len(spans)}")
    print(f"price range ${min(c for _,c,_ in rows):.2f} - "
          f"${max(c for _,c,_ in rows):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
