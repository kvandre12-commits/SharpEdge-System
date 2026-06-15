"""SPY options visual: open-interest 'walls' + IV skew, from free CBOE data.

Source: CBOE delayed quotes (no auth). We parse the OCC option symbols
(e.g. SPY260612C00400000 -> exp 2026-06-12, Call, strike 400.000), pick a
target expiration, keep strikes near spot, and render two stacked SVG panels:

  1. Open interest by strike (calls right / puts left) -- the 'walls'
  2. Implied volatility by strike -- the skew/smile

Lightweight + DRY: stdlib + requests, hand-rendered SVG.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from collections import defaultdict

import requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
SYM_RE = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")

W = 1100
PANEL_W = W - 90 - 30          # plot width (left pad 90, right pad 30)
PAD_L, PAD_R = 90, 30
ROW_H = 22                     # vertical pixels per strike row
TOP = 70
STRIKE_WINDOW = 30            # +/- strikes around spot to display


def fetch():
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    d = r.json()
    data = d["data"]
    spot = data.get("current_price") or data.get("close") or 0.0
    return data["options"], float(spot)


def parse(options):
    """-> {exp_date: {strike: {'C': row, 'P': row}}} plus set of expirations."""
    book = defaultdict(lambda: defaultdict(dict))
    for o in options:
        m = SYM_RE.match(o["option"])
        if not m:
            continue
        yymmdd, cp, strike8 = m.groups()
        exp = dt.datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike8) / 1000.0
        book[exp][strike][cp] = o
    return book


def pick_expiration(book, want: str | None):
    today = dt.date.today()
    future = sorted(e for e in book if e >= today)
    if not future:
        future = sorted(book)
    if want:
        for e in future:
            if e.isoformat() == want:
                return e
    return future[0]  # nearest


def svg_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(book, exp, spot):
    strikes_all = sorted(book[exp].keys())
    # keep strikes nearest spot
    nearest_idx = min(range(len(strikes_all)),
                      key=lambda i: abs(strikes_all[i] - spot))
    lo = max(0, nearest_idx - STRIKE_WINDOW)
    hi = min(len(strikes_all), nearest_idx + STRIKE_WINDOW + 1)
    strikes = strikes_all[lo:hi]

    rows = []
    max_oi = 1
    max_iv = 0.0001
    for k in strikes:
        c = book[exp][k].get("C", {})
        p = book[exp][k].get("P", {})
        coi = c.get("open_interest", 0) or 0
        poi = p.get("open_interest", 0) or 0
        civ = c.get("iv", 0) or 0
        piv = p.get("iv", 0) or 0
        max_oi = max(max_oi, coi, poi)
        max_iv = max(max_iv, civ, piv)
        rows.append((k, coi, poi, civ, piv))

    n = len(rows)
    oi_h = n * ROW_H
    H = TOP + oi_h + 90 + 320  # oi panel + gap + iv panel
    mid_x = PAD_L + PANEL_W / 2
    half = PANEL_W / 2 - 60     # space for center strike labels

    total_coi = sum(r[1] for r in rows)
    total_poi = sum(r[2] for r in rows)
    pcr = (total_poi / total_coi) if total_coi else 0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="monospace">',
        f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
        f'<text x="{PAD_L}" y="30" fill="#e6edf3" font-size="19" '
        f'font-weight="bold">SPY options - exp {exp.isoformat()} '
        f'&#160;<tspan fill="#7d8590" font-size="13">spot ${spot:.2f} | '
        f'P/C OI ratio {pcr:.2f}</tspan></text>',
        f'<text x="{PAD_L}" y="52" fill="#26a641" font-size="13">'
        f'PUTS (open interest)</text>',
        f'<text x="{PAD_L + PANEL_W}" y="52" fill="#f85149" font-size="13" '
        f'text-anchor="end">CALLS (open interest)</text>',
    ]

    # OI walls: puts grow left from center, calls grow right
    for i, (k, coi, poi, _, _) in enumerate(rows):
        y = TOP + i * ROW_H
        cyc = y + ROW_H / 2
        # center strike label, highlight the at-the-money strike
        atm = abs(k - spot) <= 0.5
        kcol = "#ffd33d" if atm else "#7d8590"
        parts.append(
            f'<text x="{mid_x:.1f}" y="{cyc + 4:.1f}" fill="{kcol}" '
            f'font-size="10" text-anchor="middle">{k:g}</text>'
        )
        pw = (poi / max_oi) * half
        cw = (coi / max_oi) * half
        parts.append(
            f'<rect x="{mid_x - 30 - pw:.1f}" y="{y + 3:.1f}" '
            f'width="{pw:.1f}" height="{ROW_H - 6:.1f}" fill="#26a641" '
            f'opacity="0.8"/>'
        )
        parts.append(
            f'<rect x="{mid_x + 30:.1f}" y="{y + 3:.1f}" '
            f'width="{cw:.1f}" height="{ROW_H - 6:.1f}" fill="#f85149" '
            f'opacity="0.8"/>'
        )

    # spot price guideline across OI panel
    # find vertical position between strikes bracketing spot
    parts.append(
        f'<text x="{mid_x:.1f}" y="{TOP + oi_h + 22:.1f}" fill="#ffd33d" '
        f'font-size="11" text-anchor="middle">^ yellow = at-the-money strike, '
        f'bar length = open interest (max {max_oi:,.0f})</text>'
    )

    # ---- IV skew panel ----
    iv_top = TOP + oi_h + 70
    iv_h = 260
    iv_bot = iv_top + iv_h
    pw = PANEL_W
    parts.append(
        f'<text x="{PAD_L}" y="{iv_top - 12:.1f}" fill="#e6edf3" '
        f'font-size="15" font-weight="bold">Implied Volatility skew</text>'
    )
    kmin, kmax = strikes[0], strikes[-1]
    krange = (kmax - kmin) or 1

    def ix(k):
        return PAD_L + (k - kmin) / krange * pw

    def iy(iv):
        return iv_bot - (iv / max_iv) * iv_h

    # iv gridlines
    for j in range(5):
        val = max_iv * j / 4
        yy = iy(val)
        parts.append(
            f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{PAD_L + pw}" '
            f'y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{yy + 4:.1f}" fill="#7d8590" '
            f'font-size="10" text-anchor="end">{val * 100:.0f}%</text>'
        )
    # strike x labels (a few)
    for k in strikes[::max(1, n // 8)]:
        parts.append(
            f'<text x="{ix(k):.1f}" y="{iv_bot + 16:.1f}" fill="#7d8590" '
            f'font-size="10" text-anchor="middle">{k:g}</text>'
        )
    # spot vertical
    parts.append(
        f'<line x1="{ix(spot):.1f}" y1="{iv_top}" x2="{ix(spot):.1f}" '
        f'y2="{iv_bot}" stroke="#ffd33d" stroke-width="1" '
        f'stroke-dasharray="4 4"/>'
    )
    # call iv line (red), put iv line (green)
    cline = " ".join(f"{ix(k):.1f},{iy(civ):.1f}"
                     for k, _, _, civ, _ in rows if civ > 0)
    pline = " ".join(f"{ix(k):.1f},{iy(piv):.1f}"
                     for k, _, _, _, piv in rows if piv > 0)
    if cline:
        parts.append(
            f'<polyline points="{cline}" fill="none" stroke="#f85149" '
            f'stroke-width="1.8"/>'
        )
    if pline:
        parts.append(
            f'<polyline points="{pline}" fill="none" stroke="#26a641" '
            f'stroke-width="1.8"/>'
        )
    parts.append(
        f'<text x="{PAD_L + pw}" y="{iv_top + 14:.1f}" fill="#f85149" '
        f'font-size="11" text-anchor="end">call IV</text>'
    )
    parts.append(
        f'<text x="{PAD_L + pw}" y="{iv_top + 30:.1f}" fill="#26a641" '
        f'font-size="11" text-anchor="end">put IV</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    options, spot = fetch()
    book = parse(options)
    exp = pick_expiration(book, want)
    print(f"spot=${spot:.2f}  expirations available: {len(book)}")
    print("nearest few:", ", ".join(
        e.isoformat() for e in sorted(book)[:8]))
    print(f"rendering expiration: {exp.isoformat()}")
    svg = build_svg(book, exp, spot)
    out = "/data/data/com.termux/files/home/spy_overlay/spy_options.svg"
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
