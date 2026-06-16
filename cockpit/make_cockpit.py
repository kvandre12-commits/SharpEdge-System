"""SharpEdge live cockpit: turn real-time SPY motion into data-driven reads.

Goal: reduce time-to-execution. You SEE the move; this CONFIRMS it with
numbers in the same glance -- range position, VWAP control, momentum,
volume confirmation, and options walls pinning price.

Two free sources, no auth:
  - Yahoo 1-min intraday  -> price action / VWAP / momentum / volume
  - CBOE delayed options   -> OI walls, put/call ratio, ATM IV

Lightweight + DRY: stdlib + requests, writes cockpit.html + cockpit chart svg.
Run it in a loop for live updates; the HTML meta-refreshes to pick them up.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import defaultdict

import requests

from gamma import gamma_card, gamma_profile
from setups import detect_exhaustion, detect_failed_breaks, reference_levels

UA = {"User-Agent": "Mozilla/5.0"}
INTRA_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?interval=1m&range=1d"
)
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
SYM_RE = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- data fetch -----------------------------
def fetch_intraday():
    r = requests.get(INTRA_URL, headers=UA, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    gmt = res["meta"]["gmtoffset"]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    o_, h_, l_ = q["open"], q["high"], q["low"]
    for i, (t, c, v) in enumerate(zip(ts, q["close"], q["volume"])):
        if c is None:
            continue
        local = dt.datetime.utcfromtimestamp(t + gmt)
        minute = local.hour * 60 + local.minute
        if 570 <= minute <= 960:  # regular session 09:30-16:00 ET
            o = o_[i] if o_[i] is not None else c
            h = h_[i] if h_[i] is not None else c
            low = l_[i] if l_[i] is not None else c
            # bar = (minute_of_session, open, high, low, close, volume)
            rows.append((minute - 570, o, h, low, c, v or 0))
    return rows


def fetch_options():
    r = requests.get(CBOE_URL, headers=UA, timeout=30)
    r.raise_for_status()
    d = r.json()["data"]
    spot = float(d.get("current_price") or d.get("close") or 0)
    book = defaultdict(lambda: defaultdict(dict))
    for o in d["options"]:
        m = SYM_RE.match(o["option"])
        if not m:
            continue
        yymmdd, cp, strike8 = m.groups()
        exp = dt.datetime.strptime(yymmdd, "%y%m%d").date()
        book[exp][int(strike8) / 1000.0][cp] = o
    return spot, book


# ----------------------------- analytics -----------------------------
def read_price_action(rows):
    closes = [b[4] for b in rows]
    vols = [b[5] for b in rows]
    spot = closes[-1]
    day_open = closes[0]
    hi, lo = max(closes), min(closes)
    rng = (hi - lo) or 1e-9
    rng_pos = (spot - lo) / rng * 100  # 0=low, 100=high of day

    # VWAP (who controls the day)
    cum_pv = sum(b[4] * b[5] for b in rows)
    cum_v = sum(vols) or 1
    vwap = cum_pv / cum_v

    # short-term momentum: last 15 minutes
    look = min(15, len(closes) - 1)
    mom = (spot / closes[-1 - look] - 1) * 100 if look else 0.0

    # volume confirmation: last 5 bars vs median bar
    recent = vols[-5:]
    body = sorted(vols)
    med = body[len(body) // 2] or 1
    vol_mult = (sum(recent) / len(recent)) / med

    return {
        "spot": spot,
        "day_open": day_open,
        "hi": hi,
        "lo": lo,
        "rng_pos": rng_pos,
        "day_chg": (spot / day_open - 1) * 100,
        "vwap": vwap,
        "vs_vwap": (spot - vwap) / vwap * 100,
        "mom15": mom,
        "vol_mult": vol_mult,
    }


def read_options(spot, book):
    today = dt.date.today()
    future = sorted(e for e in book if e >= today) or sorted(book)
    exp = future[0]
    strikes = sorted(book[exp].keys())
    call_oi = {k: (book[exp][k].get("C", {}).get("open_interest", 0) or 0)
               for k in strikes}
    put_oi = {k: (book[exp][k].get("P", {}).get("open_interest", 0) or 0)
              for k in strikes}
    # walls: biggest OI strike above (calls) / below (puts) spot
    calls_above = {k: v for k, v in call_oi.items() if k >= spot}
    puts_below = {k: v for k, v in put_oi.items() if k <= spot}
    call_wall = max(calls_above, key=calls_above.get) if calls_above else None
    put_wall = max(puts_below, key=puts_below.get) if puts_below else None
    tot_c = sum(call_oi.values()) or 1
    tot_p = sum(put_oi.values())
    pcr = tot_p / tot_c
    # ATM IV
    atm = min(strikes, key=lambda k: abs(k - spot))
    civ = book[exp][atm].get("C", {}).get("iv", 0) or 0
    piv = book[exp][atm].get("P", {}).get("iv", 0) or 0
    atm_iv = (civ + piv) / 2 if (civ or piv) else 0
    return {
        "exp": exp.isoformat(),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "pcr": pcr,
        "atm_iv": atm_iv,
    }


def synthesize(pa, op):
    """Plain-English, number-backed reads. Each line cites its data."""
    lines = []

    # 1. who controls the tape
    if pa["vs_vwap"] > 0.05:
        lines.append(("BULLS in control", "ok",
                      f"price ${pa['spot']:.2f} is {pa['vs_vwap']:+.2f}% "
                      f"ABOVE VWAP ${pa['vwap']:.2f}"))
    elif pa["vs_vwap"] < -0.05:
        lines.append(("BEARS in control", "bad",
                      f"price ${pa['spot']:.2f} is {pa['vs_vwap']:+.2f}% "
                      f"BELOW VWAP ${pa['vwap']:.2f}"))
    else:
        lines.append(("BALANCED / chop", "warn",
                      f"price hugging VWAP ${pa['vwap']:.2f} "
                      f"({pa['vs_vwap']:+.2f}%) - no edge, wait"))

    # 2. where in the day's range
    rp = pa["rng_pos"]
    if rp >= 80:
        lines.append(("At day HIGHS", "warn",
                      f"{rp:.0f}% of range - breakout OR exhaustion zone"))
    elif rp <= 20:
        lines.append(("At day LOWS", "warn",
                      f"{rp:.0f}% of range - breakdown OR reclaim zone"))
    else:
        lines.append(("Mid-range", "info",
                      f"{rp:.0f}% of day range (lo ${pa['lo']:.2f} / "
                      f"hi ${pa['hi']:.2f})"))

    # 3. momentum real or fading
    if abs(pa["mom15"]) < 0.05:
        lines.append(("Momentum FLAT", "info",
                      f"{pa['mom15']:+.2f}% last 15m - no thrust"))
    elif pa["mom15"] > 0:
        lines.append(("Momentum UP", "ok",
                      f"{pa['mom15']:+.2f}% last 15m"))
    else:
        lines.append(("Momentum DOWN", "bad",
                      f"{pa['mom15']:+.2f}% last 15m"))

    # 4. volume confirming?
    vm = pa["vol_mult"]
    if vm >= 1.5:
        lines.append(("Volume SURGE", "ok",
                      f"{vm:.1f}x normal - move is CONFIRMED"))
    elif vm <= 0.7:
        lines.append(("Volume THIN", "warn",
                      f"{vm:.1f}x normal - move NOT confirmed, fade risk"))
    else:
        lines.append(("Volume normal", "info", f"{vm:.1f}x typical bar"))

    # 5. options walls (magnets / levels)
    cw, pw = op["call_wall"], op["put_wall"]
    if cw is not None and pw is not None:
        lines.append(("Options box", "info",
                      f"put wall ${pw:g} (support) <-> call wall ${cw:g} "
                      f"(resistance) | exp {op['exp']}"))
    lines.append(("Sentiment", "info",
                  f"P/C OI {op['pcr']:.2f} | ATM IV {op['atm_iv'] * 100:.1f}%"))

    return lines


# ----------------------------- rendering -----------------------------
def chart_svg(rows, pa):
    W, H, PL, PR, PT, PB = 1000, 320, 60, 70, 20, 28
    pw, ph = W - PL - PR, H - PT - PB
    closes = [b[4] for b in rows]
    n = len(closes)
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1
    pad = span * 0.08
    lo -= pad
    hi += pad
    span = hi - lo

    def x(i):
        return PL + i / max(n - 1, 1) * pw

    def y(p):
        return PT + (1 - (p - lo) / span) * ph

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">',
         f'<rect width="{W}" height="{H}" fill="#0d1117"/>']
    # VWAP line
    vy = y(pa["vwap"])
    s.append(f'<line x1="{PL}" y1="{vy:.1f}" x2="{PL+pw}" y2="{vy:.1f}" '
             f'stroke="#ffd33d" stroke-width="1" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{PL+pw+4}" y="{vy+4:.1f}" fill="#ffd33d" '
             f'font-size="10">VWAP {pa["vwap"]:.2f}</text>')
    # price line, colored by vwap control
    col = "#26a641" if pa["vs_vwap"] >= 0 else "#f85149"
    pts = " ".join(f"{x(i):.1f},{y(closes[i]):.1f}" for i in range(n))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
             f'stroke-width="1.8"/>')
    # last dot
    s.append(f'<circle cx="{x(n-1):.1f}" cy="{y(closes[-1]):.1f}" r="3.5" '
             f'fill="#58a6ff"/>')
    s.append(f'<text x="{PL+pw+4}" y="{y(closes[-1])+4:.1f}" fill="#58a6ff" '
             f'font-size="11" font-weight="bold">${closes[-1]:.2f}</text>')
    s.append("</svg>")
    return "\n".join(s)


CLR = {"ok": "#26a641", "bad": "#f85149", "warn": "#d29922", "info": "#58a6ff"}


def _setup_section(setups):
    if not setups:
        return ('<div style="border:1px dashed #30363d;background:#0d1117;'
                'padding:12px;margin:8px 0;border-radius:6px;color:#7d8590;'
                'font-size:13px">No failed-break or exhaustion setup right now '
                '- stand down, wait for the trap.</div>')
    blocks = []
    for s in setups:
        c = CLR.get(s["kind"], "#58a6ff")
        blocks.append(
            f'<div style="border:2px solid {c};background:#161b22;'
            f'padding:12px;margin:8px 0;border-radius:8px">'
            f'<div style="color:{c};font-weight:bold;font-size:17px">'
            f'{s["tag"]} &#8594; {s["bias"]}</div>'
            f'<div style="color:#adbac7;font-size:13px;margin-top:4px">'
            f'{s["detail"]}</div></div>'
        )
    return "".join(blocks)


def render_html(pa, op, lines, setups):
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    sign = "+" if pa["day_chg"] >= 0 else ""
    cards = []
    for title, kind, detail in lines:
        c = CLR.get(kind, "#58a6ff")
        cards.append(
            f'<div style="border-left:4px solid {c};background:#161b22;'
            f'padding:10px 12px;margin:8px 0;border-radius:6px">'
            f'<div style="color:{c};font-weight:bold;font-size:15px">{title}'
            f'</div><div style="color:#adbac7;font-size:13px;margin-top:3px">'
            f'{detail}</div></div>'
        )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="45">
<title>SharpEdge Cockpit</title></head>
<body style="margin:0;background:#0d1117;color:#e6edf3;font-family:monospace">
<div style="padding:12px">
<div style="display:flex;justify-content:space-between;align-items:baseline">
<h2 style="margin:0;font-size:18px">SharpEdge Live Read - SPY</h2>
<span style="color:#7d8590;font-size:12px">updated {stamp} | auto 45s</span>
</div>
<div style="font-size:26px;font-weight:bold;margin:6px 0">
${pa['spot']:.2f}
<span style="font-size:16px;color:{'#26a641' if pa['day_chg']>=0 else '#f85149'}">
{sign}{pa['day_chg']:.2f}% today</span></div>
<img src="cockpit_chart.svg" style="width:100%;border:1px solid #21262d;
border-radius:8px">
<h3 style="font-size:14px;color:#e6edf3;margin:14px 0 4px">SETUPS
(failed breaks + exhaustion)</h3>
{_setup_section(setups)}
<h3 style="font-size:14px;color:#7d8590;margin:14px 0 4px">THE READ
(context)</h3>
{''.join(cards)}
<p style="color:#484f58;font-size:11px;margin-top:14px">
Free data (Yahoo 1m + CBOE delayed options). Decision support only -
you own every trade.</p>
</div></body></html>"""


def read_microstructure(rows, lookback=8):
    """OHLC-only microstructure of the session-so-far candle + a Donchian channel.

    rows = (minute, open, high, low, close, volume). All pure OHLC logic:
      - bar anatomy: lower/upper wick + body as % of the day range (wick =
        absorption/rejection; lower_wick is our strongest model feature).
      - Donchian channel over the last `lookback` bars: where price sits in the
        channel (0=floor,100=ceiling), channel width %, and channel slope.
    """
    if not rows:
        return {}
    o = rows[0][1]
    hi = max(r[2] for r in rows)
    lo = min(r[3] for r in rows)
    c = rows[-1][4]
    rng = max(hi - lo, 1e-9)
    lower_wick = (min(o, c) - lo) / rng * 100
    upper_wick = (hi - max(o, c)) / rng * 100
    body = abs(c - o) / rng * 100

    win = rows[-lookback:] if len(rows) >= lookback else rows
    ch_hi = max(r[2] for r in win)
    ch_lo = min(r[3] for r in win)
    ch_w = max(ch_hi - ch_lo, 1e-9)
    ch_pos = (c - ch_lo) / ch_w * 100
    ch_width_pct = ch_w / c * 100
    # channel slope: midline now vs midline `lookback` bars earlier
    prev = rows[-2 * lookback:-lookback] if len(rows) >= 2 * lookback else rows[:1]
    prev_mid = (max(r[2] for r in prev) + min(r[3] for r in prev)) / 2
    cur_mid = (ch_hi + ch_lo) / 2
    ch_slope_pct = (cur_mid - prev_mid) / c * 100
    return {
        "lower_wick": round(lower_wick, 1),
        "upper_wick": round(upper_wick, 1),
        "body": round(body, 1),
        "ch_pos": round(ch_pos, 1),
        "ch_hi": round(ch_hi, 2),
        "ch_lo": round(ch_lo, 2),
        "ch_width_pct": round(ch_width_pct, 3),
        "ch_slope_pct": round(ch_slope_pct, 3),
        "ch_lookback": lookback,
    }


def read_magnitude(rows, spot, atm_iv, K=2.5356):
    """Forecast the REST-OF-DAY move size (magnitude is forecastable; sign is not).

    Two estimates of the expected |move| over the remaining session:
      - realized-vol model: K * Garman-Klass(open->now). GK morning vol predicts
        afternoon |move| with OOS Spearman IC ~0.4 (0.21 OOS); K=2.54 calibrated
        on 359 days to the 11:30 split.
      - options-implied: atm_iv * sqrt(remaining trading-time).
    realized > implied => options underpricing the move ('cheap'); else 'rich'.
    """
    import math
    if len(rows) < 3:
        return {}
    terms = []
    for _m, o, h, l, c, _v in rows:
        if o > 0 and l > 0 and h > 0:
            terms.append(0.5 * math.log(h / l) ** 2
                         - (2 * math.log(2) - 1) * math.log(max(c / o, 1e-9)) ** 2)
    if not terms:
        return {}
    gk = math.sqrt(max(sum(terms) / len(terms), 0.0)) * 100  # % per-bar vol
    minute_now = rows[-1][0]
    remaining_frac = max(390 - minute_now, 5) / 390.0  # fraction of session left
    realized_pct = K * gk
    implied_pct = (atm_iv or 0) * math.sqrt(remaining_frac / 252.0) * 100
    return {
        "gk_vol": round(gk, 3),
        "exp_move_realized_pct": round(realized_pct, 3),
        "exp_move_realized_usd": round(spot * realized_pct / 100, 2),
        "exp_move_implied_pct": round(implied_pct, 3),
        "exp_move_implied_usd": round(spot * implied_pct / 100, 2),
        "premium_read": "cheap" if realized_pct > implied_pct else "rich",
        "remaining_frac": round(remaining_frac, 3),
    }


def write_signal(pa, op, gp, gcard, micro=None, magnitude=None):
    """Drop a machine-readable signal.json the trade_intent pipeline can read."""
    sig = {
        "schema": "sharpedge.signal.v1",
        "ts": dt.datetime.now().isoformat(),
        "symbol": "SPY",
        "spot": round(pa["spot"], 2),
        "day_chg": round(pa["day_chg"], 3),
        "vwap": round(pa["vwap"], 2),
        "vs_vwap": round(pa["vs_vwap"], 3),
        "rng_pos": round(pa["rng_pos"], 1),
        "mom15": round(pa["mom15"], 3),
        "vol_mult": round(pa["vol_mult"], 2),
        "call_wall": op.get("call_wall"),
        "put_wall": op.get("put_wall"),
        "pcr": round(op.get("pcr", 0), 2),
        "atm_iv": round(op.get("atm_iv", 0), 4),
        "exp": op.get("exp"),
        "gamma_regime": gp.get("regime"),
        "pin": gp.get("pin"),
        "max_pain": gp.get("max_pain"),
        "setup_tag": gcard["tag"] if gcard else None,
        "setup_bias": gcard["bias"] if gcard else None,
        "micro": micro or {},
        "magnitude": magnitude or {},
    }
    out = os.path.expanduser("~/SharpEdge-System/outputs")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "signal.json"), "w") as f:
        json.dump(sig, f, indent=2)


def main():
    rows = fetch_intraday()
    spot_opt, book = fetch_options()
    pa = read_price_action(rows)
    op = read_options(pa["spot"], book)
    lines = synthesize(pa, op)
    levels = reference_levels(rows)
    setups = detect_failed_breaks(rows, levels) + detect_exhaustion(rows, pa)
    gp = gamma_profile(book, pa["spot"])
    gcard = gamma_card(gp)
    if gcard:
        setups = [gcard] + setups  # gamma regime sits at the very top
    micro = read_microstructure(rows)
    magnitude = read_magnitude(rows, pa["spot"], op.get("atm_iv", 0))
    write_signal(pa, op, gp, gcard, micro, magnitude)  # machine-readable feed for trade_intent
    with open(f"{OUT_DIR}/cockpit_chart.svg", "w") as f:
        f.write(chart_svg(rows, pa))
    with open(f"{OUT_DIR}/cockpit.html", "w") as f:
        f.write(render_html(pa, op, lines, setups))
    print(f"spot ${pa['spot']:.2f} | day {pa['day_chg']:+.2f}% | "
          f"vs VWAP {pa['vs_vwap']:+.2f}% | rng {pa['rng_pos']:.0f}% | "
          f"vol {pa['vol_mult']:.1f}x")
    levels_str = " ".join(f"{k}=${v:.2f}" for k, v in levels.items())
    print(f"  levels: {levels_str}")
    if setups:
        for s in setups:
            print(f"  >> {s['tag']} -> {s['bias']}: {s['detail']}")
    else:
        print("  >> no failed-break/exhaustion setup right now")
    for t, k, d in lines:
        print(f"  [{k:4}] {t}: {d}")


if __name__ == "__main__":
    main()
