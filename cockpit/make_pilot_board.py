#!/usr/bin/env python3
"""SharpEdge Pilot Board.

One-screen intraday pilot board optimized for "what should I do?" rather than
"how much data can we cram into a browser?"

Writes cockpit/pilot_board.html.
"""

from __future__ import annotations

import html
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
OUTPUTS = ROOT / "outputs"
_BRIDGE_SRC = Path(os.path.expanduser("~/SharpEdge-Robinhood-Bridge/src"))
if _BRIDGE_SRC.exists():
    sys.path.insert(0, str(_BRIDGE_SRC))

from make_cockpit import fetch_intraday  # noqa: E402
from sharpedge_robinhood_bridge.analytics_context import load_execution_state  # noqa: E402
from sharpedge_robinhood_bridge.trade_intent import decide  # noqa: E402

BG = "#0d1117"
SURFACE = "#161b22"
FG = "#e6edf3"
MUTE = "#7d8590"
GRID = "#30363d"
BLUE = "#58a6ff"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"


def _esc(value) -> str:
    return html.escape(str(value))



def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}



def _fmt_price(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"



def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"



def _signal() -> dict:
    return _load_json(OUTPUTS / "signal.json")



def _scoreboard() -> dict:
    return _load_json(OUTPUTS / "reconcile_summary.json")



def _rows() -> list[tuple]:
    try:
        return fetch_intraday()
    except Exception:
        return []



def _card(title: str, body: str, accent: str) -> str:
    return (
        f'<section style="background:{SURFACE};border:1px solid {GRID};border-left:5px solid {accent};'
        'border-radius:12px;padding:14px 14px 12px;margin-bottom:12px">'
        f'<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{MUTE};margin-bottom:8px">{_esc(title)}</div>'
        f'{body}</section>'
    )



def _confidence_label(ctx, rec: dict) -> tuple[str, str]:
    if not getattr(ctx, "available", False):
        return "LOW", RED
    score = getattr(ctx, "execution_score", None)
    prob_range = getattr(ctx, "prob_range", None)
    prob_trend = getattr(ctx, "prob_trend", None)
    if score is None:
        return "LOW", RED
    if score >= 70 and ((prob_range or 0) >= 0.65 or (prob_trend or 0) >= 0.65):
        return "HIGH", GREEN
    if score >= 40:
        return "MED", AMBER
    return "LOW", RED



def _play_text(decision: dict, signal: dict, ctx) -> tuple[str, str, str]:
    action = decision.get("action", "stand_down")
    reason = decision.get("reason", signal.get("setup_bias", "No setup bias available."))
    bias = getattr(ctx, "final_bias", "") or signal.get("setup_tag", "")
    if action == "trade" and decision.get("intent") is not None:
        intent = decision["intent"]
        leg = intent.option_legs[0] if intent.option_legs else {}
        contract = f"{intent.symbol} {leg.get('expiry', '?')} {leg.get('strike', '?')} {str(leg.get('right', '?')).upper()}"
        return "TAKE THE PLAY", reason, contract
    if bias == "RANGE_FADE":
        return "FADE THE EDGE", reason, "Bias says mean-revert toward the magnet."
    if bias == "TREND_RUNNER":
        return "PRESS THE TREND", reason, "Bias says stay with momentum, not against it."
    return "STAND DOWN", reason, "No clean trade location yet."



def _overlay_levels(signal: dict) -> list[dict]:
    mag = signal.get("magnitude", {})
    micro = signal.get("micro", {})
    spot = signal.get("spot")
    exp_move = mag.get("exp_move_realized_usd")
    expected_high = (spot + exp_move) if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float)) else None
    expected_low = (spot - exp_move) if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float)) else None
    levels = [
        {"label": "Current", "value": spot, "color": FG, "dash": "", "width": 2.2},
        {"label": "VWAP", "value": signal.get("vwap"), "color": AMBER, "dash": "8 6", "width": 1.5},
        {"label": "Call wall", "value": signal.get("call_wall"), "color": RED, "dash": "4 4", "width": 1.5},
        {"label": "Put wall", "value": signal.get("put_wall"), "color": GREEN, "dash": "4 4", "width": 1.5},
        {"label": "Magnet", "value": signal.get("pin"), "color": BLUE, "dash": "10 5", "width": 1.8},
        {"label": "Exp high", "value": expected_high, "color": CYAN, "dash": "2 6", "width": 1.3},
        {"label": "Exp low", "value": expected_low, "color": CYAN, "dash": "2 6", "width": 1.3},
        {"label": "Channel hi", "value": micro.get("ch_hi"), "color": PURPLE, "dash": "6 4", "width": 1.3},
        {"label": "Channel lo", "value": micro.get("ch_lo"), "color": PURPLE, "dash": "6 4", "width": 1.3},
    ]
    return [level for level in levels if isinstance(level["value"], (int, float))]



def _chart_svg(rows: list[tuple], signal: dict) -> str:
    if not rows:
        return (
            f'<div style="display:flex;align-items:center;justify-content:center;height:520px;'
            f'background:{SURFACE};border:1px solid {GRID};border-radius:14px;color:{MUTE}">'
            'No intraday rows available.</div>'
        )

    width, height = 1120, 620
    pl, pr, pt, pb = 72, 160, 24, 42
    pw, ph = width - pl - pr, height - pt - pb
    closes = [bar[4] for bar in rows]
    minutes = [bar[0] for bar in rows]
    overlays = _overlay_levels(signal)
    prices = closes + [level["value"] for level in overlays]
    lo, hi = min(prices), max(prices)
    span = max(hi - lo, 0.25)
    pad = span * 0.12
    lo -= pad
    hi += pad
    span = hi - lo

    def x(index: int) -> float:
        return pl + (index / max(len(rows) - 1, 1)) * pw

    def y(price: float) -> float:
        return pt + (1 - ((price - lo) / span)) * ph

    path = " ".join(
        f'{"M" if i == 0 else "L"}{x(i):.1f},{y(price):.1f}'
        for i, price in enumerate(closes)
    )

    y_ticks = []
    for idx in range(6):
        frac = idx / 5
        price = hi - frac * span
        yy = y(price)
        y_ticks.append(
            f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1" opacity="0.7"/>'
            f'<text x="{pl-10}" y="{yy+4:.1f}" fill="{MUTE}" font-size="11" text-anchor="end">{price:.2f}</text>'
        )

    x_ticks = []
    for idx, minute in enumerate(minutes):
        if idx in {0, len(minutes)//2, len(minutes)-1}:
            xx = x(idx)
            hh = 9 + (30 + minute) // 60
            mm = (30 + minute) % 60
            label = f"{hh:02d}:{mm:02d}"
            x_ticks.append(
                f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt+ph}" stroke="{GRID}" stroke-width="1" opacity="0.5"/>'
                f'<text x="{xx:.1f}" y="{height-12}" fill="{MUTE}" font-size="11" text-anchor="middle">{label}</text>'
            )

    overlay_lines = []
    label_slots = 0
    for level in sorted(overlays, key=lambda item: item["value"], reverse=True):
        yy = y(level["value"])
        overlay_lines.append(
            f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+pw}" y2="{yy:.1f}" stroke="{level["color"]}" '
            f'stroke-width="{level["width"]}" stroke-dasharray="{level["dash"]}" opacity="0.95"/>'
        )
        label_y = pt + 18 + label_slots * 18
        overlay_lines.append(
            f'<rect x="{pl+pw+10}" y="{label_y-12}" width="132" height="16" fill="{BG}" opacity="0.92" rx="4"/>'
            f'<text x="{pl+pw+16}" y="{label_y:.1f}" fill="{level["color"]}" font-size="11">{_esc(level["label"])} {_esc(_fmt_price(level["value"]))}</text>'
        )
        label_slots += 1

    spot = signal.get("spot", closes[-1])
    location_hint = "ABOVE magnet" if spot > signal.get("pin", spot) else "BELOW magnet"
    if abs(spot - signal.get("pin", spot)) <= 0.15:
        location_hint = "ON magnet"

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'style="width:100%;height:auto;display:block;background:#0b1220;border-radius:14px;border:1px solid #30363d">'
        f'<rect width="{width}" height="{height}" fill="#0b1220"/>'
        + "".join(y_ticks)
        + "".join(x_ticks)
        + "".join(overlay_lines)
        + f'<path d="{path}" fill="none" stroke="{FG}" stroke-width="2.5"/>'
        + ''.join(
            f'<circle cx="{x(i):.1f}" cy="{y(price):.1f}" r="2.2" fill="{FG}" opacity="{0.35 if i < len(closes)-1 else 1}"/>'
            for i, price in enumerate(closes[-12:])
        )
        + f'<text x="{pl}" y="18" fill="{FG}" font-size="15" font-weight="bold">SPY intraday location board</text>'
        + f'<text x="{pl+230}" y="18" fill="{MUTE}" font-size="12">What should I do? Read location first. Current {location_hint}.</text>'
        + '</svg>'
    )



def _panel_regime(signal: dict) -> str:
    regime = signal.get("gamma_regime", "?").upper()
    tag = signal.get("setup_tag", "No setup")
    bias = signal.get("setup_bias", "No bias")
    color = BLUE if signal.get("gamma_regime") == "positive" else RED if signal.get("gamma_regime") == "negative" else AMBER
    body = (
        f'<div style="font-size:24px;font-weight:bold;color:{color}">{_esc(tag)}</div>'
        f'<div style="font-size:13px;color:{FG};margin-top:6px">{_esc(bias)}</div>'
        f'<div style="display:flex;gap:12px;margin-top:10px;font-size:12px;color:{MUTE}">'
        f'<span>Gamma <b style="color:{FG}">{_esc(regime)}</b></span>'
        f'<span>Pin <b style="color:{FG}">{_esc(_fmt_price(signal.get("pin")))}</b></span>'
        '</div>'
    )
    return _card("Regime", body, color)



def _panel_gate(ctx) -> str:
    if not getattr(ctx, "available", False):
        return _card("Gate", f'<div style="font-size:18px;color:{AMBER}">Gate unavailable</div><div style="font-size:12px;color:{MUTE};margin-top:6px">{_esc(getattr(ctx, "note", ""))}</div>', AMBER)
    fresh = getattr(ctx, "fresh", False)
    color = GREEN if fresh else RED
    prob_trend = getattr(ctx, "prob_trend", None)
    prob_range = getattr(ctx, "prob_range", None)
    body = (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<div style="font-size:24px;font-weight:bold;color:{FG}">{_esc(getattr(ctx, "final_bias", "?"))}</div>'
        f'<div style="font-size:12px;color:{color}">{"fresh" if fresh else "stale"} • age {getattr(ctx, "age_days", "?")}d</div></div>'
        f'<div style="font-size:12px;color:{MUTE};margin-top:8px">trend {prob_trend:.2f} • range {prob_range:.2f}</div>'
        f'<div style="font-size:12px;color:{MUTE};margin-top:4px">execution score <b style="color:{FG}">{getattr(ctx, "execution_score", "?")}</b>/100</div>'
    )
    return _card("Gate", body, color)



def _panel_play(signal: dict, ctx, decision: dict) -> str:
    heading, reason, extra = _play_text(decision, signal, ctx)
    color = GREEN if heading != "STAND DOWN" else AMBER
    body = (
        f'<div style="font-size:28px;font-weight:bold;color:{color};line-height:1.05">{_esc(heading)}</div>'
        f'<div style="font-size:14px;color:{FG};margin-top:8px">{_esc(reason)}</div>'
        f'<div style="font-size:12px;color:{MUTE};margin-top:8px">{_esc(extra)}</div>'
    )
    return _card("Play", body, color)



def _panel_risk(signal: dict) -> str:
    mag = signal.get("magnitude", {})
    spot = signal.get("spot")
    put_wall = signal.get("put_wall")
    call_wall = signal.get("call_wall")
    pin = signal.get("pin")
    exp_usd = mag.get("exp_move_realized_usd")
    support_gap = (spot - put_wall) if isinstance(spot, (int, float)) and isinstance(put_wall, (int, float)) else None
    resistance_gap = (call_wall - spot) if isinstance(spot, (int, float)) and isinstance(call_wall, (int, float)) else None
    magnet_gap = abs(pin - spot) if isinstance(spot, (int, float)) and isinstance(pin, (int, float)) else None
    body = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">'
        f'<div><div style="color:{MUTE}">Expected move</div><div style="color:{FG};font-weight:bold">{_esc(_fmt_price(exp_usd))}</div></div>'
        f'<div><div style="color:{MUTE}">Premium read</div><div style="color:{FG};font-weight:bold">{_esc(str(mag.get("premium_read", "?")).upper())}</div></div>'
        f'<div><div style="color:{MUTE}">To put wall</div><div style="color:{FG};font-weight:bold">{_esc(_fmt_price(support_gap))}</div></div>'
        f'<div><div style="color:{MUTE}">To call wall</div><div style="color:{FG};font-weight:bold">{_esc(_fmt_price(resistance_gap))}</div></div>'
        f'<div><div style="color:{MUTE}">To magnet</div><div style="color:{FG};font-weight:bold">{_esc(_fmt_price(magnet_gap))}</div></div>'
        f'<div><div style="color:{MUTE}">IV</div><div style="color:{FG};font-weight:bold">{signal.get("atm_iv", 0) * 100:.1f}%</div></div>'
        '</div>'
    )
    return _card("Risk", body, RED)



def _panel_confidence(ctx, rec: dict) -> str:
    label, color = _confidence_label(ctx, rec)
    body = (
        f'<div style="font-size:28px;font-weight:bold;color:{color}">{label}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;font-size:12px">'
        f'<div><div style="color:{MUTE}">Regime acc</div><div style="color:{FG};font-weight:bold">{rec.get("regime_accuracy", 0) * 100:.1f}%</div></div>'
        f'<div><div style="color:{MUTE}">Brier</div><div style="color:{FG};font-weight:bold">{rec.get("brier", 0):.3f}</div></div>'
        f'<div><div style="color:{MUTE}">Sessions</div><div style="color:{FG};font-weight:bold">{rec.get("sessions", 0)}</div></div>'
        f'<div><div style="color:{MUTE}">Trend base</div><div style="color:{FG};font-weight:bold">{rec.get("trend_base_rate", 0) * 100:.1f}%</div></div>'
        '</div>'
    )
    return _card("Confidence", body, color)



def render() -> str:
    signal = _signal()
    ctx = load_execution_state(symbol="SPY")
    decision = decide(signal, analytics=ctx) if signal else {"action": "stand_down", "reason": "no signal", "intent": None}
    rec = _scoreboard()
    rows = _rows()
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    spot = signal.get("spot", 0)
    change = signal.get("day_chg", 0)
    change_color = GREEN if change >= 0 else RED

    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="45">
  <title>SharpEdge Pilot Board</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:{BG}; color:{FG}; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .page {{ padding:14px; max-width:1500px; margin:0 auto; }}
    .hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:12px; }}
    .hero h1 {{ margin:0; font-size:28px; }}
    .hero .sub {{ color:{MUTE}; font-size:12px; margin-top:4px; }}
    .hero .spot {{ font-size:34px; font-weight:700; text-align:right; }}
    .layout {{ display:grid; grid-template-columns:minmax(0, 2.2fr) minmax(320px, 1fr); gap:14px; align-items:start; }}
    .chart-wrap {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:16px; padding:12px; min-height:60vh; }}
    .chart-note {{ color:{MUTE}; font-size:11px; margin-top:8px; display:flex; gap:14px; flex-wrap:wrap; }}
    @media (max-width: 980px) {{ .layout {{ grid-template-columns:1fr; }} .chart-wrap {{ min-height:auto; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>SharpEdge Pilot Board - SPY</h1>
        <div class="sub">Fast read for tired pilots • updated {now}</div>
      </div>
      <div class="spot">{_fmt_price(spot)} <span style="font-size:17px;color:{change_color}">{_fmt_pct(change)}</span></div>
    </section>
    <section class="layout">
      <div>
        <div class="chart-wrap">
          {_chart_svg(rows, signal)}
          <div class="chart-note">
            <span>Chart owns the decision: location first, story second.</span>
            <span>Expected move overlay uses the realized intraday model.</span>
            <span>Channel boundaries come from the current microstructure channel.</span>
          </div>
        </div>
      </div>
      <aside>
        {_panel_regime(signal)}
        {_panel_gate(ctx)}
        {_panel_play(signal, ctx, decision)}
        {_panel_risk(signal)}
        {_panel_confidence(ctx, rec)}
      </aside>
    </section>
  </div>
</body>
</html>'''



def main() -> int:
    out = OUT_DIR / "pilot_board.html"
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
