#!/usr/bin/env python3
"""SharpEdge Command Deck: one screen showing the whole system thinking.

Renders the full chain on a single auto-refreshing page:

  SIGNAL (live)  ->  GATE (final_bias + probs + score)  ->  DECISION (runner/fade
  + the actual option contract)  +  SCOREBOARD (reconciliation: accuracy, Brier,
  calibration, with honest n= labels).

Reads artifacts that already exist (signal.json, execution_state_daily.csv,
reconcile_summary.json) and reuses the Bridge's tested decide()/load_execution_
state so the deck shows EXACTLY what the trade engine sees. Stdlib only, so it
runs with the same python as make_cockpit.py and never needs a server/backend.

  python3 make_command_deck.py          # write command_deck.html
"""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SYS_OUT = Path(os.path.expanduser("~/SharpEdge-System/outputs"))
_BRIDGE_SRC = Path(os.path.expanduser("~/SharpEdge-Robinhood-Bridge/src"))
if _BRIDGE_SRC.exists():
    sys.path.insert(0, str(_BRIDGE_SRC))

# Bridge is pure-stdlib; import is safe with the cockpit's python.
from sharpedge_robinhood_bridge.analytics_context import load_execution_state  # noqa: E402
from sharpedge_robinhood_bridge.trade_intent import decide, load_signal  # noqa: E402

BG, FG, MUTE = "#0d1117", "#e6edf3", "#7d8590"
GREEN, RED, AMBER, BLUE = "#26a641", "#f85149", "#d29922", "#58a6ff"


def _esc(x) -> str:
    return html.escape(str(x))


def _card(inner: str, accent: str = "#21262d") -> str:
    return (f'<div style="border-left:4px solid {accent};background:#161b22;'
            f'padding:12px 14px;margin:10px 0;border-radius:8px">{inner}</div>')


def panel_signal(sig: dict) -> str:
    if not sig:
        return _card('<b style="color:#f85149">No signal.json - run make_cockpit.py</b>', RED)
    spot = sig.get("spot", 0)
    chg = sig.get("day_chg", 0)
    col = GREEN if chg >= 0 else RED
    regime = sig.get("gamma_regime", "?")
    rcol = RED if regime == "negative" else BLUE if regime == "positive" else MUTE
    rows = [
        ("gamma regime", regime, rcol),
        ("vs VWAP", f"{sig.get('vs_vwap', 0):+.2f}%", FG),
        ("15m mom", f"{sig.get('mom15', 0):+.2f}%", FG),
        ("vol mult", f"{sig.get('vol_mult', 0):.1f}x", FG),
        ("call wall", f"${sig.get('call_wall', '?')}", FG),
        ("put wall", f"${sig.get('put_wall', '?')}", FG),
        ("pin / magnet", f"${sig.get('pin', '?')}", AMBER),
    ]
    cells = "".join(
        f'<div style="flex:1;min-width:90px"><div style="color:{MUTE};font-size:11px">'
        f'{_esc(k)}</div><div style="color:{c};font-weight:bold">{_esc(v)}</div></div>'
        for k, v, c in rows
    )
    return _card(
        f'<div style="font-size:24px;font-weight:bold">${_esc(spot)} '
        f'<span style="font-size:15px;color:{col}">{chg:+.2f}% today</span></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px">{cells}</div>',
        BLUE,
    )


def _prob_bar(pt, pr) -> str:
    if pt is None or pr is None:
        return f'<div style="color:{MUTE};font-size:12px">probs unavailable</div>'
    tpct = int(round(pt * 100))
    return (
        f'<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;margin:4px 0">'
        f'<div style="width:{tpct}%;background:{RED}"></div>'
        f'<div style="width:{100 - tpct}%;background:{BLUE}"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;color:{MUTE}">'
        f'<span>trend {pt:.2f}</span><span>range {pr:.2f}</span></div>'
    )


def panel_gate(ctx) -> str:
    if not ctx.available:
        return _card(f'<b style="color:{AMBER}">Gate unavailable</b> '
                     f'<span style="color:{MUTE}">{_esc(ctx.note)}</span>', AMBER)
    fresh_col = GREEN if ctx.fresh else RED
    fresh_txt = f"fresh (age {ctx.age_days}d)" if ctx.fresh else f"STALE (age {ctx.age_days}d)"
    bias = ctx.final_bias or "?"
    score = ctx.execution_score
    score_txt = f"{score:.0f}" if score is not None else "?"
    return _card(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<b style="font-size:15px">GATE</b>'
        f'<span style="color:{fresh_col};font-size:12px">{_esc(fresh_txt)}</span></div>'
        f'<div style="font-size:20px;font-weight:bold;margin:4px 0">{_esc(bias)}</div>'
        f'{_prob_bar(ctx.prob_trend, ctx.prob_range)}'
        f'<div style="color:{MUTE};font-size:12px;margin-top:6px">execution score '
        f'<b style="color:{FG}">{_esc(score_txt)}</b>/100</div>',
        fresh_col,
    )


def panel_decision(decision: dict) -> str:
    action = decision.get("action", "?")
    reason = decision.get("reason", "")
    intent = decision.get("intent")
    if action == "trade" and intent is not None:
        leg = intent.option_legs[0] if intent.option_legs else {}
        contract = (f"{intent.symbol} {leg.get('expiry', '?')} "
                    f"{leg.get('strike', '?')} {str(leg.get('right', '?')).upper()}")
        body = (
            f'<div style="font-size:20px;font-weight:bold;color:{GREEN}">TRADE</div>'
            f'<div style="color:{FG};font-size:15px;margin:4px 0">{_esc(reason)}</div>'
            f'<div style="background:#0d1117;border-radius:6px;padding:8px;margin-top:6px">'
            f'<div style="font-size:16px;font-weight:bold">{_esc(contract)}</div>'
            f'<div style="color:{MUTE};font-size:12px">1 contract @ limit '
            f'${_esc(intent.limit_price)} (est prem ${_esc(leg.get("est_premium", "?"))})</div></div>'
            f'<div style="color:{MUTE};font-size:11px;margin-top:6px">awaiting operator confirm '
            f'- never auto-submitted</div>'
        )
        return _card(body, GREEN)
    return _card(
        f'<div style="font-size:20px;font-weight:bold;color:{AMBER}">STAND DOWN</div>'
        f'<div style="color:{FG};font-size:14px;margin-top:4px">{_esc(reason)}</div>',
        AMBER,
    )


def panel_scoreboard(rec: dict) -> str:
    if not rec:
        return _card(f'<span style="color:{MUTE}">No reconcile_summary.json yet</span>')
    ra = rec.get("regime_accuracy")
    da = rec.get("direction_accuracy")
    brier = rec.get("brier")
    base = rec.get("brier_baseline_constant")
    n_reg = rec.get("decisive_regime_calls", 0)
    n_dir = rec.get("directional_calls", 0)

    def pct(x):
        return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"

    brier_col = GREEN if (brier is not None and base is not None and brier < base) else MUTE
    rows = [
        ("regime accuracy", pct(ra), f"n={n_reg}", FG),
        ("direction accuracy", pct(da), f"n={n_dir}" + (" (thin!)" if n_dir < 20 else ""), FG),
        ("Brier", f"{brier:.3f}" if brier is not None else "n/a",
         f"base {base:.3f}" if base is not None else "", brier_col),
    ]
    cells = "".join(
        f'<div style="flex:1;min-width:110px"><div style="color:{MUTE};font-size:11px">{_esc(k)}</div>'
        f'<div style="color:{c};font-weight:bold;font-size:16px">{_esc(v)}</div>'
        f'<div style="color:{MUTE};font-size:10px">{_esc(sub)}</div></div>'
        for k, v, sub, c in rows
    )
    return _card(
        f'<b style="font-size:15px">SCOREBOARD</b> '
        f'<span style="color:{MUTE};font-size:11px">model vs reality, {rec.get("sessions", 0)} sessions</span>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px">{cells}</div>',
        BLUE,
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render() -> str:
    sig = load_signal()
    ctx = load_execution_state(symbol="SPY")
    decision = decide(sig, analytics=ctx) if sig else {"action": "stand_down",
                                                       "reason": "no signal", "intent": None}
    rec = _load_json(SYS_OUT / "reconcile_summary.json")
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="refresh" content="45">'
        '<title>SharpEdge Command Deck</title></head>'
        f'<body style="margin:0;background:{BG};color:{FG};font-family:monospace">'
        '<div style="padding:12px;max-width:680px;margin:auto">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline">'
        '<h2 style="margin:0;font-size:18px">SharpEdge Command Deck - SPY</h2>'
        f'<span style="color:{MUTE};font-size:11px">{now} | auto 45s</span></div>'
        f'<div style="color:{MUTE};font-size:11px;margin:2px 0 8px">'
        'SIGNAL &#8594; GATE &#8594; DECISION + live scoreboard</div>'
        f'{panel_signal(sig)}'
        f'{panel_gate(ctx)}'
        f'{panel_decision(decision)}'
        f'{panel_scoreboard(rec)}'
        f'<p style="color:#484f58;font-size:11px;margin-top:14px">Decision support only - '
        'every trade is operator-confirmed. n= labels are honest; thin samples are not edge.</p>'
        '</div></body></html>'
    )


def main() -> int:
    out = Path(OUT_DIR) / "command_deck.html"
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
