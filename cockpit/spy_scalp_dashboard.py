"""Focused SPY options scalping dashboard.

This module intentionally stays small and deterministic: it turns the existing
SharpEdge live SPY rows/options read into an if-then dashboard for quick
0DTE/1DTE trend-following scalps. It is decision support, not execution.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from spy_scalp_chart import render_spy_scalp_chart_svg

Row = tuple[int, float, float, float, float, int]


@dataclass(frozen=True)
class Candle:
    minute: int
    open: float
    high: float
    low: float
    close: float
    volume: int


def _regular_rows(rows: Iterable[Row]) -> list[Row]:
    return [row for row in rows if 0 <= int(row[0]) < 390]


def _ema(values: list[float], period: int) -> float | None:
    if not values:
        return None
    alpha = 2 / (period + 1)
    value = values[0]
    for item in values[1:]:
        value = item * alpha + value * (1 - alpha)
    return value


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(values[-period - 1 : -1], values[-period:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _aggregate_5m(rows: list[Row]) -> list[Candle]:
    candles: list[Candle] = []
    bucket: list[Row] = []
    bucket_start: int | None = None
    for row in _regular_rows(rows):
        minute = int(row[0])
        start = minute - (minute % 5)
        if bucket_start is None:
            bucket_start = start
        if start != bucket_start:
            candles.append(_bucket_to_candle(bucket_start, bucket))
            bucket = []
            bucket_start = start
        bucket.append(row)
    if bucket and bucket_start is not None:
        candles.append(_bucket_to_candle(bucket_start, bucket))
    return candles


def _bucket_to_candle(start: int, rows: list[Row]) -> Candle:
    return Candle(
        minute=start,
        open=float(rows[0][1]),
        high=max(float(row[2]) for row in rows),
        low=min(float(row[3]) for row in rows),
        close=float(rows[-1][4]),
        volume=sum(int(row[5]) for row in rows),
    )


def _opening_range(rows: list[Row], minutes: int = 15) -> dict:
    or_rows = [row for row in _regular_rows(rows) if 0 <= int(row[0]) < minutes]
    if not or_rows:
        return {"available": False, "minutes": minutes}
    return {
        "available": True,
        "minutes": minutes,
        "high": round(max(float(row[2]) for row in or_rows), 2),
        "low": round(min(float(row[3]) for row in or_rows), 2),
        "volume": sum(int(row[5]) for row in or_rows),
    }


def _volume_read(rows: list[Row]) -> dict:
    regular = _regular_rows(rows)
    if len(regular) < 10:
        return {"state": "unknown", "multiple": None, "reason": "not enough bars"}
    last5 = sum(row[5] for row in regular[-5:]) / 5
    baseline_values = [row[5] for row in regular[:-5]] or [row[5] for row in regular]
    baseline = median(baseline_values) or 1
    multiple = last5 / baseline
    if multiple >= 1.5:
        state = "high"
        reason = "peak participation supports fast scalps"
    elif multiple >= 0.8:
        state = "normal"
        reason = "participation is tradable but not screaming"
    else:
        state = "thin"
        reason = "thin participation; slippage/chop risk rises"
    return {"state": state, "multiple": round(multiple, 2), "reason": reason}


def _time_window(minute: int | None) -> dict:
    if minute is None:
        return {"state": "unknown", "label": "unknown", "score": 0}
    if 0 <= minute <= 60:
        return {"state": "prime", "label": "opening drive", "score": 2}
    if 330 <= minute <= 390:
        return {"state": "prime", "label": "closing drive", "score": 2}
    if 120 <= minute <= 300:
        return {"state": "avoid", "label": "midday theta/chop", "score": -2}
    return {"state": "ok", "label": "secondary window", "score": 0}


def _contract_read(op: dict, bias: str) -> dict:
    side = "call" if bias == "CALLS" else "put" if bias == "PUTS" else "call/put"
    delta_key = f"atm_{side}_delta" if side in {"call", "put"} else "atm_call_delta"
    spread_key = (
        f"atm_{side}_spread_pct" if side in {"call", "put"} else "atm_call_spread_pct"
    )
    raw_delta = op.get(delta_key)
    delta = abs(float(raw_delta)) if isinstance(raw_delta, (int, float)) else None
    spread_pct = op.get(spread_key)
    spread_pct = float(spread_pct) if isinstance(spread_pct, (int, float)) else None
    delta_ok = delta is not None and 0.45 <= delta <= 0.55
    spread_ok = spread_pct is None or spread_pct <= 0.08
    expiry = op.get("exp") or "nearest 0DTE/1DTE"
    return {
        "side": side,
        "expiry": expiry,
        "strike": op.get("atm_strike"),
        "delta": round(delta, 3) if delta is not None else None,
        "delta_ok": delta_ok,
        "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "spread_ok": spread_ok,
        "rule": "ATM/slightly ITM, 0.45-0.55 delta, high OI, limit order only",
    }


def build_spy_scalp_packet(rows: list[Row], pa: dict, op: dict, stamp: str) -> dict:
    """Build a compact if-then scalp read from live SPY rows."""
    regular = _regular_rows(rows)
    closes_1m = [float(row[4]) for row in regular]
    candles_5m = _aggregate_5m(rows)
    closes_5m = [candle.close for candle in candles_5m]
    spot = float(pa.get("spot") or (closes_1m[-1] if closes_1m else 0))
    vwap = float(pa.get("vwap") or spot or 1)
    latest_minute = int(regular[-1][0]) if regular else None
    or15 = _opening_range(rows)
    last_5m = candles_5m[-1] if candles_5m else None
    ema9_1m = _ema(closes_1m[-60:], 9)
    ema20_1m = _ema(closes_1m[-80:], 20)
    ema9_5m = _ema(closes_5m[-36:], 9)
    ema20_5m = _ema(closes_5m[-48:], 20)
    rsi_1m = _rsi(closes_1m)
    rsi_5m = _rsi(closes_5m)
    vs_vwap_pct = (spot - vwap) / vwap * 100 if vwap else 0.0

    trend = _trend_bias(spot, vwap, ema9_5m, ema20_5m, closes_5m)
    trigger = _opening_range_trigger(or15, last_5m, spot, ema9_1m, ema20_1m)
    overextension = _overextension_read(trend["bias"], rsi_1m, vs_vwap_pct)
    volume = _volume_read(rows)
    window = _time_window(latest_minute)
    score = _score_setup(trend, trigger, overextension, volume, window)
    bias = trigger["bias"] if trigger["bias"] != "NEUTRAL" else trend["bias"]
    contract = _contract_read(op, bias)
    checklist = _checklist(trend, trigger, overextension, volume, window, contract)

    return {
        "schema": "sharpedge.spy_scalp_dashboard.v1",
        "symbol": "SPY",
        "stamp": stamp,
        "spot": round(spot, 2),
        "latest_minute": latest_minute,
        "bias": bias,
        "score": score,
        "status": _status(score, trigger, overextension, window),
        "opening_range": or15,
        "trend": trend,
        "trigger": trigger,
        "risk": _risk_box(),
        "contract": contract,
        "time_window": window,
        "volume": volume,
        "indicators": {
            "ema9_1m": _round_or_none(ema9_1m),
            "ema20_1m": _round_or_none(ema20_1m),
            "ema9_5m": _round_or_none(ema9_5m),
            "ema20_5m": _round_or_none(ema20_5m),
            "rsi_1m": _round_or_none(rsi_1m, 1),
            "rsi_5m": _round_or_none(rsi_5m, 1),
            "vwap": round(vwap, 2),
            "vs_vwap_pct": round(vs_vwap_pct, 3),
        },
        "overextension": overextension,
        "checklist": checklist,
    }


def _trend_bias(
    spot: float,
    vwap: float,
    ema9_5m: float | None,
    ema20_5m: float | None,
    closes_5m: list[float],
) -> dict:
    if ema9_5m is None or ema20_5m is None or len(closes_5m) < 4:
        return {
            "bias": "NEUTRAL",
            "state": "insufficient",
            "reason": "need more 5m bars",
        }
    slope = closes_5m[-1] - closes_5m[-4]
    if spot > vwap and ema9_5m > ema20_5m and slope > 0:
        return {
            "bias": "CALLS",
            "state": "uptrend",
            "reason": "5m EMA9>EMA20, price above VWAP, slope up",
        }
    if spot < vwap and ema9_5m < ema20_5m and slope < 0:
        return {
            "bias": "PUTS",
            "state": "downtrend",
            "reason": "5m EMA9<EMA20, price below VWAP, slope down",
        }
    return {
        "bias": "NEUTRAL",
        "state": "mixed",
        "reason": "5m trend/VWAP are not aligned",
    }


def _opening_range_trigger(
    or15: dict,
    last_5m: Candle | None,
    spot: float,
    ema9_1m: float | None,
    ema20_1m: float | None,
) -> dict:
    if not or15.get("available") or last_5m is None:
        return {
            "bias": "NEUTRAL",
            "state": "waiting",
            "reason": "opening range not ready",
        }
    high = float(or15["high"])
    low = float(or15["low"])
    above_emas = (
        ema9_1m is not None and ema20_1m is not None and spot > ema9_1m > ema20_1m
    )
    below_emas = (
        ema9_1m is not None and ema20_1m is not None and spot < ema9_1m < ema20_1m
    )
    if last_5m.minute < 15:
        return {
            "bias": "NEUTRAL",
            "state": "waiting",
            "reason": "wait for a 5m close after OR15",
        }
    if last_5m.close > high and spot >= high and above_emas:
        return {
            "bias": "CALLS",
            "state": "armed",
            "reason": f"5m closed above ORH ${high:.2f} and 1m holds EMA stack",
        }
    if last_5m.close < low and spot <= low and below_emas:
        return {
            "bias": "PUTS",
            "state": "armed",
            "reason": f"5m closed below ORL ${low:.2f} and 1m holds EMA stack",
        }
    return {
        "bias": "NEUTRAL",
        "state": "waiting",
        "reason": "no clean OR15 break-and-hold trigger",
    }


def _overextension_read(bias: str, rsi_1m: float | None, vs_vwap_pct: float) -> dict:
    if rsi_1m is None:
        return {"state": "unknown", "ok": False, "reason": "RSI not ready"}
    if bias == "CALLS" and (rsi_1m >= 72 or vs_vwap_pct > 0.45):
        return {
            "state": "chasing",
            "ok": False,
            "reason": "call side is stretched; wait for pullback/hold",
        }
    if bias == "PUTS" and (rsi_1m <= 28 or vs_vwap_pct < -0.45):
        return {
            "state": "chasing",
            "ok": False,
            "reason": "put side is stretched; wait for bounce/fail",
        }
    if 35 <= rsi_1m <= 68:
        return {"state": "clean", "ok": True, "reason": "RSI is not at chase levels"}
    return {
        "state": "caution",
        "ok": True,
        "reason": "RSI is warm; size small and demand limit fills",
    }


def _score_setup(
    trend: dict, trigger: dict, overextension: dict, volume: dict, window: dict
) -> int:
    score = 0
    if trend["bias"] != "NEUTRAL":
        score += 25
    if trigger["state"] == "armed" and trigger["bias"] == trend["bias"]:
        score += 35
    elif trigger["state"] == "armed":
        score += 20
    if overextension.get("ok"):
        score += 15
    if volume.get("state") == "high":
        score += 15
    elif volume.get("state") == "normal":
        score += 7
    score += int(window.get("score", 0)) * 5
    return max(0, min(100, score))


def _status(score: int, trigger: dict, overextension: dict, window: dict) -> str:
    if window.get("state") == "avoid":
        return "AVOID MIDDAY CHOP"
    if trigger.get("state") != "armed":
        return "WAIT FOR OR15 BREAK"
    if not overextension.get("ok"):
        return "DO NOT CHASE"
    if score >= 75:
        return "SCALP SETUP ARMED"
    if score >= 55:
        return "CAUTION / HALF SIZE ONLY"
    return "NO TRADE"


def _risk_box() -> dict:
    return {
        "position_size": "0.5%-1.0% account equity max",
        "profit_target": "+15% to +30% option premium",
        "stop_loss": "-10% to -15% option premium hard stop",
        "time_stop": "exit if sideways for 15 minutes",
        "daily_drawdown": "stop trading at -3% to -5% day drawdown",
        "order_type": "limit orders only; no market orders, no chase fills",
    }


def _checklist(
    trend: dict,
    trigger: dict,
    overextension: dict,
    volume: dict,
    window: dict,
    contract: dict,
) -> list[dict]:
    return [
        {
            "label": "5m trend aligned",
            "ok": trend["bias"] != "NEUTRAL",
            "detail": trend["reason"],
        },
        {
            "label": "OR15 break confirmed",
            "ok": trigger["state"] == "armed",
            "detail": trigger["reason"],
        },
        {
            "label": "Not chasing RSI/VWAP",
            "ok": bool(overextension.get("ok")),
            "detail": overextension["reason"],
        },
        {
            "label": "Volume supports scalp",
            "ok": volume["state"] in {"high", "normal"},
            "detail": volume["reason"],
        },
        {
            "label": "Tradable time window",
            "ok": window["state"] != "avoid",
            "detail": window["label"],
        },
        {
            "label": "Contract/spread sane",
            "ok": contract["spread_ok"],
            "detail": contract["rule"],
        },
    ]


def render_spy_scalp_dashboard_html(
    packet: dict, chart_href: str = "spy_scalp_chart.svg"
) -> str:
    status_class = _css_class(packet.get("status", ""))
    checks = "".join(_render_check(item) for item in packet.get("checklist", []))
    risk = packet.get("risk") or {}
    indicators = packet.get("indicators") or {}
    contract = packet.get("contract") or {}
    opening_range = packet.get("opening_range") or {}
    html_packet = html.escape(json.dumps(packet, indent=2))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta http-equiv=\"refresh\" content=\"2\">
  <meta http-equiv=\"Cache-Control\" content=\"no-store, no-cache, must-revalidate, max-age=0\">
  <meta http-equiv=\"Pragma\" content=\"no-cache\">
  <title>SharpEdge SPY Scalp Dashboard</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <div>
      <p class=\"eyebrow\">SharpEdge one-thing dashboard</p>
      <h1>SPY Options Scalp</h1>
      <p>Trend-following OR15 checklist for 0DTE/1DTE ATM/ITM contracts. Decision support only, chief.</p>
    </div>
    <div class=\"status {status_class}\">{html.escape(packet.get("status", "UNKNOWN"))}</div>
  </header>
  <main>
    <section class=\"hero\">
      <div><span>SPY</span><strong>${packet.get("spot", 0):.2f}</strong></div>
      <div><span>Bias</span><strong>{html.escape(str(packet.get("bias", "NEUTRAL")))}</strong></div>
      <div><span>Score</span><strong>{int(packet.get("score", 0))}/100</strong></div>
      <div><span>Window</span><strong>{html.escape(str((packet.get("time_window") or {}).get("label", "unknown")))}</strong></div>
    </section>
    <section class=\"grid\">
      <article class=\"card wide\">
        <h2>If-Then Checklist</h2>
        <div class=\"checks\">{checks}</div>
      </article>
      <article class=\"card\">
        <h2>Opening Range</h2>
        <p>OR{opening_range.get("minutes", 15)} high: <b>${opening_range.get("high", 0):.2f}</b></p>
        <p>OR{opening_range.get("minutes", 15)} low: <b>${opening_range.get("low", 0):.2f}</b></p>
        <p>{html.escape(str((packet.get("trigger") or {}).get("reason", "waiting")))}</p>
      </article>
      <article class=\"card\">
        <h2>Indicators</h2>
        <p>VWAP <b>${indicators.get("vwap", 0):.2f}</b> ({indicators.get("vs_vwap_pct", 0):+.3f}%)</p>
        <p>1m EMA 9/20: <b>{indicators.get("ema9_1m")}</b> / <b>{indicators.get("ema20_1m")}</b></p>
        <p>5m EMA 9/20: <b>{indicators.get("ema9_5m")}</b> / <b>{indicators.get("ema20_5m")}</b></p>
        <p>RSI 1m/5m: <b>{indicators.get("rsi_1m")}</b> / <b>{indicators.get("rsi_5m")}</b></p>
      </article>
      <article class=\"card\">
        <h2>Contract</h2>
        <p>Target: <b>{html.escape(str(contract.get("side", "call/put"))).upper()}</b></p>
        <p>Expiry: <b>{html.escape(str(contract.get("expiry", "nearest")))}</b></p>
        <p>ATM strike: <b>{contract.get("strike")}</b> | delta <b>{contract.get("delta")}</b></p>
        <p>Spread: <b>{contract.get("spread_pct")}</b> | limit order only</p>
      </article>
      <article class=\"card\">
        <h2>Risk Box</h2>
        <p>{html.escape(risk.get("position_size", ""))}</p>
        <p>{html.escape(risk.get("profit_target", ""))}</p>
        <p>{html.escape(risk.get("stop_loss", ""))}</p>
        <p>{html.escape(risk.get("time_stop", ""))}</p>
      </article>
      <article class=\"card wide\">
        <h2>Entry Map</h2>
        <p class=\"chart-note\">Candles with OR15, VWAP, EMA9/EMA20, 20-bar channel, 8-bar trigger channel, and the exact level the scalp needs to hold.</p>
        <img src=\"{html.escape(chart_href)}\" alt=\"SPY scalp entry map\">
      </article>
    </section>
    <details>
      <summary>Raw scalp packet</summary>
      <pre>{html_packet}</pre>
    </details>
  </main>
</body>
</html>
"""


def write_spy_scalp_dashboard(
    rows: list[Row],
    pa: dict,
    op: dict,
    stamp: str,
    cockpit_dir: str,
    outputs_dir: str,
) -> dict:
    packet = build_spy_scalp_packet(rows, pa, op, stamp)
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(outputs_dir, "spy_scalp_dashboard.json").write_text(
        json.dumps(packet, indent=2)
    )
    Path(cockpit_dir, "spy_scalp_chart.svg").write_text(
        render_spy_scalp_chart_svg(rows, packet)
    )
    Path(cockpit_dir, "spy_scalp_dashboard.html").write_text(
        render_spy_scalp_dashboard_html(packet)
    )
    return packet


def _render_check(item: dict) -> str:
    icon = "OK" if item.get("ok") else "WAIT"
    state = "ok" if item.get("ok") else "wait"
    return (
        f'<div class="check {state}"><b>{icon} {html.escape(str(item.get("label", "")))}</b>'
        f"<span>{html.escape(str(item.get('detail', '')))}</span></div>"
    )


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if isinstance(value, (int, float)) else None


def _css_class(status: str) -> str:
    if "ARMED" in status:
        return "armed"
    if "CHASE" in status or "AVOID" in status:
        return "danger"
    if "CAUTION" in status:
        return "caution"
    return "wait"


def _css() -> str:
    return """
:root { color-scheme: dark; --bg:#080b10; --card:#111823; --muted:#8ea0b5; --text:#ecf4ff; --ok:#36d399; --warn:#fbbf24; --bad:#fb7185; --line:#263244; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, system-ui, -apple-system, sans-serif; background: radial-gradient(circle at top, #152033, var(--bg)); color:var(--text); }
header { display:flex; justify-content:space-between; gap:18px; align-items:center; padding:24px; border-bottom:1px solid var(--line); }
h1 { margin:0; font-size: clamp(2rem, 7vw, 4.5rem); letter-spacing:-0.06em; }
p { color:var(--muted); margin:8px 0; line-height:1.45; }
.eyebrow { text-transform:uppercase; letter-spacing:0.16em; color:#7dd3fc; font-weight:800; font-size:0.75rem; }
.status { padding:16px 18px; border-radius:18px; font-weight:900; text-align:center; min-width:150px; box-shadow:0 10px 30px #0006; }
.status.armed { background:color-mix(in srgb, var(--ok) 24%, #0b1118); color:var(--ok); }
.status.danger { background:color-mix(in srgb, var(--bad) 24%, #0b1118); color:var(--bad); }
.status.caution { background:color-mix(in srgb, var(--warn) 24%, #0b1118); color:var(--warn); }
.status.wait { background:#172033; color:#cbd5e1; }
main { padding:24px; max-width:1180px; margin:auto; }
.hero { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
.hero div, .card { background:linear-gradient(180deg,#121a27,#0d131d); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:0 14px 40px #0005; }
.hero span { display:block; color:var(--muted); text-transform:uppercase; font-size:0.72rem; letter-spacing:0.14em; }
.hero strong { display:block; margin-top:4px; font-size:1.8rem; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.card.wide { grid-column:1/-1; }
.card h2 { margin:0 0 12px; }
.checks { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.check { padding:12px; border-radius:14px; border:1px solid var(--line); background:#0a1019; }
.check.ok { border-color:#23684d; }
.check.wait { border-color:#66562a; }
.check b, .check span { display:block; }
.check span { color:var(--muted); margin-top:4px; }
.chart-note { margin-top:-4px; margin-bottom:14px; }
img { width:100%; border-radius:14px; background:#07101a; }
details { margin-top:16px; color:var(--muted); }
pre { overflow:auto; background:#05070b; border:1px solid var(--line); border-radius:14px; padding:14px; }
@media (max-width: 760px) { header { flex-direction:column; align-items:stretch; } .hero, .grid, .checks { grid-template-columns:1fr; } }
"""
