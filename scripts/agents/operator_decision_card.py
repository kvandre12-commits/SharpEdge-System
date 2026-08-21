#!/usr/bin/env python3
"""Build one compact, fail-closed operator decision artifact."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = Path("outputs")
COCKPIT_DIR = Path("cockpit")
DEFAULT_SIGNAL = OUTDIR / "signal.json"
DEFAULT_EXPRESSION = OUTDIR / "spy_option_expression.json"
DEFAULT_APPROVAL = OUTDIR / "approval_decision.json"
DEFAULT_JSON = OUTDIR / "operator_decision_card.json"
DEFAULT_HTML = COCKPIT_DIR / "operator_decision_card.html"
MAX_SIGNAL_AGE_SECONDS = 120


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if not math.isnan(parsed) else None


def _timestamp_age_seconds(value: Any, now: datetime) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max((now - parsed.astimezone(UTC)).total_seconds(), 0.0)


def _direction(signal: dict[str, Any]) -> str:
    tag = str(signal.get("entry_setup_tag") or "").upper()
    bias = str(
        signal.get("entry_setup_bias")
        or (signal.get("trade_permission") or {}).get("bias")
        or ""
    ).upper()
    if tag in {"DOWNSIDE EXHAUSTION", "FAILED BREAKDOWN"}:
        return "CALLS"
    if tag in {"UPSIDE EXHAUSTION", "FAILED BREAKOUT"}:
        return "PUTS"
    if "CALL" in bias or "BULL" in bias:
        return "CALLS"
    if "PUT" in bias or "BEAR" in bias:
        return "PUTS"
    return "NONE"


def _reference_map(signal: dict[str, Any], direction: str) -> dict[str, Any]:
    spot = _number(signal.get("spot"))
    references = {
        "VWAP": _number(signal.get("vwap")),
        "EMA9": _number(signal.get("ema9")),
        "EMA20": _number(signal.get("ema20")),
    }
    available = [
        (name, value) for name, value in references.items() if value is not None
    ]
    nearest = (
        min(available, key=lambda item: abs(item[1] - spot))
        if available and spot
        else None
    )
    closest = (
        sorted(available, key=lambda item: abs(item[1] - spot))[:2] if spot else []
    )
    zone_values = [value for _, value in closest]
    ema9 = references["EMA9"]
    ema20 = references["EMA20"]
    vwap = references["VWAP"]
    trigger_inputs = [value for value in (ema9, vwap) if value is not None]
    invalidation_inputs = [value for value in (ema20, vwap) if value is not None]
    if direction == "CALLS":
        confirmation = max(trigger_inputs, default=None)
        invalidation = min(invalidation_inputs, default=None)
        confirmation_met = bool(
            spot is not None and confirmation is not None and spot >= confirmation
        )
        relation = "above" if confirmation_met else "below"
        trigger_rule = (
            "hold or successfully retest above the confirmation line"
            if relation == "above"
            else "reclaim and hold above the confirmation line"
        )
        invalidation_rule = "lose the invalidation line and fail the reclaim"
    elif direction == "PUTS":
        confirmation = min(trigger_inputs, default=None)
        invalidation = max(invalidation_inputs, default=None)
        confirmation_met = bool(
            spot is not None and confirmation is not None and spot <= confirmation
        )
        relation = "below" if confirmation_met else "above"
        trigger_rule = (
            "hold or successfully retest below the confirmation line"
            if relation == "below"
            else "reject and accept below the confirmation line"
        )
        invalidation_rule = "reclaim the invalidation line and hold above it"
    else:
        confirmation = None
        invalidation = None
        confirmation_met = False
        relation = "unavailable"
        trigger_rule = "directional confirmation unavailable"
        invalidation_rule = "directional invalidation unavailable"
    stack = "unavailable"
    if None not in (spot, ema9, ema20):
        if spot > ema9 > ema20:
            stack = "bullish"
        elif spot < ema9 < ema20:
            stack = "bearish"
        else:
            stack = "mixed"
    return {
        "spot": spot,
        "references": references,
        "nearest": {"name": nearest[0], "price": nearest[1]} if nearest else None,
        "bounce_zone": {
            "low": min(zone_values) if zone_values else None,
            "high": max(zone_values) if zone_values else None,
            "members": [name for name, _ in closest],
        },
        "ema_stack": stack,
        "ema9_slope_5": _number(signal.get("ema9_slope_5")),
        "confirmation_level": confirmation,
        "confirmation_relation": relation,
        "confirmation_met": confirmation_met,
        "trigger_rule": trigger_rule,
        "invalidation_level": invalidation,
        "invalidation_rule": invalidation_rule,
    }


def _expression(expression: dict[str, Any], direction: str) -> dict[str, Any]:
    branches = expression.get("branch_expressions") or []
    branch = next(
        (
            item
            for item in branches
            if str(item.get("direction") or "").upper() == direction
            and item.get("structure_family") != "no_forced_position"
        ),
        None,
    )
    if not branch:
        return {
            "available": False,
            "family": "unavailable",
            "label": "No quote-validated option structure",
            "reason": "Wait for fresh contracts; do not guess single-leg versus spread.",
            "pricing": {},
        }
    family = str(branch.get("structure_family") or "unknown")
    reason = str(branch.get("expression_objective") or branch.get("thesis") or "")
    return {
        "available": True,
        "family": family,
        "label": branch.get("structure_label"),
        "reason": reason,
        "trigger": branch.get("trigger"),
        "invalidation": branch.get("invalidation"),
        "pricing": (branch.get("greek_dollar_plan") or {}).get("defined_risk") or {},
    }


def build_card(
    signal: dict[str, Any],
    expression: dict[str, Any],
    approval: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    direction = _direction(signal)
    levels = _reference_map(signal, direction)
    option_expression = _expression(expression, direction)
    age_seconds = _timestamp_age_seconds(signal.get("ts"), now)
    blockers = list(approval.get("blocking_reasons") or [])
    if not approval.get("trade_allowed"):
        blockers.append("approval_trade_not_allowed")
    if age_seconds is None:
        blockers.append("signal_timestamp_missing")
    elif age_seconds > MAX_SIGNAL_AGE_SECONDS:
        blockers.append("signal_stale")
    if direction == "NONE":
        blockers.append("direction_unresolved")
    if levels["confirmation_level"] is None:
        blockers.append("bounce_confirmation_unavailable")
    authority_allows = bool(approval.get("trade_allowed")) and not blockers
    trigger_confirmed = bool(levels["confirmation_met"])
    if not authority_allows:
        state = "BLOCKED"
    elif not option_expression["available"]:
        state = "WATCH"
    elif trigger_confirmed:
        state = "TRIGGER_READY"
    else:
        state = "WATCH"
    setup_tag = signal.get("entry_setup_tag") or signal.get("setup_tag") or "No setup"
    if state == "BLOCKED":
        action = f"BLOCKED — watch-only {direction}; no entry authority"
    elif direction == "NONE":
        action = f"{state} — no directional expression"
    elif not option_expression["available"]:
        action = f"{state} — watch {direction}; option structure unavailable"
    else:
        action = f"{state} — {direction} via {option_expression['label']}"
    return {
        "schema": "sharpedge.operator_decision_card.v1",
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "symbol": signal.get("symbol") or "SPY",
        "state": state,
        "action": action,
        "direction": direction,
        "setup": {
            "tag": setup_tag,
            "reason": (signal.get("entry_gate") or {}).get("reason")
            or signal.get("entry_setup_bias")
            or "No setup rationale available.",
            "gamma_regime": signal.get("gamma_regime"),
            "volume_multiple": _number(signal.get("vol_mult")),
            "momentum_15m_pct": _number(signal.get("mom15")),
        },
        "levels": levels,
        "option_expression": option_expression,
        "authority": {
            "trade_allowed": bool(approval.get("trade_allowed")),
            "broker_order_allowed": bool(approval.get("broker_order_allowed")),
            "decision": approval.get("decision"),
            "blockers": list(dict.fromkeys(str(item) for item in blockers)),
            "operator_confirmation_required": True,
        },
        "freshness": {
            "signal_timestamp": signal.get("ts"),
            "signal_age_seconds": round(age_seconds, 1)
            if age_seconds is not None
            else None,
            "max_signal_age_seconds": MAX_SIGNAL_AGE_SECONDS,
        },
    }


def _money(value: Any) -> str:
    parsed = _number(value)
    return f"${parsed:.2f}" if parsed is not None else "Unavailable"


def render_html(card: dict[str, Any], refresh_seconds: int = 5) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    levels = card["levels"]
    expression = card["option_expression"]
    authority = card["authority"]
    blockers = ", ".join(authority["blockers"]) or "none"
    state_class = str(card["state"]).lower()
    references = "".join(
        f"<tr><th>{esc(name)}</th><td>{_money(value)}</td></tr>"
        for name, value in levels["references"].items()
    )
    zone = levels["bounce_zone"]
    zone_text = (
        f"{_money(zone['low'])}–{_money(zone['high'])} ({', '.join(zone['members'])})"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{max(int(refresh_seconds), 1)}">
  <title>SharpEdge Decision Card</title>
  <style>
    :root {{ color-scheme:dark; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#07101e; color:#e8eefc; font-family:system-ui,sans-serif; }}
    main {{ max-width:1050px; margin:auto; padding:16px; }}
    .hero {{ border:2px solid #334a70; border-radius:16px; padding:18px; background:#0c1728; }}
    .blocked {{ border-color:#f85149; }} .watch {{ border-color:#d29922; }} .trigger_ready {{ border-color:#3fb950; }}
    h1 {{ margin:0 0 6px; font-size:clamp(24px,5vw,42px); }}
    .action {{ font-size:clamp(18px,4vw,30px); font-weight:800; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:12px; }}
    .card {{ background:#0c1728; border:1px solid #263b5d; border-radius:12px; padding:14px; overflow-wrap:anywhere; }}
    h2 {{ margin:0 0 8px; color:#8fb8ff; font-size:14px; text-transform:uppercase; letter-spacing:.08em; }}
    .big {{ font-size:21px; font-weight:750; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:5px; text-align:left; border-bottom:1px solid #20304d; }}
    .blocked-text {{ color:#ff8e8e; font-weight:750; }}
    @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero {state_class}">
    <h1>SharpEdge Decision Card</h1>
    <div class="action">{esc(card["action"])}</div>
    <p>{esc(card["setup"]["tag"])} · gamma {esc(card["setup"]["gamma_regime"])} · volume {esc(card["setup"]["volume_multiple"])}x</p>
  </section>
  <div class="grid">
    <section class="card"><h2>Direction</h2><div class="big">{esc(card["direction"])}</div><p>{esc(card["setup"]["reason"])}</p></section>
    <section class="card"><h2>Option expression</h2><div class="big">{esc(expression["label"])}</div><p>{esc(expression["reason"])}</p></section>
    <section class="card"><h2>Authority</h2><div class="big">{esc(card["state"])}</div><p class="blocked-text">{esc(blockers)}</p></section>
    <section class="card"><h2>Bounce / rejection zone</h2><div class="big">{esc(zone_text)}</div><p>EMA stack: {esc(levels["ema_stack"])}; nearest: {esc((levels["nearest"] or {{}}).get("name", "unavailable"))}</p></section>
    <section class="card"><h2>Confirmation</h2><div class="big">{_money(levels["confirmation_level"])}</div><p>{esc(levels["trigger_rule"])}</p></section>
    <section class="card"><h2>Invalidation</h2><div class="big">{_money(levels["invalidation_level"])}</div><p>{esc(levels["invalidation_rule"])}</p></section>
  </div>
  <section class="card" style="margin-top:12px"><h2>Reference map</h2><table>{references}</table></section>
  <p>Decision support only. Trade authority remains fail-closed and operator confirmation is always required.</p>
</main>
</body>
</html>
"""


def write_card(card: dict[str, Any], json_path: Path, html_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(card), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    card = build_card(
        _read_json(args.signal),
        _read_json(args.expression),
        _read_json(args.approval),
    )
    write_card(card, args.json_out, args.html_out)
    print(json.dumps({"state": card["state"], "action": card["action"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
