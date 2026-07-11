#!/usr/bin/env python3
"""Validate asymmetric execution logic that can loudly favor CALLS or PUTS.

This is a proof script only. It does not mutate cockpit scoring, approval, token,
or broker behavior. It builds synthetic market states and runs the real
score_trade_permission engine to prove directional asymmetry both ways.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from trade_permission import score_trade_permission  # noqa: E402

OUT_JSON = ROOT / "outputs/asymmetric_execution_validation.json"
OUT_MD = ROOT / "outputs/asymmetric_execution_validation.md"
SCHEMA = "sharpedge.asymmetric_execution_validation.v1"
SCREAM_SCORE = 72
SCREAM_BIAS_STRENGTH = 0.20


def _trend_bars(
    direction: str, count: int = 64
) -> list[tuple[int, float, float, float, float, int]]:
    sign = 1 if direction == "up" else -1
    price = 100.0
    bars = []
    for idx in range(count):
        open_ = price
        close = open_ + sign * 0.09
        high = max(open_, close) + 0.04
        low = min(open_, close) - 0.02
        if direction == "up":
            close = high - 0.01
        else:
            close = low + 0.01
        volume = 1200 + idx * 18
        bars.append(
            (
                80 + idx,
                round(open_, 4),
                round(high, 4),
                round(low, 4),
                round(close, 4),
                volume,
            )
        )
        price = close
    return bars


def _mixed_bars() -> list[tuple[int, float, float, float, float, int]]:
    bars = []
    price = 100.0
    for idx in range(64):
        drift = 0.06 if idx < 50 else (-0.08 if idx % 2 else 0.03)
        open_ = price
        close = open_ + drift
        high = max(open_, close) + 0.10
        low = min(open_, close) - 0.10
        bars.append(
            (
                80 + idx,
                round(open_, 4),
                round(high, 4),
                round(low, 4),
                round(close, 4),
                900 + idx * 4,
            )
        )
        price = close
    return bars


def _pa(
    bars: list[tuple[int, float, float, float, float, int]],
    side: str,
    **overrides: Any,
) -> dict[str, Any]:
    closes = [bar[4] for bar in bars]
    spot = closes[-1]
    bullish = side == "CALLS"
    pa = {
        "spot": spot,
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "balance_high": spot - 0.12 if bullish else spot + 0.62,
        "balance_low": spot - 0.62 if bullish else spot + 0.12,
        "position_in_balance": 1.0 if bullish else 0.0,
        "balance_state": "above" if bullish else "below",
        "balance_label": "TOP" if bullish else "BOTTOM",
        "balance_width_pct": 0.45,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "recent_balance",
        "dominant_balance_reason": "validation fixture",
        "dominant_balance_previous_name": "recent_balance",
        "dominant_balance_flip": {"flipped": False},
        "balance_models": {},
        "balance_confluence": {
            "state": "agreement",
            "score": 78,
            "bias": side,
            "reason": f"validation balance alignment favors {side}",
        },
        "balance_disagreement": {"has_disagreement": False},
        "session_position_in_range": 0.90 if bullish else 0.10,
        "rng_pos": 90.0 if bullish else 10.0,
        "day_chg": (spot / closes[0] - 1) * 100,
        "vwap": spot - 0.42 if bullish else spot + 0.42,
        "vs_vwap": 0.40 if bullish else -0.40,
        "mom15": 0.42 if bullish else -0.42,
        "vol_mult": 1.90,
    }
    pa.update(overrides)
    return pa


def _levels(spot: float, side: str) -> dict[str, float]:
    if side == "CALLS":
        return {
            "ORH": spot - 0.70,
            "ORL": spot - 4.00,
            "PDH": spot - 1.20,
            "PDL": spot - 5.00,
            "PDC": spot - 3.00,
        }
    return {
        "ORH": spot + 4.00,
        "ORL": spot + 0.70,
        "PDH": spot + 5.00,
        "PDL": spot + 1.20,
        "PDC": spot + 3.00,
    }


def _op(spot: float) -> dict[str, Any]:
    return {
        "atm_iv": 0.18,
        "call_wall": round(spot + 8, 2),
        "put_wall": round(spot - 8, 2),
    }


def _case(
    name: str,
    side: str,
    setup: dict[str, Any],
    volatility_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direction = "up" if side == "CALLS" else "down"
    bars = _trend_bars(direction)
    spot = bars[-1][4]
    card = score_trade_permission(
        bars,
        _pa(bars, side),
        _levels(spot, side),
        [setup],
        _op(spot),
        {
            "regime": "negative",
            "pin": round(spot - 5 if side == "CALLS" else spot + 5, 2),
        },
        {"premium_read": "cheap"},
        volatility_structure or {},
    )
    return _summarize(name, card, expected_bias=side)


def _conflict_case() -> dict[str, Any]:
    bars = _mixed_bars()
    spot = bars[-1][4]
    card = score_trade_permission(
        bars,
        _pa(
            bars,
            "CALLS",
            rng_pos=84.0,
            mom15=0.04,
            vol_mult=0.75,
            balance_confluence={
                "state": "agreement",
                "score": 52,
                "bias": "NEUTRAL",
                "reason": "weak agreement",
            },
        ),
        {"ORH": spot - 0.40, "ORL": spot - 4.00, "PDC": spot - 2.00},
        [{"tag": "STICKY DAY (calm/chop)", "bias": "FADE the edges", "kind": "warn"}],
        _op(spot),
        {"regime": "positive", "pin": spot},
        {"premium_read": "rich"},
        {},
    )
    return _summarize("conflict_should_not_scream", card, expected_bias="NEUTRAL")


def _top_directional_rows(card: dict[str, Any], side: str) -> list[dict[str, Any]]:
    rows = []
    for name, item in (card.get("scores") or {}).items():
        if item.get("bias") == side:
            rows.append(
                {"name": name, "score": item.get("score"), "reason": item.get("reason")}
            )
    return sorted(rows, key=lambda row: int(row.get("score") or 0), reverse=True)[:6]


def _summarize(name: str, card: dict[str, Any], expected_bias: str) -> dict[str, Any]:
    bias = card.get("bias")
    score = int(card.get("trade_permission_score") or 0)
    strength = float(card.get("bias_strength") or 0)
    setup = card.get("setup_conviction") or {}
    screams = (
        bias in {"CALLS", "PUTS"}
        and score >= SCREAM_SCORE
        and strength >= SCREAM_BIAS_STRENGTH
    )
    expected_scream = expected_bias in {"CALLS", "PUTS"}
    passed = bias == expected_bias if expected_bias != "NEUTRAL" else not screams
    if expected_scream:
        passed = passed and screams and setup.get("bias") == expected_bias
    return {
        "name": name,
        "passed": passed,
        "expected_bias": expected_bias,
        "actual_bias": bias,
        "screams": screams,
        "trade_gate": card.get("trade_gate"),
        "trade_permission_score": score,
        "bias_strength": strength,
        "setup_gate": setup.get("setup_gate"),
        "setup_bias": setup.get("bias"),
        "setup_score": setup.get("setup_conviction_score"),
        "top_directional_rows": _top_directional_rows(card, bias),
        "warnings": card.get("warning_reasons"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Asymmetric Execution Validation",
        "",
        "Validation-only proof for CALLS/PUTS directional asymmetry. No broker action, no approval mutation.",
        "",
        f"Overall status: **{payload['status']}**",
        "",
        "| Case | Expected | Actual | Screams | Gate/Score | Bias Strength | Setup |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {name} | {expected_bias} | {actual_bias} | {screams} | {trade_gate}/{trade_permission_score} | {bias_strength:.3f} | {setup_gate}/{setup_bias}/{setup_score} |".format(
                **case
            )
        )
    lines.extend(["", "## Directional evidence"])
    for case in payload["cases"]:
        lines.append(f"\n### {case['name']}")
        for row in case["top_directional_rows"]:
            lines.append(f"- `{row['name']}` {row['score']}: {row['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    call_setup = {
        "tag": "FAILED BREAKDOWN",
        "bias": "CALLS (bullish)",
        "kind": "ok",
        "level_name": "ORL",
        "level_price": 100.0,
        "trigger_price": 100.4,
        "bars_ago": 0,
        "detail": "sellers trapped below ORL; reclaim favors calls",
    }
    put_setup = {
        "tag": "FAILED BREAKOUT",
        "bias": "PUTS (bearish)",
        "kind": "bad",
        "level_name": "ORH",
        "level_price": 100.0,
        "trigger_price": 99.6,
        "bars_ago": 0,
        "detail": "buyers trapped above ORH; rejection favors puts",
    }
    cases = [
        _case(
            "calls_reclaim_scream",
            "CALLS",
            call_setup,
            {
                "coil": True,
                "structure_state": "channel_breakout_setup",
                "bias": "neutral_to_bullish",
                "trigger_high": 105.78,
            },
        ),
        _case(
            "puts_rejection_scream",
            "PUTS",
            put_setup,
            {
                "coil": True,
                "structure_state": "channel_breakout_setup",
                "bias": "neutral_to_bearish",
                "trigger_low": 94.22,
            },
        ),
        _conflict_case(),
    ]
    payload = {
        "schema": SCHEMA,
        "status": "passed" if all(case["passed"] for case in cases) else "failed",
        "scream_definition": {
            "bias": "CALLS or PUTS",
            "min_trade_permission_score": SCREAM_SCORE,
            "min_bias_strength": SCREAM_BIAS_STRENGTH,
            "setup_bias_must_match": True,
        },
        "cases": cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
