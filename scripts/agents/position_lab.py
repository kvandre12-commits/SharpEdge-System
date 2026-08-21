#!/usr/bin/env python3
"""Build a trigger-first, two-sided position lab from live SharpEdge artifacts.

The point is not to declare the operator bullish or bearish. The point is to
frame the market in front of us with defensible, defined-risk structures on
both sides, then wait for geometry / acceptance to choose the branch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.agents.option_expression_logic import build_branch_greek_dollar_plan
    from scripts.agents.position_lab_calendar import build_calendar_context
    from scripts.agents.position_lab_view import render_text as _render_text
    from scripts.agents.position_lab_view import write_outputs
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from option_expression_logic import build_branch_greek_dollar_plan
    from position_lab_calendar import build_calendar_context
    from position_lab_view import render_text as _render_text
    from position_lab_view import write_outputs

OUTDIR = Path("outputs")
SIGNAL_JSON = OUTDIR / "signal.json"
STANDARD_SNAPSHOT_JSON = OUTDIR / "nerv_cockpit_standard" / "nerv_options_snapshot.json"
POSITION_LAB_SNAPSHOT_JSON = OUTDIR / "nerv_position_lab" / "nerv_options_snapshot.json"
CURATOR_JSON = OUTDIR / "nerv_curator.json"
APPROVAL_JSON = OUTDIR / "approval_decision.json"
MAX_DEFENSIBLE_WING_WIDTH = 3.0


@dataclass(frozen=True)
class OptionQuote:
    contract_symbol: str
    expiration: str
    option_type: str
    strike: float
    bid: float
    ask: float
    midpoint: float
    volume: int
    open_interest: int
    width_pct: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    greeks_source: str
    quote_timestamp: str
    fetch_timestamp: str
    fresh_quote_required: bool
    manual_validation_priority: str
    rejection_flags: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    dt_value = datetime.fromisoformat(candidate)
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC)


def age_minutes(value: str | None) -> float | None:
    dt_value = parse_timestamp(value)
    if dt_value is None:
        return None
    return round((datetime.now(UTC) - dt_value).total_seconds() / 60.0, 1)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def pick_symbol(signal: dict[str, Any], requested_symbol: str) -> str:
    signal_symbol = str(signal.get("symbol") or "").upper()
    if signal_symbol == requested_symbol.upper():
        return signal_symbol
    return requested_symbol.upper()


def quote_from_row(row: dict[str, Any]) -> OptionQuote:
    return OptionQuote(
        contract_symbol=str(row.get("contract_symbol") or ""),
        expiration=str(row.get("expiration") or ""),
        option_type=str(row.get("option_type") or "").lower(),
        strike=as_float(row.get("strike")),
        bid=as_float(row.get("bid")),
        ask=as_float(row.get("ask")),
        midpoint=as_float(row.get("midpoint")),
        volume=as_int(row.get("volume")),
        open_interest=as_int(row.get("open_interest")),
        width_pct=as_float(row.get("width_pct")),
        implied_volatility=as_float(row.get("implied_volatility")),
        delta=as_float(row.get("delta")),
        gamma=as_float(row.get("gamma")),
        theta=as_float(row.get("theta")),
        vega=as_float(row.get("vega")),
        greeks_source=str(row.get("greeks_source") or ""),
        quote_timestamp=str(row.get("quote_timestamp") or ""),
        fetch_timestamp=str(row.get("fetch_timestamp") or ""),
        fresh_quote_required=bool(row.get("fresh_quote_required", True)),
        manual_validation_priority=str(row.get("manual_validation_priority") or ""),
        rejection_flags=str(row.get("rejection_flags") or ""),
    )


def load_symbol_quotes(
    snapshot: dict[str, Any], symbol: str, expiration: str
) -> list[OptionQuote]:
    rows = snapshot.get("quotes") or []
    symbol = symbol.upper()
    filtered = [
        quote_from_row(row)
        for row in rows
        if str(row.get("underlying") or "").upper() == symbol
        and str(row.get("expiration") or "") == expiration
        and as_float(row.get("bid")) > 0
        and as_float(row.get("ask")) > 0
        and as_float(row.get("midpoint")) > 0
    ]
    preferred = [
        quote
        for quote in filtered
        if quote.manual_validation_priority == "high"
        and not quote.rejection_flags
        and quote.open_interest >= 1000
        and quote.volume >= 500
    ]
    return preferred or filtered


def calls_and_puts(
    quotes: list[OptionQuote],
) -> tuple[list[OptionQuote], list[OptionQuote]]:
    calls = sorted(
        (quote for quote in quotes if quote.option_type == "call"),
        key=lambda q: q.strike,
    )
    puts = sorted(
        (quote for quote in quotes if quote.option_type == "put"),
        key=lambda q: q.strike,
    )
    return calls, puts


def first_strike_at_or_above(
    quotes: list[OptionQuote], spot: float
) -> OptionQuote | None:
    for quote in quotes:
        if quote.strike >= spot:
            return quote
    return quotes[-1] if quotes else None


def first_strike_at_or_below(
    quotes: list[OptionQuote], spot: float
) -> OptionQuote | None:
    for quote in reversed(quotes):
        if quote.strike <= spot:
            return quote
    return quotes[0] if quotes else None


def next_higher(quotes: list[OptionQuote], strike: float) -> OptionQuote | None:
    for quote in quotes:
        if quote.strike > strike:
            return quote
    return None


def next_lower(quotes: list[OptionQuote], strike: float) -> OptionQuote | None:
    for quote in reversed(quotes):
        if quote.strike < strike:
            return quote
    return None


def liquidity_note(*quotes: OptionQuote) -> str:
    width = sum(quote.width_pct for quote in quotes) / max(len(quotes), 1)
    if width <= 0.03:
        return "tight enough to study"
    if width <= 0.08:
        return "usable but not luxurious"
    return "wide enough that chasing would be rude"


def serialize_leg(side: str, quote: OptionQuote) -> dict[str, Any]:
    return {
        "side": side,
        "contract_symbol": quote.contract_symbol,
        "expiration": quote.expiration,
        "option_type": quote.option_type,
        "strike": quote.strike,
        "midpoint": quote.midpoint,
        "bid": quote.bid,
        "ask": quote.ask,
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "width_pct": quote.width_pct,
        "implied_volatility": quote.implied_volatility,
        "delta": quote.delta,
        "gamma": quote.gamma,
        "theta": quote.theta,
        "vega": quote.vega,
        "greeks_source": quote.greeks_source,
    }


def build_spread(
    branch_id: str,
    direction: str,
    long_leg: OptionQuote,
    short_leg: OptionQuote,
    trigger_text: str,
    invalidation_text: str,
    thesis: str,
    caution: str,
) -> dict[str, Any]:
    width = abs(short_leg.strike - long_leg.strike)
    debit = round(long_leg.midpoint - short_leg.midpoint, 3)
    max_loss = max(debit, 0.0)
    max_gain = round(max(width - max_loss, 0.0), 3)
    if direction == "CALLS":
        breakeven = round(long_leg.strike + max_loss, 3)
        family = "call_debit_spread"
        label = f"{long_leg.expiration} {long_leg.strike:g}/{short_leg.strike:g} call debit spread"
    else:
        breakeven = round(long_leg.strike - max_loss, 3)
        family = "put_debit_spread"
        label = f"{long_leg.expiration} {long_leg.strike:g}/{short_leg.strike:g} put debit spread"
    reward_risk = round((max_gain / max_loss), 2) if max_loss > 0 else None
    return {
        "branch_id": branch_id,
        "direction": direction,
        "status": "watch_only_until_trigger",
        "structure_family": family,
        "structure_label": label,
        "trigger": trigger_text,
        "invalidation": invalidation_text,
        "thesis": thesis,
        "caution": caution,
        "quote_quality": liquidity_note(long_leg, short_leg),
        "pricing": {
            "debit": max_loss,
            "width": width,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "reward_risk": reward_risk,
        },
        "legs": [serialize_leg("buy", long_leg), serialize_leg("sell", short_leg)],
    }


def build_single(
    branch_id: str,
    direction: str,
    leg: OptionQuote,
    trigger_text: str,
    invalidation_text: str,
    thesis: str,
    caution: str,
) -> dict[str, Any]:
    breakeven = round(
        leg.strike + leg.midpoint
        if direction == "CALLS"
        else leg.strike - leg.midpoint,
        3,
    )
    family = "long_call" if direction == "CALLS" else "long_put"
    label = f"{leg.expiration} {leg.strike:g} {leg.option_type}"
    return {
        "branch_id": branch_id,
        "direction": direction,
        "status": "watch_only_until_trigger",
        "structure_family": family,
        "structure_label": label,
        "trigger": trigger_text,
        "invalidation": invalidation_text,
        "thesis": thesis,
        "caution": caution,
        "quote_quality": liquidity_note(leg),
        "pricing": {
            "debit": leg.midpoint,
            "width": None,
            "max_gain": None,
            "max_loss": leg.midpoint,
            "breakeven": breakeven,
            "reward_risk": None,
        },
        "legs": [serialize_leg("buy", leg)],
    }


def finalize_branch(
    branch: dict[str, Any],
    geometry: dict[str, Any],
    *,
    trigger_level: float | None = None,
    invalidation_level: float | None = None,
) -> dict[str, Any]:
    branch["levels"] = {
        "spot": geometry.get("spot"),
        "trigger_level": trigger_level,
        "invalidation_level": invalidation_level,
    }
    branch["greek_dollar_plan"] = build_branch_greek_dollar_plan(branch, geometry)
    return branch


def build_geometry_summary(signal: dict[str, Any]) -> dict[str, Any]:
    permission = signal.get("trade_permission") or {}
    setup = permission.get("setup_conviction") or {}
    dealer = signal.get("dealer_positioning") or {}
    event_radar = signal.get("event_radar") or {}
    return {
        "spot": as_float(signal.get("display_spot") or signal.get("spot")),
        "vwap": as_float(signal.get("vwap")),
        "vs_vwap": as_float(signal.get("vs_vwap")),
        "vol_mult": as_float(signal.get("vol_mult")),
        "setup_tag": str(
            setup.get("setup_tag") or signal.get("setup_tag") or "unknown"
        ),
        "setup_reason": str(setup.get("reason") or ""),
        "trade_gate": str(permission.get("trade_gate") or "unknown"),
        "trade_permission_score": as_int(permission.get("trade_permission_score")),
        "gamma_regime": str(
            dealer.get("gamma_regime") or signal.get("gamma_regime") or "unknown"
        ),
        "dealer_state": str(dealer.get("dealer_state") or "unknown"),
        "dealer_story": str(dealer.get("story") or ""),
        "pin": as_float(signal.get("pin") or dealer.get("gamma_wall_strike")),
        "call_wall": as_float(
            signal.get("call_wall") or dealer.get("actionable_call_wall_strike")
        ),
        "put_wall": as_float(
            signal.get("put_wall") or dealer.get("actionable_put_wall_strike")
        ),
        "balance_low": as_float(signal.get("balance_low")),
        "balance_high": as_float(signal.get("balance_high")),
        "premium_read": str(
            (signal.get("magnitude") or {}).get("premium_read")
            or signal.get("premium_read")
            or "unknown"
        ),
        "exp_move_implied_usd": as_float(
            (signal.get("magnitude") or {}).get("exp_move_implied_usd")
            or signal.get("exp_move_implied_usd")
        ),
        "event_headline": str(event_radar.get("headline") or ""),
        "event_story": str(event_radar.get("story") or ""),
    }


def prefer_single_leg(geometry: dict[str, Any]) -> bool:
    setup_tag = str(geometry.get("setup_tag") or "").upper()
    return bool(
        geometry.get("gamma_regime") == "negative"
        and geometry.get("premium_read") == "cheap"
        and as_float(geometry.get("vol_mult")) >= 1.2
        and ("RUNNER" in setup_tag or "HANDOFF" in setup_tag)
    )


def build_branches(
    signal: dict[str, Any],
    quotes: list[OptionQuote],
) -> list[dict[str, Any]]:
    geometry = build_geometry_summary(signal)
    spot = geometry["spot"]
    calls, puts = calls_and_puts(quotes)
    bull_long = first_strike_at_or_above(calls, spot)
    bear_long = first_strike_at_or_below(puts, spot)
    bull_short = next_higher(calls, bull_long.strike) if bull_long else None
    bear_short = next_lower(puts, bear_long.strike) if bear_long else None
    branches: list[dict[str, Any]] = []

    reclaim_level = max(
        geometry["vwap"], geometry["pin"], bull_long.strike if bull_long else 0.0
    )
    fail_level = min(
        value
        for value in [geometry["balance_low"], bear_long.strike if bear_long else 10**9]
        if value > 0
    )

    single_leg_preferred = prefer_single_leg(geometry)
    bull_spread_is_clean = bool(
        bull_long
        and bull_short
        and abs(bull_short.strike - bull_long.strike) <= MAX_DEFENSIBLE_WING_WIDTH
    )
    bear_spread_is_clean = bool(
        bear_long
        and bear_short
        and abs(bear_long.strike - bear_short.strike) <= MAX_DEFENSIBLE_WING_WIDTH
    )

    if bull_spread_is_clean and not single_leg_preferred:
        branches.append(
            finalize_branch(
                build_spread(
                    branch_id="bull_reclaim_branch",
                    direction="CALLS",
                    long_leg=bull_long,
                    short_leg=bull_short,
                    trigger_text=(
                        f"reclaim and hold above {reclaim_level:.2f}; better if price stops living under the magnet"
                    ),
                    invalidation_text=(
                        "fail back under "
                        f"{bull_long.strike:.2f} or lose balance support near "
                        f"{geometry['balance_low']:.2f}"
                    ),
                    thesis=(
                        "positive gamma says do not pre-guess upside. "
                        "If SPY gets back above VWAP/pin, use a defined-risk "
                        "reclaim spread instead of raw long premium."
                    ),
                    caution=(
                        "This is a premium-sensitive upside branch. "
                        f"{geometry['premium_read']} premium and "
                        f"{geometry['vol_mult']:.2f}x volume mean chasing "
                        "without acceptance is clown behavior."
                    ),
                ),
                geometry,
                trigger_level=reclaim_level,
                invalidation_level=geometry["balance_low"],
            )
        )
    elif bull_long:
        branches.append(
            finalize_branch(
                build_single(
                    branch_id="bull_reclaim_branch",
                    direction="CALLS",
                    leg=bull_long,
                    trigger_text=(
                        f"reclaim and hold above {reclaim_level:.2f}; better if price stops living under the magnet"
                    ),
                    invalidation_text=(
                        f"fail back under {bull_long.strike:.2f} or lose balance support near "
                        f"{geometry['balance_low']:.2f}"
                    ),
                    thesis=(
                        "Cheap premium, negative gamma, and a confirmed runner favor "
                        "uncapped single-leg convexity after reclaim acceptance."
                        if single_leg_preferred
                        else "No clean nearby short call wing is available, so the bullish "
                        "fallback is a single defined-debit call after reclaim acceptance."
                    ),
                    caution=(
                        "Single-leg premium carries uncapped theta exposure; do not use it "
                        "as a substitute for a missing trigger or stale quote confirmation."
                    ),
                ),
                geometry,
                trigger_level=reclaim_level,
                invalidation_level=geometry["balance_low"],
            )
        )

    if bear_spread_is_clean and not single_leg_preferred:
        branches.append(
            finalize_branch(
                build_spread(
                    branch_id="bear_fail_branch",
                    direction="PUTS",
                    long_leg=bear_long,
                    short_leg=bear_short,
                    trigger_text=(
                        f"accept below {fail_level:.2f} and keep pressure under {bear_long.strike:.2f}"
                    ),
                    invalidation_text=(
                        f"reclaim above {reclaim_level:.2f} or snap back into the magnet/pin"
                    ),
                    thesis=(
                        "The tape already has accepted-below / defensive hints. "
                        "If price keeps leaning below balance instead of "
                        "snapping back, the put side becomes the cleaner branch."
                    ),
                    caution=(
                        "Sticky positive gamma can still yank price upward. "
                        "Take the branch only on acceptance, not because puts "
                        "feel dramatic."
                    ),
                ),
                geometry,
                trigger_level=fail_level,
                invalidation_level=reclaim_level,
            )
        )
    elif bear_long:
        branches.append(
            finalize_branch(
                build_single(
                    branch_id="bear_fail_branch",
                    direction="PUTS",
                    leg=bear_long,
                    trigger_text=(
                        f"accept below {fail_level:.2f} and keep pressure under {bear_long.strike:.2f}"
                    ),
                    invalidation_text=(
                        f"reclaim above {reclaim_level:.2f} or snap back into the magnet/pin"
                    ),
                    thesis=(
                        "Cheap premium, negative gamma, and a confirmed runner favor "
                        "uncapped single-leg downside convexity after failure acceptance."
                        if single_leg_preferred
                        else "The bigger calendar has no clean nearby short leg, so the "
                        "defensible bearish fallback is a single defined-debit put."
                    ),
                    caution=(
                        "Respect theta and do not pay up blindly just because the "
                        "0DTE lane looked noisy."
                    ),
                ),
                geometry,
                trigger_level=fail_level,
                invalidation_level=reclaim_level,
            )
        )

    branches.append(
        finalize_branch(
            {
                "branch_id": "neutral_wait_branch",
                "direction": "NEUTRAL",
                "status": "preferred_right_now",
                "structure_family": "no_forced_position",
                "structure_label": "wait for branch confirmation",
                "trigger": "do nothing until one side accepts and the other side clearly fails",
                "invalidation": "N/A",
                "thesis": (
                    f"Current read is {geometry['setup_tag']} with "
                    f"{geometry['gamma_regime']} gamma, "
                    f"{geometry['dealer_state']} dealer state, and only "
                    f"{geometry['vol_mult']:.2f}x volume. "
                    "That is geometry first, opinion second."
                ),
                "caution": (
                    "Avoid forcing long straddles/strangles here: "
                    f"premium looks {geometry['premium_read']} and the "
                    "remaining implied move is only about "
                    f"${geometry['exp_move_implied_usd']:.2f}."
                ),
            },
            geometry,
        )
    )
    return branches


def build_payload(
    signal: dict[str, Any],
    snapshot: dict[str, Any],
    curator: dict[str, Any],
    approval: dict[str, Any],
    *,
    requested_symbol: str = "SPY",
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    symbol = pick_symbol(signal, requested_symbol)
    snapshot_quotes = snapshot.get("quotes") or []
    calendar_context = build_calendar_context(signal, snapshot_quotes, symbol)
    expiration = calendar_context.get("selected_expiration")
    quotes = load_symbol_quotes(snapshot, symbol, expiration) if expiration else []
    geometry = build_geometry_summary(signal)
    branches = build_branches(signal, quotes) if quotes else []
    price_authority = signal.get("price_authority") or {}
    branch_posture = "branch_defined_debit_spread"
    branch_reason = "Use defined-risk branches on both sides, then let acceptance choose the direction."
    if geometry["gamma_regime"] == "negative":
        branch_posture = "momentum_defined_risk_vertical"
        branch_reason = "Negative gamma can run; keep risk defined but allow directional follow-through."
    return {
        "schema": "sharpedge.position_lab.v1",
        "generated_at_utc": utc_now(),
        "symbol": symbol,
        "primary_posture": branch_posture,
        "posture_reason": branch_reason,
        "source_artifacts": {
            "signal": str(SIGNAL_JSON),
            "snapshot": str(snapshot_path or STANDARD_SNAPSHOT_JSON),
            "curator": str(CURATOR_JSON),
            "approval": str(APPROVAL_JSON),
        },
        "calendar_context": calendar_context,
        "freshness": {
            "signal_minutes_old": age_minutes(price_authority.get("display_time_utc")),
            "quote_minutes_old_max": max(
                (age_minutes(q.quote_timestamp) or 0.0) for q in quotes
            )
            if quotes
            else None,
            "quote_minutes_old_min": min(
                (age_minutes(q.quote_timestamp) or 0.0) for q in quotes
            )
            if quotes
            else None,
            "expiration": expiration,
            "fresh_quote_required": any(q.fresh_quote_required for q in quotes),
        },
        "geometry": geometry,
        "curator_context": {
            "headline": curator.get("headline"),
            "stance": curator.get("stance"),
            "watch_next": list(curator.get("watch_next") or [])[:3],
            "warnings": list(curator.get("warnings") or [])[:3],
        },
        "branches": branches,
        "execution_boundary": {
            "trade_allowed": bool(approval.get("trade_allowed")),
            "broker_order_allowed": bool(approval.get("broker_order_allowed")),
            "decision": approval.get("decision"),
            "blocking_reasons": list(approval.get("blocking_reasons") or [])[:5],
            "note": "This artifact proposes structures for study. It does not authorize or place an order.",
        },
    }


def render_text(payload: dict[str, Any]) -> str:
    return _render_text(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SharpEdge position lab.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--output-base", default="")
    return parser.parse_args(argv)


def resolve_snapshot_path() -> Path:
    return (
        POSITION_LAB_SNAPSHOT_JSON
        if POSITION_LAB_SNAPSHOT_JSON.exists()
        else STANDARD_SNAPSHOT_JSON
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal = read_json(SIGNAL_JSON)
    snapshot_path = resolve_snapshot_path()
    snapshot = read_json(snapshot_path)
    curator = read_json(CURATOR_JSON)
    approval = read_json(APPROVAL_JSON)
    payload = build_payload(
        signal,
        snapshot,
        curator,
        approval,
        requested_symbol=str(args.symbol).upper(),
        snapshot_path=snapshot_path,
    )
    base = (
        Path(args.output_base)
        if args.output_base
        else OUTDIR / f"{str(args.symbol).lower()}_position_lab"
    )
    json_path, txt_path = write_outputs(payload, base)
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
