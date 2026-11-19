"""Deterministic noon-to-12:30 ET long-call ladder research leads.

A Helios approval in this module means research-review approval only. It never
creates broker authority, submits an order, or modifies approval_decision.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from scripts.alpha_swarm.contracts import payload_sha256

SCHEMA = "sharpedge.helios_call_ladder.v1"
NY = ZoneInfo("America/New_York")
SYMBOLS = frozenset({"SPY", "NVDA"})
ALLOWED_DTE = frozenset({1, 2})
WALLET_CAP_DOLLARS = 250.0
MAX_ASK = 1.0
MAX_WIDTH_PCT = 0.25
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 50
MAX_TOTAL_CONTRACTS = 5


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _dte(expiration: Any, session_date: date) -> int | None:
    try:
        return (date.fromisoformat(str(expiration)) - session_date).days
    except ValueError:
        return None


def _quote_width_pct(quote: dict[str, Any]) -> float | None:
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2.0
    return (ask - bid) / midpoint if midpoint > 0 else None


def evaluate_trigger(features: dict[str, Any]) -> dict[str, Any]:
    values = {
        "vs_vwap_pct": _float(features.get("vs_vwap_pct")),
        "momentum_15m_pct": _float(features.get("momentum_15m_pct")),
        "volume_ratio": _float(features.get("volume_ratio")),
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        return {"triggered": False, "state": "abstain", "reasons": [f"missing_{key}" for key in missing]}
    reasons: list[str] = []
    if values["vs_vwap_pct"] < 0.05:
        reasons.append("vs_vwap_below_0.05pct")
    if values["momentum_15m_pct"] < 0.05:
        reasons.append("momentum_15m_below_0.05pct")
    if values["volume_ratio"] < 1.2:
        reasons.append("volume_ratio_below_1.2")
    return {
        "triggered": not reasons,
        "state": "approve" if not reasons else "reject",
        "reasons": reasons or ["bullish_vwap_momentum_volume_trigger"],
        "features": values,
    }


def _eligible_quote(
    quote: dict[str, Any], symbol: str, session_date: date
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    spot = _float(quote.get("underlying_price"))
    strike = _float(quote.get("strike"))
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    dte = _dte(quote.get("expiration"), session_date)
    width_pct = _quote_width_pct(quote)
    if str(quote.get("underlying") or "").upper() != symbol:
        reasons.append("wrong_symbol")
    if str(quote.get("option_type") or "").lower() != "call":
        reasons.append("not_call")
    if spot is None or strike is None or strike <= spot:
        reasons.append("not_otm")
    if dte not in ALLOWED_DTE:
        reasons.append("dte_not_1_or_2")
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("invalid_quote")
    elif ask > MAX_ASK:
        reasons.append("ask_above_1.00")
    if width_pct is None or width_pct > MAX_WIDTH_PCT:
        reasons.append("spread_too_wide")
    if _int(quote.get("volume")) < MIN_VOLUME:
        reasons.append("volume_below_10")
    if _int(quote.get("open_interest")) < MIN_OPEN_INTEREST:
        reasons.append("open_interest_below_50")
    return not reasons, reasons


def _score_quote(quote: dict[str, Any], nearest_target: float | None) -> float:
    width = _quote_width_pct(quote) or MAX_WIDTH_PCT
    volume = _int(quote.get("volume"))
    open_interest = _int(quote.get("open_interest"))
    strike = _float(quote.get("strike")) or 0.0
    target_bonus = 10.0 if nearest_target and strike <= nearest_target else 0.0
    dte_bonus = 5.0 if _dte(quote.get("expiration"), date.fromisoformat(str(quote["session_date"]))) == 2 else 0.0
    return round(
        35.0 * max(0.0, 1.0 - width / MAX_WIDTH_PCT)
        + 25.0 * min(volume / 500.0, 1.0)
        + 25.0 * min(open_interest / 1000.0, 1.0)
        + target_bonus
        + dte_bonus,
        3,
    )


def build_ladder(
    quotes: list[dict[str, Any]],
    *,
    symbol: str,
    session_date: date,
    nearest_target: float | None,
    wallet_cap: float = WALLET_CAP_DOLLARS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for quote in quotes:
        passed, reasons = _eligible_quote(quote, symbol, session_date)
        if not passed:
            rejected.append({"contract_symbol": quote.get("contract_symbol"), "reasons": reasons})
            continue
        normalized = dict(quote)
        normalized["session_date"] = session_date.isoformat()
        normalized["score"] = _score_quote(normalized, nearest_target)
        eligible.append(normalized)
    eligible.sort(key=lambda item: (-item["score"], int(_dte(item["expiration"], session_date) or 99), float(item["strike"])))

    ladder: list[dict[str, Any]] = []
    spent = 0.0
    for quote in eligible:
        contract_cost = round(float(quote["ask"]) * 100.0, 2)
        if spent + contract_cost > wallet_cap or len(ladder) >= MAX_TOTAL_CONTRACTS:
            continue
        ladder.append(
            {
                "contract_symbol": quote.get("contract_symbol"),
                "expiration": quote.get("expiration"),
                "dte": _dte(quote.get("expiration"), session_date),
                "strike": _float(quote.get("strike")),
                "quantity": 1,
                "limit_reference": _float(quote.get("ask")),
                "max_debit_dollars": contract_cost,
                "bid": _float(quote.get("bid")),
                "ask": _float(quote.get("ask")),
                "width_pct": round(_quote_width_pct(quote) or 0.0, 4),
                "volume": _int(quote.get("volume")),
                "open_interest": _int(quote.get("open_interest")),
                "implied_volatility": _float(quote.get("implied_volatility")),
                "delta": _float(quote.get("delta")),
                "score": quote["score"],
            }
        )
        spent += contract_cost
    return ladder, rejected


def _review(role: str, decision: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "role": role,
        "decision": decision,
        "reasons": reasons,
        "authoritative": False,
        "execution_permitted": False,
    }


def build_lead(
    *,
    symbol: str,
    now: datetime,
    features: dict[str, Any],
    quotes: list[dict[str, Any]],
    nearest_target: float | None,
    source_hashes: dict[str, str],
    cycle_state: str = "IDLE",
    data_issues: list[str] | None = None,
) -> dict[str, Any]:
    symbol = symbol.upper()
    local = now.astimezone(NY)
    blockers: list[str] = []
    if symbol not in SYMBOLS:
        blockers.append("unsupported_symbol")
    if local.weekday() >= 5:
        blockers.append("weekend")
    if (local.hour, local.minute) < (12, 0):
        blockers.append("before_noon_et")
    if (local.hour, local.minute) > (12, 30):
        blockers.append("after_1230_et")
    if cycle_state != "IDLE":
        blockers.append(f"cycle_not_idle:{cycle_state}")

    trigger = evaluate_trigger(features)
    ladder, rejected = build_ladder(
        quotes,
        symbol=symbol,
        session_date=local.date(),
        nearest_target=nearest_target,
    )
    data_reasons = list(data_issues or [])
    if not source_hashes or not quotes:
        data_reasons.append("missing_sources_or_quotes")
    data_decision = "approve" if not data_reasons else "abstain"
    trigger_decision = trigger["state"]
    liquidity_decision = "approve" if ladder else "abstain"
    skeptic_decision = "reject" if blockers else "approve"
    reviews = [
        _review("data_steward", data_decision, data_reasons),
        _review("trigger_researcher", trigger_decision, list(trigger["reasons"])),
        _review("nerv_liquidity", liquidity_decision, [] if ladder else ["no_eligible_1dte_2dte_otm_calls"]),
        _review("skeptic", skeptic_decision, blockers),
    ]
    decisions = {review["decision"] for review in reviews}
    aggregate = "approve" if decisions == {"approve"} else "reject" if "reject" in decisions else "abstain"
    payload = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "session_date": local.date().isoformat(),
        "entry_window": {
            "opens_at": local.replace(hour=12, minute=0, second=0, microsecond=0).isoformat(),
            "closes_at": local.replace(hour=12, minute=30, second=0, microsecond=0).isoformat(),
        },
        "symbol": symbol,
        "strategy": "long_otm_call_wallet_ladder",
        "trigger": trigger,
        "nearest_refill_target": nearest_target,
        "wallet": {
            "cap_dollars": WALLET_CAP_DOLLARS,
            "proposed_debit_dollars": round(sum(item["max_debit_dollars"] for item in ladder), 2),
            "remaining_dollars": round(WALLET_CAP_DOLLARS - sum(item["max_debit_dollars"] for item in ladder), 2),
        },
        "ladder": ladder,
        "rejected_quote_count": len(rejected),
        "reviews": reviews,
        "aggregate_decision": aggregate,
        "entry_handoff": {
            "state": "ENTRY_PROPOSED" if aggregate == "approve" else "STAND_DOWN",
            "next_state_after_verified_fill": "OPEN_POSITION",
            "same_trigger_reentry_allowed": False,
            "operator_confirmation_required": True,
        },
        "seller_handoff_policy": {
            "enabled_only_after_verified_fill": True,
            "profit_management_starts_pct": 20,
            "tighten_review_pct": 35,
            "strong_take_profit_review_pct": 55,
        },
        "source_hashes": source_hashes,
        "authority": {
            "purpose": "research_entry_lead",
            "authoritative": False,
            "execution_permitted": False,
            "broker_action_allowed": False,
            "may_modify_approval_decision": False,
        },
    }
    payload["lead_id"] = payload_sha256(payload)
    return payload


__all__ = ["build_ladder", "build_lead", "evaluate_trigger"]
