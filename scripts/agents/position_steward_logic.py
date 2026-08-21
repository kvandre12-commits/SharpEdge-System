from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

NON_AUTHORITATIVE_NOTICE = (
    "Research only. This artifact cannot authorize, draft, or execute an order. "
    "Every position change requires fresh evidence and operator approval."
)


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _recovery_price(anchor_low: float, anchor_high: float, level: float) -> float:
    low = Decimal(str(anchor_low))
    high = Decimal(str(anchor_high))
    value = low + (high - low) * Decimal(str(level)) / Decimal("100")
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_recovery_snapshot(
    spot: float, anchor_low: float, anchor_high: float
) -> dict[str, Any]:
    span = anchor_high - anchor_low
    if spot <= 0 or span <= 0:
        return {
            "valid": False,
            "recovery_pct": None,
            "reason": "positive spot and ordered recovery anchors are required",
            "ladder": {},
        }

    recovery_pct = ((spot - anchor_low) / span) * 100.0
    ladder = {
        f"{level:g}%": _recovery_price(anchor_low, anchor_high, level)
        for level in (25.0, 50.0, 61.8, 75.0, 100.0)
    }
    if recovery_pct < 0:
        phase = "below_event_low"
    elif recovery_pct < 25:
        phase = "base_building"
    elif recovery_pct < 50:
        phase = "recovery_confirmed"
    elif recovery_pct < 75:
        phase = "meaningful_recovery"
    elif recovery_pct < 100:
        phase = "advanced_recovery"
    else:
        phase = "full_recovery_or_better"
    return {
        "valid": True,
        "anchor_low": round(anchor_low, 4),
        "anchor_high": round(anchor_high, 4),
        "displacement": round(span, 4),
        "recovered_dollars": round(spot - anchor_low, 4),
        "recovery_pct": round(recovery_pct, 1),
        "phase": phase,
        "ladder": ladder,
    }


def build_session_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no regular-session bars"}
    session_date = str(rows[-1].get("date") or "")
    session_rows = [row for row in rows if str(row.get("date") or "") == session_date]
    if not session_rows:
        return {"available": False, "reason": "no bars for latest session"}

    volumes = [int(row.get("volume") or 0) for row in session_rows]
    total_volume = sum(volumes)
    vwap_numerator = 0.0
    for row, volume in zip(session_rows, volumes):
        typical = (
            float(row.get("high") or 0)
            + float(row.get("low") or 0)
            + float(row.get("close") or 0)
        ) / 3.0
        vwap_numerator += typical * volume
    vwap = vwap_numerator / total_volume if total_volume else 0.0
    closes = [float(row.get("close") or 0) for row in session_rows]
    spot = closes[-1]
    momentum_15m = (
        (spot / closes[-4] - 1) * 100 if len(closes) >= 4 and closes[-4] else None
    )
    return {
        "available": True,
        "session_date": session_date,
        "spot": round(spot, 4),
        "open": round(float(session_rows[0].get("open") or spot), 4),
        "high": round(max(float(row.get("high") or spot) for row in session_rows), 4),
        "low": round(min(float(row.get("low") or spot) for row in session_rows), 4),
        "vwap": round(vwap, 4) if vwap else None,
        "above_vwap": bool(vwap and spot > vwap),
        "momentum_15m_pct": round(momentum_15m, 3)
        if momentum_15m is not None
        else None,
        "volume": total_volume,
        "bar_count": len(session_rows),
    }


def _days_to_expiry(expiration: str, as_of: date) -> int | None:
    try:
        return (date.fromisoformat(expiration) - as_of).days
    except (TypeError, ValueError):
        return None


def build_position_snapshot(
    position: dict[str, Any],
    quotes: dict[str, dict[str, Any]],
    *,
    as_of: date,
) -> dict[str, Any]:
    legs = position.get("legs") or []
    if not legs:
        return {
            "available": False,
            "reason": "no position legs supplied",
            "missing_contracts": [],
        }

    net_mid = 0.0
    liquidation_value = 0.0
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    normalized_legs = []
    missing_contracts = []
    short_dtes = []
    for leg in legs:
        contract = str(leg.get("contract") or "").upper()
        quote = quotes.get(contract)
        if not quote:
            missing_contracts.append(contract)
            continue
        side = str(leg.get("side") or "").lower()
        quantity = max(int(leg.get("quantity") or 0), 0)
        if side not in {"buy", "sell"} or quantity <= 0:
            missing_contracts.append(contract)
            continue
        sign = 1.0 if side == "buy" else -1.0
        bid = float(_as_float(quote.get("bid"), 0.0) or 0.0)
        ask = float(_as_float(quote.get("ask"), 0.0) or 0.0)
        midpoint = (bid + ask) / 2.0
        multiplier = 100.0 * quantity
        net_mid += sign * midpoint * multiplier
        liquidation_value += (bid if side == "buy" else -ask) * multiplier
        net_delta += (
            sign * float(_as_float(quote.get("delta"), 0.0) or 0.0) * multiplier
        )
        net_gamma += (
            sign * float(_as_float(quote.get("gamma"), 0.0) or 0.0) * multiplier
        )
        net_theta += (
            sign * float(_as_float(quote.get("theta"), 0.0) or 0.0) * multiplier
        )
        net_vega += sign * float(_as_float(quote.get("vega"), 0.0) or 0.0) * multiplier
        expiration = str(leg.get("expiration") or "")
        dte = _days_to_expiry(expiration, as_of)
        if side == "sell" and dte is not None:
            short_dtes.append(dte)
        normalized_legs.append(
            {
                "contract": contract,
                "side": side,
                "quantity": quantity,
                "expiration": expiration,
                "strike": _as_float(leg.get("strike"), None),
                "bid": bid,
                "ask": ask,
                "midpoint": round(midpoint, 4),
                "delta": _as_float(quote.get("delta"), None),
                "theta": _as_float(quote.get("theta"), None),
                "iv": _as_float(quote.get("iv"), None),
                "open_interest": int(_as_float(quote.get("open_interest"), 0.0) or 0),
                "volume": int(_as_float(quote.get("volume"), 0.0) or 0),
                "dte": dte,
            }
        )

    entry_debit = _as_float(position.get("entry_debit_dollars"), None)
    return {
        "available": not missing_contracts and bool(normalized_legs),
        "missing_contracts": missing_contracts,
        "legs": normalized_legs,
        "entry_debit_dollars": entry_debit,
        "midpoint_value_dollars": round(net_mid, 2),
        "conservative_liquidation_value_dollars": round(liquidation_value, 2),
        "estimated_pnl_dollars": (
            round(liquidation_value - entry_debit, 2)
            if entry_debit is not None
            else None
        ),
        "net_delta_shares": round(net_delta, 1),
        "net_gamma_shares_per_1_dollar": round(net_gamma, 1),
        "theta_dollars_per_day": round(net_theta, 2),
        "vega_dollars_per_1iv": round(net_vega, 2),
        "nearest_short_dte": min(short_dtes) if short_dtes else None,
    }


def classify_action(
    spec: dict[str, Any],
    *,
    spot: float,
    recovery: dict[str, Any],
    session: dict[str, Any],
    position: dict[str, Any],
    freshness: dict[str, Any],
) -> dict[str, Any]:
    position_spec = spec.get("position") or {}
    risk = spec.get("risk") or {}
    management = spec.get("management") or {}
    thesis = spec.get("thesis") or {}
    status = str(position_spec.get("status") or "research").lower()
    risk_budget = _as_float(risk.get("max_total_debit_dollars"), None)

    if status != "active" or risk_budget is None:
        return {
            "state": "insufficient_position_context",
            "reason": "position must be operator-confirmed active and include a dollar risk budget",
            "operator_approval_required": True,
        }
    if not freshness.get("market_open"):
        return {
            "state": "market_closed_review",
            "reason": "off-hours snapshots may inform preparation but cannot justify a position change",
            "operator_approval_required": True,
        }
    if not freshness.get("fresh_for_management") or not position.get("available"):
        return {
            "state": "refresh_required",
            "reason": "fresh price, option quotes, and complete leg marks are required",
            "operator_approval_required": True,
        }

    invalidation = float(_as_float(thesis.get("invalidation_price"), 0.0) or 0.0)
    if invalidation and spot <= invalidation:
        return {
            "state": "exit_thesis_broken",
            "reason": f"spot {spot:.2f} is at or below thesis invalidation {invalidation:.2f}",
            "operator_approval_required": True,
        }

    roll_short_dte = int(management.get("roll_short_dte") or 5)
    nearest_short_dte = position.get("nearest_short_dte")
    if nearest_short_dte is not None and nearest_short_dte <= roll_short_dte:
        return {
            "state": "roll_short_leg",
            "reason": f"nearest short leg has {nearest_short_dte} DTE; assignment/expiry review is due",
            "operator_approval_required": True,
        }

    recovery_pct = float(recovery.get("recovery_pct") or 0.0)
    trim_recovery_pct = float(management.get("trim_recovery_pct") or 75.0)
    if recovery_pct >= trim_recovery_pct:
        return {
            "state": "reduce_or_protect",
            "reason": f"recovery reached {recovery_pct:.1f}%, above the {trim_recovery_pct:.1f}% harvest threshold",
            "operator_approval_required": True,
        }

    min_recovery = float(management.get("grow_min_recovery_pct") or 25.0)
    max_recovery = float(management.get("grow_max_recovery_pct") or 61.8)
    pnl = position.get("estimated_pnl_dollars")
    target_delta = float(_as_float(risk.get("target_delta_shares"), 0.0) or 0.0)
    entry_debit = float(position.get("entry_debit_dollars") or 0.0)
    risk_headroom = risk_budget - entry_debit
    growth_confirmed = (
        min_recovery <= recovery_pct <= max_recovery
        and bool(session.get("above_vwap"))
        and float(session.get("momentum_15m_pct") or 0.0) > 0
        and pnl is not None
        and pnl >= 0
        and risk_headroom > 0
        and float(position.get("net_delta_shares") or 0.0) < target_delta
    )
    if growth_confirmed:
        return {
            "state": "consider_add",
            "reason": (
                "recovery, VWAP, momentum, winner-only, delta-headroom, and dollar-risk "
                "gates all pass; a fresh structure/payoff review is still required"
            ),
            "operator_approval_required": True,
        }
    return {
        "state": "hold",
        "reason": "thesis remains intact, but every growth gate is not simultaneously satisfied",
        "operator_approval_required": True,
    }


def build_payload(
    spec: dict[str, Any],
    *,
    generated_at: str,
    spot: float,
    session: dict[str, Any],
    quotes: dict[str, dict[str, Any]],
    freshness: dict[str, Any],
    as_of: date,
    sources: dict[str, Any],
) -> dict[str, Any]:
    thesis = spec.get("thesis") or {}
    recovery = build_recovery_snapshot(
        spot,
        float(_as_float(thesis.get("anchor_low"), 0.0) or 0.0),
        float(_as_float(thesis.get("anchor_high"), 0.0) or 0.0),
    )
    position = build_position_snapshot(spec.get("position") or {}, quotes, as_of=as_of)
    action = classify_action(
        spec,
        spot=spot,
        recovery=recovery,
        session=session,
        position=position,
        freshness=freshness,
    )
    return {
        "schema": "sharpedge.position_steward.v1",
        "generated_at": generated_at,
        "symbol": str(spec.get("symbol") or "").upper(),
        "authority": {
            "authoritative": False,
            "execution_permitted": False,
            "approval_policy": "operator_confirm_required",
            "notice": NON_AUTHORITATIVE_NOTICE,
        },
        "action": action,
        "spot": round(spot, 4),
        "recovery": recovery,
        "session": session,
        "position": position,
        "freshness": freshness,
        "sources": sources,
    }


__all__ = [
    "NON_AUTHORITATIVE_NOTICE",
    "build_payload",
    "build_position_snapshot",
    "build_recovery_snapshot",
    "build_session_snapshot",
    "classify_action",
]
