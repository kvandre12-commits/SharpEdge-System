"""Conditional execution framework for Candle Coach outputs.

Candles are event detectors, not trade theses. This module turns a named
configuration into a small gate packet that makes the missing evidence explicit
instead of letting folklore masquerade as permission.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from options_flow_proxy import build_options_flow_proxy

CANDLE_FRAMEWORK_SCHEMA = "sharpedge.candle_framework.v1"
EV_FORMULA = "EV = P(W) × avg(W) - P(L) × avg(L) - execution_cost > 0"
TRADEABLE_OUTPUTS = (
    "No information",
    "Watch",
    "Qualified setup",
    "Execution permitted",
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(label: str, state: str, message: str) -> dict[str, str]:
    return {"label": label, "status": state, "message": message}


def _state_packet(context: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = context.get(name)
        if isinstance(value, dict):
            return value
    permission = context.get("permission") or context.get("trade_permission") or {}
    if isinstance(permission, dict):
        for name in names:
            value = permission.get(name)
            if isinstance(value, dict):
                return value
    return {}


def _state_reason(packet: dict[str, Any], fallback: str) -> str:
    return str(packet.get("reason") or packet.get("summary") or fallback)


def data_integrity_gate(bar: dict[str, Any]) -> dict[str, Any]:
    """Validate whether the current bar can support any directional read."""
    candle_range = _num(bar.get("range"))
    volume = _num(bar.get("volume"))

    checks = {
        "nonzero_range": candle_range > 0,
        "positive_volume": volume > 0,
        "trade_count_available": bar.get("trade_count") is not None,
        "quote_updates_available": bar.get("quote_updates") is not None,
        "spread_available": bar.get("spread") is not None,
        "feed_continuity_known": bar.get("stale_feed") is not None,
    }

    if not checks["nonzero_range"] or not checks["positive_volume"]:
        status = "fail"
        output = "No information"
        message = (
            "Insufficient bar information — verify volume, trade count, spread, "
            "and feed continuity. No directional inference permitted."
        )
    elif not all(checks.values()):
        status = "partial"
        output = "Watch"
        message = (
            "OHLCV-only bar. Shape can describe an event, but missing trade-count, "
            "quote-update, spread, and feed-continuity evidence blocks trade permission."
        )
    else:
        status = "pass"
        output = "Watch"
        message = (
            "Bar passes basic information checks. Continue through location, regime, "
            "participation, acceptance, and expectancy gates before any setup upgrade."
        )

    return {
        "label": "Data integrity",
        "status": status,
        "output": output,
        "message": message,
        "checks": checks,
    }


def _participation_gate(
    packets: list[dict[str, Any]], current: dict[str, Any], context: dict[str, Any]
) -> dict[str, str]:
    volume_state = _state_packet(context, "execution_volume_state", "volume_state")
    if volume_state:
        confirmation = str(
            volume_state.get("confirmation") or volume_state.get("state") or "observed"
        )
        return _status(
            "Participation and order flow",
            confirmation,
            _state_reason(
                volume_state,
                "SharpEdge volume surface observed, but order-flow/depth evidence is still unavailable.",
            )
            + " Order-flow/depth still missing.",
        )
    prior = [
        _num(bar.get("volume")) for bar in packets[:-1] if _num(bar.get("volume")) > 0
    ]
    current_volume = _num(current.get("volume"))
    if not prior or current_volume <= 0:
        return _status(
            "Participation and order flow",
            "not_evaluable",
            "Need relative volume, trade-arrival rate, dollar volume, order-flow imbalance, and depth.",
        )
    baseline = median(prior)
    rel = current_volume / baseline if baseline else 0.0
    state = "observed" if rel >= 1.2 else "weak"
    return _status(
        "Participation and order flow",
        state,
        (
            f"Volume is {rel:.2f}× recent median, but raw volume is not enough. "
            "Need aggressive buy/sell imbalance, queue/depth, spread, sweeps, absorption, and resilience."
        ),
    )


def _location_gate(context: dict[str, Any]) -> dict[str, str]:
    location = _state_packet(context, "execution_location_state", "location_state")
    if not location:
        return _status(
            "Location",
            "required",
            "Locate the event versus PDH/PDL/PDC, overnight range, ORH/ORL, VWAP, HVN/LVN, swings, gaps, expected move, and option/gamma levels.",
        )
    nearest = location.get("nearest_reference") or {}
    suffix = ""
    if nearest:
        suffix = (
            f" Nearest: {nearest.get('reference_name')} "
            f"{nearest.get('distance_pct')}% {nearest.get('relation')}"
        )
    return _status(
        "Location",
        str(location.get("state") or "observed"),
        _state_reason(location, "Location observed.") + suffix,
    )


def _regime_gate(context_text: str, context: dict[str, Any]) -> dict[str, str]:
    dealer = _state_packet(context, "execution_dealer_state", "dealer_state")
    trend = _state_packet(context, "execution_trend_state", "trend_state")
    volatility = context.get("volatility_structure") or {}
    parts = [f"Local candle context: {context_text}."]
    if dealer:
        parts.append(_state_reason(dealer, "Dealer/gamma state observed."))
    if trend:
        parts.append(_state_reason(trend, "Trend state observed."))
    if isinstance(volatility, dict) and volatility:
        parts.append(
            f"Volatility/structure: {volatility.get('volatility_state')} / {volatility.get('structure_state')}."
        )
    state = str((dealer or trend or volatility).get("state") or "observed")
    return _status("Regime", state, " ".join(parts))


def _acceptance_gate(context: dict[str, Any]) -> dict[str, str]:
    acceptance = _state_packet(
        context, "execution_acceptance_state", "acceptance_state"
    )
    if not acceptance:
        return _status(
            "Acceptance",
            "required",
            "A boundary touch is not enough. Need penetration distance, time beyond level, volume beyond, retest hold, VWAP/microprice migration, and failed reclaim evidence.",
        )
    representative = acceptance.get("representative_level") or {}
    suffix = ""
    if representative:
        suffix = f" Representative: {representative.get('reason')}"
    return _status(
        "Acceptance",
        str(acceptance.get("state") or "observed"),
        _state_reason(acceptance, "Acceptance state observed.") + suffix,
    )


def _micro_value_status(value: Any) -> str:
    return "available" if value is not None else "missing"


def _compact_fact(label: str, status: str, value: Any, read: str) -> dict[str, Any]:
    return {"label": label, "status": status, "value": value, "read": read}


def _auction_execution_box(
    current: dict[str, Any], context_text: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Compress candle-lost execution information into one human box."""
    permission = context.get("permission") or context.get("trade_permission") or {}
    location = _state_packet(context, "execution_location_state", "location_state")
    acceptance = _state_packet(
        context, "execution_acceptance_state", "acceptance_state"
    )
    volume = _state_packet(context, "execution_volume_state", "volume_state")
    trend = _state_packet(context, "execution_trend_state", "trend_state")
    dealer = _state_packet(context, "execution_dealer_state", "dealer_state")
    pa = context.get("pa") if isinstance(context.get("pa"), dict) else {}
    op = context.get("op") if isinstance(context.get("op"), dict) else {}
    options_source = (
        context.get("options_source")
        if isinstance(context.get("options_source"), dict)
        else {}
    )
    price_source = (
        context.get("price_source")
        if isinstance(context.get("price_source"), dict)
        else {}
    )
    options_flow_proxy = build_options_flow_proxy(
        op, options_source, price_source.get("session_date")
    )
    micro = context.get("micro") if isinstance(context.get("micro"), dict) else {}
    magnitude = (
        context.get("magnitude") if isinstance(context.get("magnitude"), dict) else {}
    )
    transition = (
        context.get("transition_pressure")
        if isinstance(context.get("transition_pressure"), dict)
        else {}
    )
    volume_profile = (
        pa.get("volume_profile") if isinstance(pa.get("volume_profile"), dict) else {}
    )
    nearest = location.get("nearest_reference") or {}
    representative = acceptance.get("representative_level") or {}
    volume_reason = (
        _state_reason(volume, "volume surface unavailable")
        if volume
        else str(volume_profile.get("reason") or "unavailable")
    )
    trade_count = current.get("trade_count")
    spread = current.get("spread")
    quote_updates = current.get("quote_updates")
    call_spread = op.get("atm_call_spread_pct")
    put_spread = op.get("atm_put_spread_pct")
    friction_value = None
    if call_spread is not None or put_spread is not None:
        friction_value = {
            "atm_call_spread_pct": call_spread,
            "atm_put_spread_pct": put_spread,
        }
    speed_value = {
        "local_mult": volume_profile.get("local_mult"),
        "session_mult": volume_profile.get("session_mult"),
        "composite_mult": volume_profile.get("composite_mult"),
    }
    alignment_value = {
        "move_direction": volume_profile.get("move_direction"),
        "aligned_volume_share": volume_profile.get("aligned_volume_share"),
        "path_efficiency": volume_profile.get("path_efficiency"),
    }
    absorption_value = {
        "lower_wick_pct_of_session_range": micro.get("lower_wick"),
        "upper_wick_pct_of_session_range": micro.get("upper_wick"),
        "body_pct_of_session_range": micro.get("body"),
        "channel_position": micro.get("ch_pos"),
    }
    pressure_value = {
        "state": transition.get("transition_state"),
        "score": transition.get("transition_pressure_score"),
        "attention": transition.get("attention_state"),
    }
    magnitude_value = {
        "premium_read": magnitude.get("premium_read"),
        "realized_expected_move_pct": magnitude.get("exp_move_realized_pct"),
        "implied_expected_move_pct": magnitude.get("exp_move_implied_pct"),
    }

    facts = [
        _compact_fact(
            "Sequence of trades",
            _micro_value_status(trade_count),
            {"trade_count": trade_count},
            "Need prints/trade count to know if the candle came from sustained flow or one print.",
        ),
        _compact_fact(
            "Speed of trading",
            "proxy" if volume_profile else "missing",
            speed_value,
            "Existing volume_profile gives local/session/composite pace; true trade-arrival speed still needs prints.",
        ),
        _compact_fact(
            "Aggressor side / imbalance",
            "proxy" if volume_profile else "missing",
            alignment_value,
            "Aligned candle volume/path efficiency is a directional-flow proxy, not proof of bid/ask aggression.",
        ),
        _compact_fact(
            "Spread / execution friction",
            "proxy" if friction_value else _micro_value_status(spread),
            friction_value if friction_value else {"spread": spread},
            "Delayed CBOE ATM bid/ask spread is useful friction context; live quote spread at the level is still preferred.",
        ),
        _compact_fact(
            "Available depth",
            "missing",
            None,
            "No queue/depth ladder; cannot see whether size is actually available.",
        ),
        _compact_fact(
            "Replenishment / cancellation",
            _micro_value_status(quote_updates),
            {"quote_updates": quote_updates},
            "Need quote-update/depth changes to distinguish replenishment from vanishing liquidity.",
        ),
        _compact_fact(
            "Absorption",
            "inferred_only" if micro else "missing",
            absorption_value,
            "Wicks/body/channel position can flag rejection or failed progress; prints/depth are needed to prove absorption.",
        ),
        _compact_fact(
            "Sustained participation vs one print",
            "proxy" if volume_profile else "missing",
            alignment_value,
            "Path efficiency + aligned volume share helps separate grind participation from churn, but cannot see individual prints.",
        ),
        _compact_fact(
            "Transition pressure",
            "proxy" if transition else "missing",
            pressure_value,
            "Existing transition_pressure surface shows whether auction conditions are pressurizing or still sleepy.",
        ),
        _compact_fact(
            "Premium / movement friction",
            "proxy" if magnitude else "missing",
            magnitude_value,
            "Magnitude surface compares realized move potential with implied pricing; not order flow, but it matters for execution quality.",
        ),
        _compact_fact(
            "CBOE option flow proxy",
            "delayed_proxy"
            if options_flow_proxy.get("available")
            else "stale_proxy"
            if options_flow_proxy.get("stale")
            else "missing",
            {
                "flow": (options_flow_proxy.get("flow_pressure") or {}).get("state"),
                "call_spread": (options_flow_proxy.get("spread_proxy") or {}).get(
                    "call_quality"
                ),
                "put_spread": (options_flow_proxy.get("spread_proxy") or {}).get(
                    "put_quality"
                ),
                "iv": (options_flow_proxy.get("iv_context") or {}).get("read"),
            },
            options_flow_proxy.get("summary") or "Delayed options proxy unavailable.",
        ),
        _compact_fact(
            "Acceptance / rejection",
            str(acceptance.get("state") or "missing"),
            representative or acceptance,
            _state_reason(
                acceptance, "Acceptance is the auction proof that matters most."
            ),
        ),
    ]

    permission_score = None
    if isinstance(permission, dict):
        permission_score = permission.get(
            "execution_permission_score"
        ) or permission.get("trade_permission_score")
    human_read_parts = [
        _state_reason(location, "location unknown"),
        _state_reason(acceptance, "acceptance unknown"),
        volume_reason,
    ]
    if trend:
        human_read_parts.append(_state_reason(trend, "trend unknown"))
    if dealer:
        human_read_parts.append(_state_reason(dealer, "dealer/gamma unknown"))
    if transition:
        human_read_parts.append(
            str(transition.get("reason") or transition.get("transition_state") or "")
        )
    if magnitude:
        human_read_parts.append(f"premium {magnitude.get('premium_read', 'unknown')}")

    return {
        "schema": "sharpedge.auction_execution_box.v1",
        "authority": "human_execution_context_not_trade_permission",
        "premise": {
            "candle_context": context_text,
            "latest_close": current.get("close"),
            "range": current.get("range"),
            "body_pct": current.get("body_pct"),
        },
        "location": {
            "state": location.get("state"),
            "nearest_reference": nearest,
            "reason": _state_reason(location, "location unknown"),
        },
        "acceptance": {
            "state": acceptance.get("state"),
            "representative_level": representative,
            "reason": _state_reason(acceptance, "acceptance unknown"),
        },
        "participation": {
            "state": volume.get("confirmation")
            or volume.get("state")
            or volume_profile.get("confirmation"),
            "reason": volume_reason,
            "volume_profile": volume_profile,
        },
        "micro_proxy": micro,
        "transition_pressure": pressure_value,
        "magnitude_context": magnitude_value,
        "options_flow_proxy": options_flow_proxy,
        "execution_friction": friction_value or {"bar_spread": spread},
        "permission_context": {
            "gate": permission.get("trade_gate")
            if isinstance(permission, dict)
            else None,
            "score": permission_score,
            "bias": permission.get("bias") if isinstance(permission, dict) else None,
        },
        "facts": facts,
        "human_read": " | ".join(str(part) for part in human_read_parts if part),
        "missing_microstructure": [
            "aggressor_side",
            "order_flow_imbalance",
            "bid_ask_spread_at_level",
            "depth_ladder",
            "replenishment_and_cancellation",
            "print_sequence",
            "sweep_or_absorption_proof",
        ],
        "doctrine": (
            "The candle identifies the shape. Location supplies relevance. Order flow supplies pressure. "
            "Acceptance supplies confirmation. Empirical net expectancy supplies permission."
        ),
    }


def _expectancy_gate(context: dict[str, Any]) -> dict[str, str]:
    permission = context.get("permission") or context.get("trade_permission") or {}
    if not isinstance(permission, dict) or not permission:
        return _status(
            "Net expectancy",
            "required",
            "No trade permission without out-of-sample EV after spread, slippage, commissions, adverse selection, failed fills, and stop/fill behavior.",
        )
    score = permission.get("execution_permission_score") or permission.get(
        "trade_permission_score"
    )
    gate = permission.get("trade_gate") or "UNKNOWN"
    return _status(
        "Net expectancy",
        "missing_empirical_ev",
        f"SharpEdge permission spine says {gate} {score}/100, but candle-conditioned out-of-sample EV is not yet attached.",
    )


def _next_vector_surface(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "candle_conditioned_expectancy_surface",
        "priority": "next",
        "why": (
            "SharpEdge already has location, acceptance, volume, trend, dealer/gamma, and permission surfaces. "
            "The missing vector is empirical expectancy conditioned on the candle event plus those surfaces."
        ),
        "minimum_keys": [
            "event_name",
            "nearest_reference",
            "acceptance_state",
            "volume_confirmation",
            "gamma_regime",
            "volatility_structure",
            "forecast_horizon",
            "target_before_stop_label",
            "gross_ev",
            "execution_cost",
            "net_ev",
            "sample_size",
            "out_of_sample_window",
        ],
        "current_blocker": "No historical/live-shadow table maps candle event + SharpEdge state -> net EV after execution costs.",
    }


def build_candle_framework(
    packets: list[dict[str, Any]],
    current: dict[str, Any],
    context: str,
    sharpedge_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the six-gate conditional framework for the current candle."""
    edge_context = sharpedge_context or {}
    data_gate = data_integrity_gate(current)
    output = data_gate["output"]

    gates: list[dict[str, Any]] = [
        data_gate,
        _location_gate(edge_context),
        _regime_gate(context, edge_context),
        _participation_gate(packets, current, edge_context),
        _acceptance_gate(edge_context),
        _expectancy_gate(edge_context),
    ]

    return {
        "schema": CANDLE_FRAMEWORK_SCHEMA,
        "output": output,
        "allowed_outputs": list(TRADEABLE_OUTPUTS),
        "expected_value_formula": EV_FORMULA,
        "gates": gates,
        "auction_execution_box": _auction_execution_box(current, context, edge_context),
        "next_vector_surface": _next_vector_surface(edge_context),
        "lesson": (
            "A candle is an event detector: contraction, expansion, rejection, or close location. "
            "It becomes tradeable only inside validated data, location, regime, participation, acceptance, and net-expectancy gates."
        ),
    }


__all__ = [
    "CANDLE_FRAMEWORK_SCHEMA",
    "EV_FORMULA",
    "TRADEABLE_OUTPUTS",
    "build_candle_framework",
    "data_integrity_gate",
]
