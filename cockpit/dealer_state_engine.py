"""Dealer microstructure state engine for SharpEdge.

Dealer owns measured options/dealer microstructure facts:
- gamma regime
- pin gravity
- wall pressure

It does not own premium richness, trend, or execution permission.
"""

from __future__ import annotations

from typing import Any

from reference_geometry import distance_pct

WALL_PROXIMITY_PCT = 0.20
PIN_PROXIMITY_PCT = 0.25


def _wall_state(spot: float | None, call_wall: Any, put_wall: Any) -> dict[str, Any]:
    call_dist = (
        distance_pct(spot, float(call_wall))
        if isinstance(call_wall, (int, float))
        else None
    )
    put_dist = (
        distance_pct(spot, float(put_wall))
        if isinstance(put_wall, (int, float))
        else None
    )
    if call_dist is not None and call_dist <= WALL_PROXIMITY_PCT:
        return {
            "state": "near_call_wall",
            "bias": "PUTS",
            "call_wall": call_wall,
            "put_wall": put_wall,
            "call_dist_pct": call_dist,
            "put_dist_pct": put_dist,
            "reason": f"near call wall {float(call_wall):g}; upside resistance",
        }
    if put_dist is not None and put_dist <= WALL_PROXIMITY_PCT:
        return {
            "state": "near_put_wall",
            "bias": "CALLS",
            "call_wall": call_wall,
            "put_wall": put_wall,
            "call_dist_pct": call_dist,
            "put_dist_pct": put_dist,
            "reason": f"near put wall {float(put_wall):g}; downside support",
        }
    return {
        "state": "no_near_wall",
        "bias": "NEUTRAL",
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_dist_pct": call_dist,
        "put_dist_pct": put_dist,
        "reason": "no nearby wall pressure",
    }


def _pin_state(spot: float | None, pin: Any) -> dict[str, Any]:
    if not isinstance(pin, (int, float)):
        return {
            "state": "pin_unavailable",
            "pin": pin,
            "pin_dist_pct": None,
            "reason": "pin unavailable",
        }
    pin_dist = distance_pct(spot, float(pin))
    if pin_dist is not None and pin_dist <= PIN_PROXIMITY_PCT:
        return {
            "state": "near_pin",
            "pin": pin,
            "pin_dist_pct": pin_dist,
            "reason": f"near pin {float(pin):g}",
        }
    return {
        "state": "far_pin",
        "pin": pin,
        "pin_dist_pct": pin_dist,
        "reason": f"pin {pin_dist:.2f}% away"
        if pin_dist is not None
        else "pin unavailable",
    }


def _gamma_state(gp: dict[str, Any] | None) -> dict[str, Any]:
    data = gp or {}
    regime = str(data.get("regime") or "unknown").lower()
    quality = str(data.get("gamma_data_quality") or "missing").lower()
    dte = data.get("dte")
    expired = isinstance(dte, (int, float)) and dte < 0
    if expired or quality != "ok" or regime not in {"positive", "negative"}:
        reason = (
            "gamma contract is expired"
            if expired
            else "gamma data quality is weak or unknown"
        )
        return {
            "state": "gamma_unknown",
            "regime": regime,
            "quality": quality,
            "dte": dte,
            "reason": reason,
        }
    if regime == "positive":
        return {
            "state": "gamma_damping",
            "regime": regime,
            "quality": quality,
            "reason": "positive gamma/OI proxy may dampen directional follow-through",
        }
    return {
        "state": "gamma_expansion",
        "regime": regime,
        "quality": quality,
        "reason": "negative gamma/OI proxy may support expansion",
    }


def build_dealer_state(
    pa: dict[str, Any] | None,
    op: dict[str, Any] | None,
    gp: dict[str, Any] | None,
) -> dict[str, Any]:
    spot = (pa or {}).get("spot")
    call_wall = (op or {}).get("call_wall")
    put_wall = (op or {}).get("put_wall")
    pin = (gp or {}).get("pin")
    gamma_state = _gamma_state(gp)
    pin_state = _pin_state(float(spot) if isinstance(spot, (int, float)) else None, pin)
    wall_state = _wall_state(
        float(spot) if isinstance(spot, (int, float)) else None, call_wall, put_wall
    )
    packet = {
        "schema": "sharpedge.dealer_state.v1",
        "state": "dealer_unknown",
        "bias": "NEUTRAL",
        "reason": "dealer state unavailable",
        "spot": spot,
        "gamma_state": gamma_state,
        "pin_state": pin_state,
        "wall_state": wall_state,
    }
    gamma_name = str(gamma_state.get("state") or "gamma_unknown")
    wall_reason = str(wall_state.get("reason") or "")
    pin_reason = str(pin_state.get("reason") or "")
    wall_bias = str(wall_state.get("bias") or "NEUTRAL")
    if gamma_name == "gamma_unknown":
        return {
            **packet,
            "state": "dealer_unknown",
            "bias": "NEUTRAL",
            "reason": f"dealer unknown: {gamma_state['reason']}; {pin_reason}; {wall_reason}",
        }
    if gamma_name == "gamma_damping" and (
        pin_state.get("state") == "near_pin"
        or wall_state.get("state") != "no_near_wall"
    ):
        return {
            **packet,
            "state": "positive_gamma_gravity",
            "bias": wall_bias,
            "reason": f"positive gamma/OI proxy pinning: {pin_reason}; {wall_reason if wall_state.get('state') != 'no_near_wall' else 'pin/chop risk'}",
        }
    if gamma_name == "gamma_damping":
        return {
            **packet,
            "state": "positive_gamma_context",
            "bias": wall_bias,
            "reason": gamma_state["reason"],
        }
    return {
        **packet,
        "state": "negative_gamma_expansion",
        "bias": wall_bias,
        "reason": f"negative gamma/OI proxy may support expansion; {wall_reason if wall_state.get('state') != 'no_near_wall' else 'accepted breaks can run'}",
    }


__all__ = [
    "PIN_PROXIMITY_PCT",
    "WALL_PROXIMITY_PCT",
    "build_dealer_state",
]
