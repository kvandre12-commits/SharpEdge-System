"""Potential-energy surfaces for transition pressure."""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _distance_pct(spot: float, level: Any) -> float | None:
    if not isinstance(level, (int, float)) or spot <= 0:
        return None
    return abs(float(level) - spot) / spot * 100.0


def _compression_score(volatility_structure: dict[str, Any]) -> dict[str, Any]:
    if not volatility_structure:
        return {"score": 0, "state": "unavailable", "reason": "no volatility structure"}
    score = 18
    reasons = []
    if volatility_structure.get("compression"):
        score += 28
        reasons.append("compression active")
    if volatility_structure.get("narrow_channel"):
        score += 20
        reasons.append("channel is narrow")
    if volatility_structure.get("coil"):
        score += 18
        reasons.append("coil detected")
    channel_pct = float(volatility_structure.get("channel_pct") or 0.0)
    if channel_pct and channel_pct <= 0.12:
        score += 12
        reasons.append("rotation is tight")
    return {
        "score": _clamp_score(score),
        "state": str(volatility_structure.get("volatility_state") or "unknown"),
        "reason": "; ".join(reasons) or "compression not notable",
    }


def _failed_auction_score(
    setups: list[dict[str, Any]] | None,
    decision_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    setups = setups or []
    events = (decision_receipt or {}).get("setup_events") or []
    failed_tags = [
        item
        for item in setups
        if "FAILED" in str(item.get("tag") or "").upper()
        or "EXHAUSTION" in str(item.get("tag") or "").upper()
    ]
    score = 12
    reasons = []
    if failed_tags:
        score += min(45, 18 * len(failed_tags))
        reasons.append(f"{len(failed_tags)} failed-auction style setup(s) active")
    confirmed_events = [
        item
        for item in events
        if "FAILED" in str(item.get("event_type") or "").upper()
        and str(item.get("status") or "").lower() == "confirmed"
    ]
    if confirmed_events:
        score += 22
        reasons.append("failed-auction event confirmed")
    return {
        "score": _clamp_score(score),
        "reason": "; ".join(reasons) or "no failed-auction pressure building",
    }


def _location_pressure(
    pa: dict[str, Any],
    op: dict[str, Any],
    volatility_structure: dict[str, Any],
) -> dict[str, Any]:
    spot = float(pa.get("spot") or 0.0)
    posture = build_range_posture(pa)
    balance_width = float(pa.get("balance_width_pct") or 0.0)
    near_call = _distance_pct(spot, op.get("call_wall"))
    near_put = _distance_pct(spot, op.get("put_wall"))
    score = 10
    bias = "unclear"
    reasons = []
    if bool(posture.get("is_extreme")) and str(posture.get("side")) == "upside":
        score += 22
        bias = "upside"
        reasons.append("price is pressing session highs")
    elif bool(posture.get("is_extreme")) and str(posture.get("side")) == "downside":
        score += 22
        bias = "downside"
        reasons.append("price is pressing session lows")
    if 0 < balance_width <= 0.08:
        score += 18
        reasons.append("balance is very narrow")
    if near_call is not None and near_call <= 0.18:
        score += 18
        bias = "upside"
        reasons.append("price is crowding call-wall resistance")
    if near_put is not None and near_put <= 0.18:
        score += 18
        bias = "downside"
        reasons.append("price is crowding put-wall support")
    channel_pct = float(volatility_structure.get("channel_pct") or 0.0)
    if channel_pct and channel_pct <= 0.12:
        score += 10
        reasons.append("rotation range is tight")
    return {
        "score": _clamp_score(score),
        "bias": bias,
        "reason": "; ".join(reasons) or "location pressure is muted",
    }


def _gamma_constraint(
    pa: dict[str, Any], op: dict[str, Any], gp: dict[str, Any]
) -> dict[str, Any]:
    spot = float(pa.get("spot") or 0.0)
    regime = str(gp.get("regime") or "unknown")
    pin_dist = _distance_pct(spot, gp.get("pin"))
    call_dist = _distance_pct(spot, op.get("call_wall"))
    put_dist = _distance_pct(spot, op.get("put_wall"))
    score = 12
    reasons = []
    bias = "unclear"
    if regime == "positive":
        score += 20
        reasons.append("positive gamma is constraining travel")
        if pin_dist is not None and pin_dist <= 0.12:
            score += 24
            reasons.append("price is pinned near gamma magnet")
        if call_dist is not None and call_dist <= 0.16:
            score += 14
            bias = "upside"
            reasons.append("upside is leaning on call-wall constraint")
        if put_dist is not None and put_dist <= 0.16:
            score += 14
            bias = "downside"
            reasons.append("downside is leaning on put-wall constraint")
    elif regime == "negative":
        score += 26
        reasons.append("negative gamma can destabilize balance once released")
        bias = "two_way"
    else:
        reasons.append("gamma regime is unclear")
    return {
        "score": _clamp_score(score),
        "bias": bias,
        "regime": regime,
        "reason": "; ".join(reasons),
    }


def _level_state_pressure(
    level_states: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    packets = level_states or {}
    candidates: list[dict[str, Any]] = []
    for name, state in packets.items():
        event_state = str(state.get("event_state") or "")
        role = str(state.get("role") or "reference")
        summary = str(state.get("summary") or f"{name} state unavailable")
        if event_state == "failed_break_reclaimed":
            candidates.append(
                {
                    "score": 82,
                    "bias": "upside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "failed_break_rejected":
            candidates.append(
                {
                    "score": 82,
                    "bias": "downside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "testing_resistance":
            candidates.append(
                {
                    "score": 64,
                    "bias": "upside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "testing_support":
            candidates.append(
                {
                    "score": 64,
                    "bias": "downside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "accepted_above_resistance":
            candidates.append(
                {
                    "score": 58,
                    "bias": "upside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "lost_support":
            candidates.append(
                {
                    "score": 58,
                    "bias": "downside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif role == "support" and event_state == "holding_above_support":
            candidates.append(
                {
                    "score": 44,
                    "bias": "upside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif role == "resistance" and event_state == "holding_below_resistance":
            candidates.append(
                {
                    "score": 44,
                    "bias": "downside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "accepted_above_reference":
            candidates.append(
                {
                    "score": 36,
                    "bias": "upside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
        elif event_state == "accepted_below_reference":
            candidates.append(
                {
                    "score": 36,
                    "bias": "downside",
                    "level_name": name,
                    "state": event_state,
                    "reason": summary,
                }
            )
    if not candidates:
        return {
            "score": 0,
            "bias": "unclear",
            "state": "unavailable",
            "reason": "no level-state pressure packet",
        }
    chosen = max(candidates, key=lambda item: item["score"])
    return {
        "score": _clamp_score(chosen["score"]),
        "bias": chosen["bias"],
        "state": chosen["state"],
        "level_name": chosen["level_name"],
        "reason": chosen["reason"],
    }


def build_potential_energy(
    pa: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
    volatility_structure: dict[str, Any],
    setups: list[dict[str, Any]] | None,
    decision_receipt: dict[str, Any] | None,
    level_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    compression = _compression_score(volatility_structure or {})
    failed_auction = _failed_auction_score(setups, decision_receipt)
    location = _location_pressure(pa or {}, op or {}, volatility_structure or {})
    gamma = _gamma_constraint(pa or {}, op or {}, gp or {})
    level_state = _level_state_pressure(level_states)
    total = _clamp_score(
        (compression["score"] * 0.30)
        + (failed_auction["score"] * 0.16)
        + (location["score"] * 0.20)
        + (gamma["score"] * 0.20)
        + (level_state["score"] * 0.14)
    )
    return {
        "schema": "sharpedge.potential_energy.v1",
        "compression_score": compression,
        "failed_auction_score": failed_auction,
        "location_pressure": location,
        "gamma_constraint": gamma,
        "level_state_pressure": level_state,
        "total_energy_score": total,
    }


__all__ = ["build_potential_energy"]
