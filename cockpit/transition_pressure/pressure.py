"""Synthesize transition pressure state."""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _auction_directional_bias(
    pa: dict[str, Any],
    potential_energy: dict[str, Any],
    permission_velocity: int,
    trend_velocity: int,
    acceptance_velocity: int,
    participation_velocity: int,
) -> str:
    posture = build_range_posture(pa)
    mom15 = float(pa.get("mom15") or 0.0)
    compression = int(
        (potential_energy.get("compression_score") or {}).get("score") or 0
    )
    location = potential_energy.get("location_pressure") or {}
    gamma = potential_energy.get("gamma_constraint") or {}
    failed_auction = int(
        (potential_energy.get("failed_auction_score") or {}).get("score") or 0
    )
    level_state = potential_energy.get("level_state_pressure") or {}
    location_bias = str(location.get("bias") or "unclear")
    gamma_bias = str(gamma.get("bias") or "unclear")
    level_bias = str(level_state.get("bias") or "unclear")

    upside_drive = (
        max(permission_velocity, 0) * 1.8
        + max(trend_velocity, 0) * 1.2
        + max(acceptance_velocity, 0) * 1.1
        + max(participation_velocity, 0) * 0.8
        + (10 if location_bias == "upside" else 0)
        + (6 if gamma_bias == "upside" else 0)
        + (8 if level_bias == "upside" else 0)
    )
    downside_drive = (
        max(-permission_velocity, 0) * 1.8
        + max(-trend_velocity, 0) * 1.2
        + max(-acceptance_velocity, 0) * 1.1
        + max(-participation_velocity, 0) * 0.8
        + (10 if location_bias == "downside" else 0)
        + (6 if gamma_bias == "downside" else 0)
        + (8 if level_bias == "downside" else 0)
    )

    upside_failure = (
        bool(posture.get("is_extreme"))
        and str(posture.get("side")) == "upside"
        and compression >= 55
        and location_bias == "upside"
        and (
            (permission_velocity <= 0 and participation_velocity <= 0) or mom15 < -0.02
        )
    )
    downside_failure = (
        bool(posture.get("is_extreme"))
        and str(posture.get("side")) == "downside"
        and compression >= 55
        and location_bias == "downside"
        and ((permission_velocity <= 0 and participation_velocity <= 0) or mom15 > 0.02)
    )
    if upside_failure:
        return "failed_upside_release"
    if downside_failure:
        return "failed_downside_release"
    if compression >= 65 and abs(upside_drive - downside_drive) <= 10:
        return "two_way_compression"
    if upside_drive >= 24 and upside_drive - downside_drive >= 8:
        return "upside_release_possible"
    if downside_drive >= 24 and downside_drive - upside_drive >= 8:
        return "downside_release_possible"
    if failed_auction >= 55 and acceptance_velocity > 0 and permission_velocity > 0:
        return "upside_release_possible"
    if failed_auction >= 55 and acceptance_velocity < 0 and permission_velocity < 0:
        return "downside_release_possible"
    return "unclear"


def _attention_state(transition_state: str) -> str:
    return {
        "dormant": "ignore",
        "building": "watch",
        "pressurized": "prepare",
        "release_candidate": "require_trigger",
        "resolving": "execution_takes_over",
    }.get(transition_state, "watch")


def _transition_state(score: int, pa: dict[str, Any], permission_velocity: int) -> str:
    if score < 32:
        return "dormant"
    if score < 52:
        return "building"
    if score < 72:
        return "pressurized"
    if abs(float(pa.get("mom15") or 0.0)) <= 0.06 or permission_velocity > 0:
        return "release_candidate"
    return "resolving"


def _permission_leads_price(
    permission_delta: dict[str, Any],
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    history = [*prior_receipts[-2:], current_receipt]
    if len(history) < 2:
        return {"active": False, "reason": "no prior receipt to compare"}
    permissions = [row.get("permission") for row in history]
    spots = [row.get("spot") for row in history]
    if not all(isinstance(value, (int, float)) for value in permissions + spots):
        return {"active": False, "reason": "permission/spot history unavailable"}
    permission_change = int(float(permissions[-1]) - float(permissions[0]))
    spot_start = float(spots[0])
    spot_end = float(spots[-1])
    price_delta_pct = (
        ((spot_end - spot_start) / spot_start * 100) if spot_start else 0.0
    )
    latest_velocity = int(permission_delta.get("velocity") or 0)
    price_still_balanced = abs(price_delta_pct) <= 0.15
    improving = permission_change >= 5 and latest_velocity > 0
    active = improving and price_still_balanced
    streak = 1
    for idx in range(len(permissions) - 1, 0, -1):
        delta = float(permissions[idx]) - float(permissions[idx - 1])
        if delta > 0:
            streak += 1
        else:
            break
    reason = f"permission {permission_change:+d} across {len(history)} reads while spot changed {price_delta_pct:+.2f}%"
    return {
        "active": active,
        "price_delta_pct": round(price_delta_pct, 3),
        "permission_change": permission_change,
        "streak_reads": streak,
        "reason": reason,
    }


def _top_reasons(reason_map: list[tuple[str, int, str]]) -> list[str]:
    ranked = sorted(reason_map, key=lambda row: row[1], reverse=True)
    return [item[2] for item in ranked[:3] if item[1] > 0]


def _pressure_proxy(receipt: dict[str, Any]) -> int | None:
    permission = receipt.get("permission")
    scores = receipt.get("feature_scores") or {}
    trend = (scores.get("trend_score") or {}).get("score")
    acceptance = (scores.get("acceptance_score") or {}).get("score")
    participation = (scores.get("volume_score") or {}).get("score")
    values = [permission, trend, acceptance, participation]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return _clamp_score(
        (float(permission) * 0.36)
        + (float(trend) * 0.21)
        + (float(acceptance) * 0.25)
        + (float(participation) * 0.18)
    )


def _pressure_persistence(
    current_score: int,
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    proxies = [
        value
        for value in [
            _pressure_proxy(item) for item in [*prior_receipts[-3:], current_receipt]
        ]
        if isinstance(value, int)
    ]
    if len(proxies) < 2:
        return {
            "state": "new",
            "bars": len(proxies) or 1,
            "label": f"new_{len(proxies) or 1}_bar",
            "reason": "no persistence history yet",
        }
    deltas = [proxies[idx] - proxies[idx - 1] for idx in range(1, len(proxies))]
    latest_delta = deltas[-1] if deltas else 0
    if all(delta >= 2 for delta in deltas[-2:]) and latest_delta >= 2:
        state = "building"
    elif all(abs(delta) <= 2 for delta in deltas[-2:]):
        state = "holding"
    elif latest_delta <= -3:
        state = "decaying"
    elif len(deltas) >= 2 and deltas[-2] <= -3 and latest_delta >= 2:
        state = "recycling"
    else:
        state = "holding" if latest_delta >= 0 else "decaying"

    bars = 1
    for delta in reversed(deltas):
        match = (
            (state == "building" and delta >= 2)
            or (state == "holding" and abs(delta) <= 2)
            or (state == "decaying" and delta <= -3)
            or (state == "recycling" and delta >= 2)
        )
        if not match:
            break
        bars += 1
    return {
        "state": state,
        "bars": bars,
        "label": f"{state}_{bars}_bars",
        "reason": f"pressure proxy path {proxies} -> current {current_score}",
    }


def build_transition_pressure_state(
    pa: dict[str, Any],
    deltas: dict[str, dict[str, Any]],
    potential_energy: dict[str, Any],
    interactions: dict[str, Any],
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    permission_delta = deltas.get("permission_delta") or {}
    trend_delta = deltas.get("trend_delta") or {}
    acceptance_delta = deltas.get("acceptance_delta") or {}
    participation_delta = deltas.get("participation_delta") or {}

    permission_velocity = int(permission_delta.get("velocity") or 0)
    trend_velocity = int(trend_delta.get("velocity") or 0)
    acceptance_velocity = int(acceptance_delta.get("velocity") or 0)
    participation_velocity = int(participation_delta.get("velocity") or 0)

    energy_total = int(potential_energy.get("total_energy_score") or 0)
    interaction_surfaces = interactions.get("surfaces") or []
    interaction_score = max(
        (int(item.get("score") or 0) for item in interaction_surfaces), default=0
    )

    score = _clamp_score(
        12
        + max(permission_velocity, 0) * 3.0
        + max(acceptance_velocity, 0) * 1.9
        + max(trend_velocity, 0) * 1.5
        + max(participation_velocity, 0) * 1.4
        + energy_total * 0.34
        + interaction_score * 0.18
    )

    transition_state = _transition_state(score, pa, permission_velocity)
    attention = _attention_state(transition_state)
    permission_lead = _permission_leads_price(
        permission_delta, current_receipt, prior_receipts
    )
    persistence = _pressure_persistence(score, current_receipt, prior_receipts)

    reason_map = [
        ("permission", max(permission_velocity, 0) * 5, "permission rising"),
        ("acceptance", max(acceptance_velocity, 0) * 4, "acceptance building"),
        (
            "balance",
            int((potential_energy.get("location_pressure") or {}).get("score") or 0),
            "balance narrowing into a decision area",
        ),
        (
            "gamma",
            int((potential_energy.get("gamma_constraint") or {}).get("score") or 0),
            "gamma constraint active",
        ),
        (
            "compression",
            int((potential_energy.get("compression_score") or {}).get("score") or 0),
            "compression storing energy",
        ),
        (
            "level_state",
            int(
                (potential_energy.get("level_state_pressure") or {}).get("score") or 0
            ),
            "level posture is leaning one way",
        ),
    ]
    top_reasons = _top_reasons(reason_map)
    reason = ", ".join(top_reasons) or "transition pressure is muted"

    return {
        "schema": "sharpedge.transition_pressure.v1",
        "transition_pressure_score": score,
        "transition_state": transition_state,
        "attention_state": attention,
        "directional_bias": _auction_directional_bias(
            pa,
            potential_energy,
            permission_velocity,
            trend_velocity,
            acceptance_velocity,
            participation_velocity,
        ),
        "persistence": persistence,
        "reason": reason,
        "permission_leads_price": permission_lead,
        "pressure_sources": top_reasons,
    }


__all__ = ["build_transition_pressure_state"]
