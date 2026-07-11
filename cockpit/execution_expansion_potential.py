"""Advisory read for why price is able to move.

This packet separates participation confirmation from expansion fuel. It is
explanatory only and must not score, gate, or override trade permission.
"""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture


def _score(scores: dict[str, Any], name: str) -> int:
    return int((scores.get(name) or {}).get("score") or 0)


def _reason(scores: dict[str, Any], name: str) -> str:
    return str((scores.get(name) or {}).get("reason") or "")


def _bias(scores: dict[str, Any], name: str) -> str:
    return str((scores.get(name) or {}).get("bias") or "NEUTRAL").upper()


def _band(value: int) -> str:
    if value >= 80:
        return "high"
    if value >= 60:
        return "moderate"
    return "low"


def _mechanism(
    mechanism_id: str,
    label: str,
    family: str,
    strength: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "mechanism_id": mechanism_id,
        "label": label,
        "family": family,
        "strength": strength,
        "reason": reason,
    }


def has_expansion_fuel_without_participation(
    scores: dict[str, Any],
    *,
    pa: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
) -> bool:
    pa = pa or {}
    gp = gp or {}
    volume = _score(scores, "volume_score")
    dealer = _score(scores, "dealer_gamma_score")
    pressure = _score(scores, "pressure_score")
    acceptance = _score(scores, "acceptance_score")
    time_of_day = _score(scores, "time_of_day_score")
    compression = _score(scores, "compression_score")
    regime = str((gp or {}).get("regime") or "").lower()
    posture = build_range_posture(pa)
    return (
        volume <= 45
        and regime == "negative"
        and dealer >= 58
        and (pressure >= 48 or compression >= 65 or acceptance >= 60)
        and (time_of_day >= 52 or bool(posture.get("has_directional_displacement")))
    )


def _dominant_bias(scores: dict[str, Any], names: tuple[str, ...]) -> str:
    candidates = []
    for name in names:
        label = _bias(scores, name)
        if label not in {"CALLS", "PUTS"}:
            continue
        candidates.append((_score(scores, name), label))
    if not candidates:
        return "NEUTRAL"
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_expansion_fuel_surface(
    scores: dict[str, Any],
    *,
    pa: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet = build_execution_expansion_potential(scores, pa=pa, gp=gp)
    summary = packet.get("summary") or {}
    state = str(summary.get("state") or "mixed").lower()
    expansion_fuel = str(summary.get("expansion_fuel") or "low").lower()
    dominant = str(summary.get("dominant_mechanism") or "participation_confirmation")
    bias = _dominant_bias(
        scores,
        (
            "pressure_score",
            "acceptance_score",
            "trap_score",
            "rejection_score",
            "trend_score",
            "location_score",
            "volume_score",
        ),
    )
    score = {
        "low": 38,
        "moderate": 64,
        "high": 82,
    }.get(expansion_fuel, 38)
    if state == "high_confirmation_high_fuel":
        score = max(score, 85)
    elif state == "low_confirmation_high_fuel":
        score = max(score, 80)
    reason = {
        "dealer_gamma_feedback": "expansion fuel is active: dealer hedging feedback can keep price moving",
        "thin_liquidity_vacuum_proxy": "expansion fuel is active: thin participation can still let price travel",
        "structural_acceptance": "expansion fuel is active: structural acceptance is carrying the move",
        "counterparty_trap": "expansion fuel is active: trapped counterparties/stops can keep feeding the move",
        "stored_energy_release": "expansion fuel is active: stored compression energy can release into travel",
    }.get(dominant, "expansion fuel is limited beyond participation confirmation")
    note = str(summary.get("note") or "")
    if note:
        reason = f"{reason}; {note[0].lower() + note[1:]}"
    return {
        "score": score,
        "bias": bias,
        "reason": reason,
        "state": state,
        "expansion_fuel": expansion_fuel,
        "dominant_mechanism": dominant,
    }


def build_execution_expansion_potential(
    scores: dict[str, Any],
    *,
    pa: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pa = pa or {}
    gp = gp or {}
    volume = _score(scores, "volume_score")
    dealer = _score(scores, "dealer_gamma_score")
    acceptance = _score(scores, "acceptance_score")
    location = _score(scores, "location_score")
    trap = _score(scores, "trap_score")
    rejection = _score(scores, "rejection_score")
    time_of_day = _score(scores, "time_of_day_score")
    compression = _score(scores, "compression_score")
    regime = str((gp or {}).get("regime") or "").lower()
    mechanisms = []

    participation_band = _band(volume)
    if volume >= 80:
        mechanisms.append(
            _mechanism(
                "participation_confirmation",
                "Institutions are participating",
                "confirmation",
                "high",
                _reason(scores, "volume_score") or "participation confirms the move",
            )
        )
    elif volume >= 60:
        mechanisms.append(
            _mechanism(
                "participation_confirmation",
                "Participation is present",
                "confirmation",
                "moderate",
                _reason(scores, "volume_score")
                or "participation is present behind the move",
            )
        )
    else:
        mechanisms.append(
            _mechanism(
                "participation_confirmation",
                "Participation is thin",
                "confirmation",
                "low",
                _reason(scores, "volume_score")
                or "participation is not meaningfully confirming the move",
            )
        )

    if regime == "negative" and dealer >= 58:
        mechanisms.append(
            _mechanism(
                "dealer_gamma_feedback",
                "Dealers are chasing the move",
                "fuel",
                "high" if dealer >= 70 else "moderate",
                _reason(scores, "dealer_gamma_score")
                or "negative gamma supports hedging feedback and expansion",
            )
        )

    if has_expansion_fuel_without_participation(scores, pa=pa, gp=gp):
        mechanisms.append(
            _mechanism(
                "thin_liquidity_vacuum_proxy",
                "Thin participation can still travel",
                "fuel",
                "moderate",
                "Participation is weak, but negative gamma plus one-sided pressure/acceptance means price may still move because the tape is thin rather than institutionally sponsored.",
            )
        )

    if acceptance >= 70 and location >= 68:
        mechanisms.append(
            _mechanism(
                "structural_acceptance",
                "Auction acceptance is enabling travel",
                "fuel",
                "high" if acceptance >= 78 else "moderate",
                _reason(scores, "acceptance_score")
                or "multi-close acceptance can keep price moving away from value",
            )
        )

    if max(trap, rejection) >= 68:
        mechanisms.append(
            _mechanism(
                "counterparty_trap",
                "Trapped traders / stops may be feeding the move",
                "fuel",
                "moderate",
                _reason(scores, "trap_score")
                or _reason(scores, "rejection_score")
                or "failed-break and rejection evidence suggest counterparty pain can help propel price",
            )
        )

    if compression >= 65 and time_of_day >= 52:
        mechanisms.append(
            _mechanism(
                "stored_energy_release",
                "Stored energy is available",
                "fuel",
                "moderate",
                _reason(scores, "compression_score")
                or "compression can release into directional travel when the session window is supportive",
            )
        )

    fuel_strength = max(
        [
            3
            if item["strength"] == "high"
            else 2
            if item["strength"] == "moderate"
            else 1
            for item in mechanisms
            if item["family"] == "fuel"
        ],
        default=1,
    )
    expansion_fuel = {1: "low", 2: "moderate", 3: "high"}[fuel_strength]
    dominant = next(
        (item["mechanism_id"] for item in mechanisms if item["family"] == "fuel"),
        "participation_confirmation",
    )
    if participation_band == "low" and expansion_fuel in {"moderate", "high"}:
        state = "low_confirmation_high_fuel"
        note = "Participation is not confirming much, but other mechanisms can still let price travel."
    elif participation_band in {"moderate", "high"} and expansion_fuel == "low":
        state = "high_confirmation_low_fuel"
        note = "Participation is visible, but the tape may not have much extra expansion fuel beyond confirmation."
    elif participation_band == "high" and expansion_fuel in {"moderate", "high"}:
        state = "high_confirmation_high_fuel"
        note = "Both confirmation and expansion fuel are present."
    else:
        state = "mixed"
        note = "Confirmation and fuel are not saying the same thing cleanly."

    surface = build_expansion_fuel_surface_from_summary(
        scores,
        state=state,
        participation_band=participation_band,
        expansion_fuel=expansion_fuel,
        dominant=dominant,
        note=note,
    )
    return {
        "schema": "sharpedge.execution_expansion_potential.v1",
        "summary": {
            "state": state,
            "participation_confirmation": participation_band,
            "expansion_fuel": expansion_fuel,
            "dominant_mechanism": dominant,
            "note": note,
        },
        "surface": surface,
        "mechanisms": mechanisms,
    }


def build_expansion_fuel_surface_from_summary(
    scores: dict[str, Any],
    *,
    state: str,
    participation_band: str,
    expansion_fuel: str,
    dominant: str,
    note: str,
) -> dict[str, Any]:
    bias = _dominant_bias(
        scores,
        (
            "pressure_score",
            "acceptance_score",
            "trap_score",
            "rejection_score",
            "trend_score",
            "location_score",
            "volume_score",
        ),
    )
    score = {
        "low": 38,
        "moderate": 64,
        "high": 82,
    }.get(expansion_fuel, 38)
    if state == "high_confirmation_high_fuel":
        score = max(score, 85)
    elif state == "low_confirmation_high_fuel":
        score = max(score, 80)
    reason = {
        "dealer_gamma_feedback": "expansion fuel is active: dealer hedging feedback can keep price moving",
        "thin_liquidity_vacuum_proxy": "expansion fuel is active: thin participation can still let price travel",
        "structural_acceptance": "expansion fuel is active: structural acceptance is carrying the move",
        "counterparty_trap": "expansion fuel is active: trapped counterparties/stops can keep feeding the move",
        "stored_energy_release": "expansion fuel is active: stored compression energy can release into travel",
    }.get(dominant, "expansion fuel is limited beyond participation confirmation")
    if note:
        reason = f"{reason}; {note[0].lower() + note[1:]}"
    return {
        "score": score,
        "bias": bias,
        "reason": reason,
        "state": state,
        "participation_confirmation": participation_band,
        "expansion_fuel": expansion_fuel,
        "dominant_mechanism": dominant,
    }


__all__ = [
    "build_execution_expansion_potential",
    "build_expansion_fuel_surface",
    "build_expansion_fuel_surface_from_summary",
    "has_expansion_fuel_without_participation",
]
