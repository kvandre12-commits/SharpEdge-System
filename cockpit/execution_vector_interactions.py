"""Advisory interaction reads across execution-vector surfaces.

This module is intentionally explanatory only. It may compare vector parts,
rank favorable/conflicted combinations, and warn about correlated evidence. It
must not score, gate, or override trade permission.
"""

from __future__ import annotations

from typing import Any

from execution_expansion_potential import has_expansion_fuel_without_participation
from range_posture import build_range_posture

NEUTRAL = "NEUTRAL"
CALLS = "CALLS"
PUTS = "PUTS"

STRONGLY_GOOD = "strongly_good"
WEAKLY_GOOD = "weakly_good"
STRONGLY_BAD = "strongly_bad"
WEAKLY_BAD = "weakly_bad"


def _score(scores: dict[str, Any], name: str) -> int:
    return int((scores.get(name) or {}).get("score") or 0)


def _normalize_bias(value: Any) -> str:
    text = str(value or "").upper()
    if text in {"CALLS", "BULLISH", "BULL"}:
        return CALLS
    if text in {"PUTS", "BEARISH", "BEAR"}:
        return PUTS
    return NEUTRAL


def _bias(scores: dict[str, Any], name: str) -> str:
    return _normalize_bias((scores.get(name) or {}).get("bias"))


def _directionally_aligned(scores: dict[str, Any], *names: str) -> tuple[bool, str]:
    biases = [_bias(scores, name) for name in names]
    directional = [bias for bias in biases if bias != NEUTRAL]
    if len(directional) != len(names):
        return False, NEUTRAL
    if len(set(directional)) != 1:
        return False, NEUTRAL
    return True, directional[0]


def _interaction(
    interaction_id: str,
    classification: str,
    label: str,
    reason: str,
    participants: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "interaction_id": interaction_id,
        "classification": classification,
        "polarity": "good" if "good" in classification else "bad",
        "strength": "strong" if classification.startswith("strongly") else "weak",
        "label": label,
        "reason": reason,
        "participants": list(participants),
    }


def _trend_acceptance_interactions(scores: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = []
    trend = _score(scores, "trend_score")
    acceptance = _score(scores, "acceptance_score")
    aligned, bias = _directionally_aligned(scores, "trend_score", "acceptance_score")
    if trend >= 70 and acceptance >= 70 and aligned:
        interactions.append(
            _interaction(
                "trend_acceptance_alignment",
                STRONGLY_GOOD,
                "Trend + acceptance aligned",
                f"{bias} has both directional drive and multi-close acceptance.",
                ("trend_score", "acceptance_score"),
            )
        )
    elif trend >= 58 and acceptance >= 60 and aligned:
        interactions.append(
            _interaction(
                "trend_acceptance_alignment",
                WEAKLY_GOOD,
                "Trend + acceptance leaning together",
                f"{bias} has some directional push with acceptable level acceptance, but not full conviction yet.",
                ("trend_score", "acceptance_score"),
            )
        )
    elif trend >= 70 and acceptance <= 45:
        interactions.append(
            _interaction(
                "trend_acceptance_conflict",
                STRONGLY_BAD,
                "Trend without acceptance",
                "Directional motion is showing up, but the tape has not actually accepted the move around a meaningful level.",
                ("trend_score", "acceptance_score"),
            )
        )
    elif trend >= 58 and acceptance < 55:
        interactions.append(
            _interaction(
                "trend_acceptance_conflict",
                WEAKLY_BAD,
                "Trend leads acceptance",
                "The trend read is ahead of the auction read, so continuation is still vulnerable to snap-back.",
                ("trend_score", "acceptance_score"),
            )
        )
    return interactions


def _trend_volume_interactions(
    scores: dict[str, Any], pa: dict[str, Any], gp: dict[str, Any]
) -> list[dict[str, Any]]:
    interactions = []
    trend = _score(scores, "trend_score")
    volume = _score(scores, "volume_score")
    aligned, bias = _directionally_aligned(scores, "trend_score", "volume_score")
    thin_but_fueled = has_expansion_fuel_without_participation(scores, pa=pa, gp=gp)
    if trend >= 70 and volume >= 80 and aligned:
        interactions.append(
            _interaction(
                "trend_volume_alignment",
                STRONGLY_GOOD,
                "Trend + participation aligned",
                f"{bias} has both directional control and participation behind it.",
                ("trend_score", "volume_score"),
            )
        )
    elif trend >= 58 and volume >= 60 and aligned:
        interactions.append(
            _interaction(
                "trend_volume_alignment",
                WEAKLY_GOOD,
                "Trend + participation acceptable",
                f"{bias} has enough participation to stay alive, but not enough to call it dominant yet.",
                ("trend_score", "volume_score"),
            )
        )
    elif trend >= 70 and volume <= 35:
        interactions.append(
            _interaction(
                "trend_volume_conflict",
                WEAKLY_BAD if thin_but_fueled else STRONGLY_BAD,
                "Trend without participation"
                if not thin_but_fueled
                else "Thin participation, but fuel exists",
                (
                    "The tape is moving, but participation is not confirming it. That is still a trust issue."
                    if not thin_but_fueled
                    else "Participation is weak, but negative gamma / pressure context means the move may still travel on fuel rather than broad sponsorship."
                ),
                ("trend_score", "volume_score"),
            )
        )
    elif trend >= 58 and volume <= 45:
        interactions.append(
            _interaction(
                "trend_volume_conflict",
                WEAKLY_BAD,
                "Trend on thin participation",
                (
                    "The move still exists, but thinner participation makes follow-through less trustworthy."
                    if not thin_but_fueled
                    else "Participation is thin, but fuel context keeps the move from being dismissed outright."
                ),
                ("trend_score", "volume_score"),
            )
        )
    return interactions


def _location_confirmation_interactions(scores: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = []
    location = _score(scores, "location_score")
    rejection = _score(scores, "rejection_score")
    trap = _score(scores, "trap_score")
    exhaustion = _score(scores, "exhaustion_score")

    reject_aligned, reject_bias = _directionally_aligned(
        scores, "location_score", "rejection_score"
    )
    trap_aligned, trap_bias = _directionally_aligned(
        scores, "location_score", "trap_score"
    )
    if location >= 70 and rejection >= 68 and reject_aligned:
        interactions.append(
            _interaction(
                "location_rejection_alignment",
                STRONGLY_GOOD,
                "Good location + clean rejection",
                f"{reject_bias} found a meaningful location and the tape rejected the other side there.",
                ("location_score", "rejection_score"),
            )
        )
    elif location >= 70 and trap >= 70 and trap_aligned:
        interactions.append(
            _interaction(
                "location_trap_alignment",
                STRONGLY_GOOD,
                "Good location + trapped counterparty",
                f"{trap_bias} has edge-location context and a trapped opposing side, which is a real combo instead of a random candle story.",
                ("location_score", "trap_score"),
            )
        )

    exhaustion_bias = _bias(scores, "exhaustion_score")
    location_bias = _bias(scores, "location_score")
    if (
        location >= 68
        and exhaustion >= 68
        and location_bias != NEUTRAL
        and exhaustion_bias != NEUTRAL
        and location_bias != exhaustion_bias
    ):
        interactions.append(
            _interaction(
                "location_exhaustion_conflict",
                STRONGLY_BAD,
                "Good location but stretched tape",
                "Location looks attractive, but exhaustion is leaning the other way hard enough to warn against chasing that edge late.",
                ("location_score", "exhaustion_score"),
            )
        )
    elif location <= 45 and max(rejection, trap) >= 70:
        interactions.append(
            _interaction(
                "tactical_signal_without_location",
                WEAKLY_BAD,
                "Tactical signal away from real location",
                "A sharp candle or trap exists, but location context is weak, so the pattern may be narratively cute and structurally mediocre.",
                ("location_score", "rejection_score", "trap_score"),
            )
        )
    return interactions


def _dealer_session_interactions(
    scores: dict[str, Any], pa: dict[str, Any], gp: dict[str, Any]
) -> list[dict[str, Any]]:
    interactions = []
    dealer = _score(scores, "dealer_gamma_score")
    time_of_day = _score(scores, "time_of_day_score")
    volume = _score(scores, "volume_score")
    compression = _score(scores, "compression_score")
    regime = str((gp or {}).get("regime") or "").lower()
    posture = build_range_posture(pa)

    if regime == "negative" and dealer >= 62 and time_of_day >= 68 and volume >= 60:
        interactions.append(
            _interaction(
                "negative_gamma_expansion_window",
                STRONGLY_GOOD,
                "Negative-gamma expansion window",
                "Dealer context, session window, and participation all lean toward expansion instead of pin drift.",
                ("dealer_gamma_score", "time_of_day_score", "volume_score"),
            )
        )
    elif regime == "negative" and dealer >= 58 and compression >= 65:
        interactions.append(
            _interaction(
                "negative_gamma_compression_release",
                WEAKLY_GOOD,
                "Compression with expansion backdrop",
                "The tape is coiled and dealer context is not fighting expansion, so a release would be more believable than usual.",
                ("dealer_gamma_score", "compression_score"),
            )
        )
    if (
        regime == "positive"
        and dealer <= 45
        and time_of_day <= 42
        and bool(posture.get("is_near_value"))
    ):
        interactions.append(
            _interaction(
                "positive_gamma_midday_pin_risk",
                STRONGLY_BAD,
                "Positive-gamma pin/chop combo",
                "Dealer pinning plus a weak session window and no real VWAP displacement is classic chop bait for directional trades.",
                ("dealer_gamma_score", "time_of_day_score"),
            )
        )
    return interactions


def _momentum_echo_interactions(scores: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = []
    trend = _score(scores, "trend_score")
    pressure = _score(scores, "pressure_score")
    regime = _score(scores, "regime_score")
    volume = _score(scores, "volume_score")
    acceptance = _score(scores, "acceptance_score")
    aligned, bias = _directionally_aligned(
        scores,
        "trend_score",
        "pressure_score",
        "regime_score",
    )
    if not aligned or min(trend, pressure, regime) < 58:
        return interactions
    if volume >= 60 or acceptance >= 60:
        interactions.append(
            _interaction(
                "momentum_chorus_with_support",
                WEAKLY_GOOD,
                "Momentum chorus is supported",
                f"Trend, pressure, and regime all lean {bias}, but they are correlated voices, so the confirmation stays useful rather than magical.",
                ("trend_score", "pressure_score", "regime_score"),
            )
        )
    else:
        interactions.append(
            _interaction(
                "momentum_chorus_without_support",
                WEAKLY_BAD,
                "Momentum chorus without outside proof",
                "Trend, pressure, and regime agree, but they mostly rhyme with each other. Without acceptance or participation, that chorus can be score soup.",
                ("trend_score", "pressure_score", "regime_score"),
            )
        )
    return interactions


def _classification_rank(classification: str) -> int:
    return {
        STRONGLY_BAD: 4,
        STRONGLY_GOOD: 3,
        WEAKLY_BAD: 2,
        WEAKLY_GOOD: 1,
    }.get(classification, 0)


def build_execution_vector_interactions(
    scores: dict[str, Any],
    *,
    pa: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ranked interaction reads across vector surfaces.

    Input `scores` is expected to be the serialized score packet already exposed
    on the trade-permission card.
    """
    pa = pa or {}
    gp = gp or {}
    interactions = []
    interactions.extend(_trend_acceptance_interactions(scores))
    interactions.extend(_trend_volume_interactions(scores, pa, gp))
    interactions.extend(_location_confirmation_interactions(scores))
    interactions.extend(_dealer_session_interactions(scores, pa, gp))
    interactions.extend(_momentum_echo_interactions(scores))
    ranked = sorted(
        interactions,
        key=lambda item: (_classification_rank(item["classification"]), item["label"]),
        reverse=True,
    )
    favorable = [item for item in ranked if item["polarity"] == "good"]
    warnings = [item for item in ranked if item["polarity"] == "bad"]
    strong_good = sum(item["strength"] == "strong" for item in favorable)
    strong_bad = sum(item["strength"] == "strong" for item in warnings)
    if strong_bad > strong_good and warnings:
        balance = "adverse"
    elif strong_good > strong_bad and favorable:
        balance = "favorable"
    elif ranked:
        balance = "mixed"
    else:
        balance = "sparse"
    return {
        "schema": "sharpedge.execution_vector_interactions.v1",
        "summary": {
            "interaction_balance": balance,
            "favorable_count": len(favorable),
            "warning_count": len(warnings),
            "strong_favorable_count": strong_good,
            "strong_warning_count": strong_bad,
        },
        "best": favorable[:4],
        "warnings": warnings[:4],
        "all": ranked,
    }


__all__ = [
    "STRONGLY_BAD",
    "STRONGLY_GOOD",
    "WEAKLY_BAD",
    "WEAKLY_GOOD",
    "build_execution_vector_interactions",
]
