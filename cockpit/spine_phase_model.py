"""Phase model for SharpEdge's core execution spine.

Scores answer "how strong?".
Bias answers "which side?".
Phase answers "where in the edge lifecycle are we?".

This module is metadata-only. It must not change permission math.
"""

from __future__ import annotations

from typing import Any

from execution_hierarchy import CORE_EXECUTION_SPINE_PART_NAMES
from range_posture import build_range_posture

HEAD = "head"
BODY = "body"
TAIL = "tail"
INACTIVE = "inactive"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pin_distance_pct(pa: dict[str, Any], gp: dict[str, Any]) -> float | None:
    spot = _f(pa.get("spot"))
    pin = gp.get("pin")
    if not spot or not isinstance(pin, (int, float)):
        return None
    return abs(spot - float(pin)) / spot * 100


def _setup_tags(setups: list[dict[str, Any]] | None) -> set[str]:
    return {str((setup or {}).get("tag") or "").upper() for setup in setups or []}


def _phase_meta(phase: str, reason: str) -> dict[str, str]:
    return {"phase": phase, "phase_reason": reason}


def _structure_phase(
    item: dict[str, Any], range_posture: dict[str, Any]
) -> dict[str, str]:
    score = int(item.get("score") or 0)
    if score < 55:
        return _phase_meta(INACTIVE, "structure edge is not clean enough yet")
    if score >= 78 and bool(range_posture.get("is_terminal_extreme")):
        return _phase_meta(
            TAIL, "structure is strong but already pressing a session extreme"
        )
    if score >= 78:
        return _phase_meta(HEAD, "clean sequence is asserting and can still expand")
    return _phase_meta(BODY, "directional structure is in force but not pristine")


def _acceptance_phase(
    item: dict[str, Any],
    range_posture: dict[str, Any],
    setups: list[dict[str, Any]] | None,
) -> dict[str, str]:
    score = int(item.get("score") or 0)
    tags = _setup_tags(setups)
    if score < 55:
        return _phase_meta(INACTIVE, "acceptance has not proven itself yet")
    if score >= 78 and tags & {"FAILED BREAKDOWN", "FAILED BREAKOUT"}:
        return _phase_meta(
            HEAD, "acceptance is fresh and backed by a recent failed-break event"
        )
    if score >= 78 and bool(range_posture.get("is_extended_from_value")):
        return _phase_meta(
            TAIL, "acceptance still holds, but price is getting extended from value"
        )
    return _phase_meta(BODY, "acceptance is established and holding")


def _trend_phase(
    item: dict[str, Any],
    pa: dict[str, Any],
    range_posture: dict[str, Any],
) -> dict[str, str]:
    score = int(item.get("score") or 0)
    mom15 = abs(_f(pa.get("mom15")))
    if score < 50:
        return _phase_meta(INACTIVE, "trend alignment is not active yet")
    if (
        score >= 78
        and bool(range_posture.get("is_emerging_from_value"))
        and mom15 >= 0.15
    ):
        return _phase_meta(HEAD, "trend thrust is emerging with room to expand")
    if score >= 58 and (
        bool(range_posture.get("is_trend_tail_displacement")) or mom15 <= 0.05
    ):
        return _phase_meta(
            TAIL,
            "trend still points the same way, but extension or fade risk is rising",
        )
    return _phase_meta(BODY, "trend alignment is in force")


def _location_phase(
    item: dict[str, Any],
    range_posture: dict[str, Any],
) -> dict[str, str]:
    score = int(item.get("score") or 0)
    reason = str(item.get("reason") or "").lower()
    if score < 55:
        return _phase_meta(
            INACTIVE, "location is not offering a clean decision edge yet"
        )
    if "at decision level" in reason:
        return _phase_meta(HEAD, "price is testing the decision area right now")
    if bool(range_posture.get("is_terminal_extreme")) or bool(
        range_posture.get("is_stretched_from_value")
    ):
        return _phase_meta(
            TAIL, "location edge is getting stretched away from the decision area"
        )
    return _phase_meta(BODY, "location edge is usable and in force")


def _volume_phase(item: dict[str, Any], pa: dict[str, Any]) -> dict[str, str]:
    score = int(item.get("score") or 0)
    vol_mult = _f(pa.get("vol_mult"))
    if score >= 80 and vol_mult >= 3.5:
        return _phase_meta(
            TAIL, "participation is climactic; late chase risk is rising"
        )
    if score >= 80:
        return _phase_meta(HEAD, "participation is arriving with the move")
    if score >= 60:
        return _phase_meta(BODY, "participation is supporting the move")
    if vol_mult < 0.9:
        return _phase_meta(TAIL, "participation has fallen away from the move")
    return _phase_meta(INACTIVE, "volume is not adding conviction yet")


def _time_of_day_phase(item: dict[str, Any]) -> dict[str, str]:
    score = int(item.get("score") or 0)
    reason = str(item.get("reason") or "").lower()
    if score < 50:
        return _phase_meta(INACTIVE, "session timing is not helping right now")
    if "opening auction" in reason or "morning continuation" in reason:
        return _phase_meta(HEAD, "the session window is just opening or still fresh")
    if "power hour" in reason:
        return _phase_meta(
            TAIL, "late-session positioning can overpower clean execution"
        )
    if "midday chop" in reason:
        return _phase_meta(TAIL, "the clean window has faded into midday churn")
    return _phase_meta(BODY, "session timing is supportive but not especially fresh")


def _dealer_gamma_phase(
    item: dict[str, Any], pa: dict[str, Any], gp: dict[str, Any]
) -> dict[str, str]:
    score = int(item.get("score") or 0)
    regime = str(gp.get("regime") or "").lower()
    pin_dist = _pin_distance_pct(pa, gp)
    if regime == "negative" and score >= 70:
        return _phase_meta(HEAD, "negative gamma leaves room for expansion")
    if regime == "positive" and pin_dist is not None and pin_dist <= 0.15:
        return _phase_meta(
            TAIL, "pin gravity is pulling the move back toward the magnet"
        )
    if score >= 55:
        return _phase_meta(
            BODY, "dealer positioning context is active but not at an extreme"
        )
    return _phase_meta(INACTIVE, "dealer positioning is not adding much edge right now")


_PHASE_RESOLVERS = {
    "structure_score": _structure_phase,
    "acceptance_score": _acceptance_phase,
    "trend_score": _trend_phase,
    "location_score": _location_phase,
    "volume_score": _volume_phase,
    "time_of_day_score": _time_of_day_phase,
    "dealer_gamma_score": _dealer_gamma_phase,
}


def annotate_spine_score_phases(
    scores: dict[str, dict[str, Any]],
    pa: dict[str, Any],
    op: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
    market_day: dict[str, Any] | None = None,
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return score rows annotated with head/body/tail lifecycle metadata."""
    annotated = {name: dict(item) for name, item in (scores or {}).items()}
    del op, market_day  # reserved for future doctrine; do not fork logic yet.
    gp = gp or {}
    range_posture = build_range_posture(pa)
    for name in CORE_EXECUTION_SPINE_PART_NAMES:
        item = annotated.get(name)
        if not item:
            continue
        resolver = _PHASE_RESOLVERS[name]
        if name == "acceptance_score":
            item.update(resolver(item, range_posture, setups))
        elif name == "dealer_gamma_score":
            item.update(resolver(item, pa, gp))
        elif name == "structure_score":
            item.update(resolver(item, range_posture))
        elif name == "trend_score":
            item.update(resolver(item, pa, range_posture))
        elif name == "location_score":
            item.update(resolver(item, range_posture))
        elif name == "volume_score":
            item.update(resolver(item, pa))
        else:
            item.update(resolver(item))
    return annotated


__all__ = [
    "BODY",
    "HEAD",
    "INACTIVE",
    "TAIL",
    "annotate_spine_score_phases",
]
