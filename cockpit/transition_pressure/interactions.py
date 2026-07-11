"""Interaction surfaces for transition pressure."""

from __future__ import annotations

from typing import Any


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _surface(name: str, score: float, bias: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "score": _clamp_score(score),
        "bias": bias,
        "reason": reason,
    }


def build_interactions(
    deltas: dict[str, dict[str, Any]],
    potential_energy: dict[str, Any],
) -> dict[str, Any]:
    permission_velocity = int(
        (deltas.get("permission_delta") or {}).get("velocity") or 0
    )
    trend_velocity = int((deltas.get("trend_delta") or {}).get("velocity") or 0)
    acceptance_velocity = int(
        (deltas.get("acceptance_delta") or {}).get("velocity") or 0
    )
    participation_velocity = int(
        (deltas.get("participation_delta") or {}).get("velocity") or 0
    )

    compression = int(
        ((potential_energy.get("compression_score") or {}).get("score") or 0)
    )
    location = potential_energy.get("location_pressure") or {}
    gamma = potential_energy.get("gamma_constraint") or {}

    surfaces = [
        _surface(
            "permission_x_compression",
            28 + (permission_velocity * 2.8) + (compression * 0.42),
            "upside" if permission_velocity > 0 else "unclear",
            "permission change is interacting with stored compression energy",
        ),
        _surface(
            "acceptance_x_compression",
            24 + (acceptance_velocity * 2.6) + (compression * 0.38),
            "upside"
            if acceptance_velocity > 0
            else "downside"
            if acceptance_velocity < 0
            else "unclear",
            "acceptance building inside compression is more meaningful than either alone",
        ),
        _surface(
            "trend_x_gamma",
            20 + (trend_velocity * 2.5) + (int(gamma.get("score") or 0) * 0.33),
            str(gamma.get("bias") or "unclear"),
            "trend change matters more when dealer positioning is constraining or destabilizing travel",
        ),
        _surface(
            "participation_x_location",
            22
            + (participation_velocity * 2.4)
            + (int(location.get("score") or 0) * 0.35),
            str(location.get("bias") or "unclear"),
            "participation matters more when price is already crowding a decision area",
        ),
    ]
    dominant = max(surfaces, key=lambda item: item["score"]) if surfaces else {}
    return {
        "schema": "sharpedge.transition_interactions.v1",
        "surfaces": surfaces,
        "dominant_interaction": dominant,
    }


__all__ = ["build_interactions"]
