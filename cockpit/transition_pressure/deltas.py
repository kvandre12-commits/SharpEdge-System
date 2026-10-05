"""Delta helpers for transition pressure."""

from __future__ import annotations

from typing import Any

FEATURE_MAP = {
    "trend_delta": "trend_score",
    "acceptance_delta": "acceptance_score",
    "participation_delta": "volume_score",
}


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _status(velocity: int, acceleration: int) -> str:
    if abs(velocity) <= 1 and abs(acceleration) <= 1:
        return "flat"
    if velocity > 0 and acceleration > 1:
        return "accelerating"
    if velocity > 0:
        return "strengthening"
    if velocity < 0 and acceleration < -1:
        return "fading_fast"
    if velocity < 0:
        return "weakening"
    return "flat"


def _metric(
    label: str, current: int | None, previous: int | None, older: int | None
) -> dict[str, Any]:
    if current is None:
        return {
            "label": label,
            "current": None,
            "previous": previous,
            "velocity": 0,
            "acceleration": 0,
            "status": "unavailable",
        }
    if previous is None:
        return {
            "label": label,
            "current": current,
            "previous": None,
            "velocity": 0,
            "acceleration": 0,
            "status": "new",
        }
    velocity = current - previous
    prior_velocity = previous - older if older is not None else 0
    acceleration = velocity - prior_velocity
    return {
        "label": label,
        "current": current,
        "previous": previous,
        "velocity": velocity,
        "acceleration": acceleration,
        "status": _status(velocity, acceleration),
    }


def _feature_value(receipt: dict[str, Any] | None, feature_name: str) -> int | None:
    scores = (receipt or {}).get("feature_scores") or {}
    return _as_int((scores.get(feature_name) or {}).get("score"))


def build_permission_delta(
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    previous = prior_receipts[-1] if prior_receipts else {}
    older = prior_receipts[-2] if len(prior_receipts) >= 2 else {}
    return _metric(
        "Permission",
        _as_int(current_receipt.get("permission")),
        _as_int(previous.get("permission")),
        _as_int(older.get("permission")),
    )


def build_feature_delta(
    key: str,
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_name = FEATURE_MAP[key]
    previous = prior_receipts[-1] if prior_receipts else {}
    older = prior_receipts[-2] if len(prior_receipts) >= 2 else {}
    label = key.replace("_", " ").replace("delta", "").strip().title()
    return _metric(
        label,
        _feature_value(current_receipt, feature_name),
        _feature_value(previous, feature_name),
        _feature_value(older, feature_name),
    )


def build_transition_deltas(
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "permission_delta": build_permission_delta(current_receipt, prior_receipts),
        "trend_delta": build_feature_delta(
            "trend_delta", current_receipt, prior_receipts
        ),
        "acceptance_delta": build_feature_delta(
            "acceptance_delta", current_receipt, prior_receipts
        ),
        "participation_delta": build_feature_delta(
            "participation_delta", current_receipt, prior_receipts
        ),
    }


__all__ = ["build_transition_deltas"]
