"""BehaviorAnnotation producer for the SharpEdge cockpit.

This module names structured observations about interactions between existing
protocols: evidence, market bucket, TargetPlan, readiness, and permission. It is
diagnostic commentary, not authority. It does not score, gate, authorize, mint
tokens, execute tokens, or mutate trade logic.
"""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture

SCHEMA = "sharpedge.regime_refinement.v1"
TOKEN_ELIGIBLE_LABELS = {
    "confirmed_magnet_fade",
    "trap_candidate_waiting_confirmation",
    "upper_edge_exhaustion_watch",
}


def _score(permission: dict[str, Any], name: str) -> int | None:
    scores = permission.get("scores") or {}
    item = scores.get(name) or scores.get(f"{name}_score") or {}
    value = item.get("score")
    return int(value) if isinstance(value, (int, float)) else None


def _reason(permission: dict[str, Any], name: str) -> str:
    scores = permission.get("scores") or {}
    item = scores.get(name) or scores.get(f"{name}_score") or {}
    return str(item.get("reason") or "")


def _setup_tag(setups: list[dict[str, Any]]) -> str:
    for setup in setups:
        tag = str(setup.get("tag") or "")
        if tag:
            return tag
    return ""


def _target_distance(target_plan: dict[str, Any]) -> float | None:
    value = target_plan.get("distance")
    return float(value) if isinstance(value, (int, float)) else None


def _remaining_move(
    target_plan: dict[str, Any], magnitude: dict[str, Any]
) -> float | None:
    reachable = target_plan.get("reachable_today") or {}
    value = reachable.get("remaining_expected_move")
    if isinstance(value, (int, float)):
        return float(value)
    value = magnitude.get("exp_move_realized_usd")
    return float(value) if isinstance(value, (int, float)) else None


def _annotation(
    *,
    bucket: str,
    label: str,
    behavior: str,
    evidence: list[str],
    confirms: list[str] | None = None,
    invalidates: list[str] | None = None,
    token_eligible: bool = False,
    severity: str = "info",
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "label": label,
        "behavior": behavior,
        "evidence": [item for item in evidence if item],
        "confirms": confirms or [],
        "invalidates": invalidates or [],
        "token_eligible": bool(token_eligible),
        "severity": severity,
    }


def _core_spine_annotations(
    pa: dict[str, Any],
    permission: dict[str, Any],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    acceptance = _score(permission, "acceptance_score") or 0
    trend = _score(permission, "trend_score") or 0
    volume = _score(permission, "volume_score") or 0
    regime = _score(permission, "regime_score") or 0
    posture = build_range_posture(pa)

    if acceptance >= 70 and trend >= 70 and volume >= 60:
        side = "upper" if str(posture.get("side")) == "upside" else "lower"
        annotations.append(
            _annotation(
                bucket="core_execution_spine",
                label=f"{side}_rail_acceptance",
                behavior="Auction is accepting away from balance with trend/volume support.",
                evidence=[
                    f"acceptance_score={acceptance}: {_reason(permission, 'acceptance_score')}",
                    f"trend_score={trend}: {_reason(permission, 'trend_score')}",
                    f"volume_score={volume}: {_reason(permission, 'volume_score')}",
                ],
                confirms=["continued closes beyond the accepted level"],
                invalidates=["failed acceptance back through the level"],
            )
        )

    if regime >= 75 and trend >= 70 and bool(posture.get("is_extreme")):
        direction = (
            "late_extension_up"
            if str(posture.get("side")) == "upside"
            else "late_extension_down"
        )
        annotations.append(
            _annotation(
                bucket="core_execution_spine",
                label="trend_day_late_extension",
                behavior=f"Trend-day structure is pushing into a range extreme ({direction}).",
                evidence=[
                    f"regime_score={regime}: {_reason(permission, 'regime_score')}",
                    f"trend_score={trend}: {_reason(permission, 'trend_score')}",
                    f"range_state={posture.get('range_state')}",
                ],
                confirms=["acceptance remains stacked with no trap/rejection"],
                invalidates=["rejection/trap confirms at the range edge"],
                severity="watch",
            )
        )
    return annotations


def _secondary_annotations(permission: dict[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    rejection = _score(permission, "rejection_score") or 0
    trap = _score(permission, "trap_score") or 0
    if trap >= 60:
        annotations.append(
            _annotation(
                bucket="secondary_confirmations",
                label="trap_candidate_waiting_confirmation",
                behavior="Failed-break/trap evidence is present but must persist before token entry.",
                evidence=[
                    f"trap_score={trap}: {_reason(permission, 'trap_score')}",
                    f"rejection_score={rejection}: {_reason(permission, 'rejection_score')}",
                ],
                confirms=["setup lifecycle reaches confirmed status"],
                invalidates=["price re-accepts beyond the failed-break level"],
                token_eligible=True,
                severity="watch",
            )
        )

    if rejection >= 65:
        annotations.append(
            _annotation(
                bucket="secondary_confirmations",
                label="confirmed_rejection_response",
                behavior="Rejection evidence says the edge is rejecting price.",
                evidence=[
                    f"rejection_score={rejection}: {_reason(permission, 'rejection_score')}",
                ],
                confirms=["next bar follows through away from rejected prices"],
                invalidates=["price re-accepts through the rejected edge"],
                token_eligible=True,
                severity="watch",
            )
        )
    return annotations


def _context_annotations(
    pa: dict[str, Any],
    gp: dict[str, Any],
    permission: dict[str, Any],
    target_plan: dict[str, Any],
    magnitude: dict[str, Any],
    setups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    setup_tag = _setup_tag(setups).upper()
    gamma_regime = str(gp.get("regime") or "").lower()
    posture = build_range_posture(pa)
    acceptance = _score(permission, "acceptance_score") or 0
    trend = _score(permission, "trend_score") or 0
    rejection = _score(permission, "rejection_score") or 0
    trap = _score(permission, "trap_score") or 0
    balance = pa.get("balance_confluence") or {}
    target_distance = _target_distance(target_plan)
    remaining_move = _remaining_move(target_plan, magnitude)

    if "STICKY DAY" in setup_tag and acceptance >= 70 and trend >= 70:
        annotations.append(
            _annotation(
                bucket="context_governors",
                label="magnet_fade_denied_by_acceptance",
                behavior="Sticky-day magnet pull exists, but acceptance/trend says the fade is not confirmed yet.",
                evidence=[
                    f"setup_tag={_setup_tag(setups)}",
                    f"acceptance_score={acceptance}: {_reason(permission, 'acceptance_score')}",
                    f"trend_score={trend}: {_reason(permission, 'trend_score')}",
                ],
                confirms=["rejection_score or trap_score confirms the fade"],
                invalidates=["continued accepted closes away from the magnet"],
                severity="guardrail",
            )
        )

    if (
        gamma_regime == "positive"
        and bool(posture.get("is_terminal_extreme"))
        and str(posture.get("side")) == "upside"
    ):
        annotations.append(
            _annotation(
                bucket="context_governors",
                label="sticky_upper_rail_drift",
                behavior="Positive-gamma/sticky tape is drifting along the upper rail instead of cleanly fading.",
                evidence=[
                    "gamma_regime=positive",
                    f"range_state={posture.get('range_state')}",
                    f"rejection_score={rejection}",
                    f"trap_score={trap}",
                ],
                confirms=["rejection/trap rises and price loses upper-rail acceptance"],
                invalidates=["range expands through the upper rail with volume"],
                severity="watch",
            )
        )

    if target_distance is not None and remaining_move is not None:
        if target_distance > remaining_move:
            annotations.append(
                _annotation(
                    bucket="context_governors",
                    label="magnet_target_unreachable_today",
                    behavior="Strategic target is farther than the remaining expected move; partial travel is more realistic.",
                    evidence=[
                        f"target_distance={target_distance:.2f}",
                        f"remaining_expected_move={remaining_move:.2f}",
                    ],
                    confirms=[
                        "expected move expands or price accelerates toward target"
                    ],
                    invalidates=[
                        "target becomes reachable inside remaining expected move"
                    ],
                    severity="guardrail",
                )
            )

    if balance.get("state") == "disagreement" or (
        pa.get("balance_disagreement") or {}
    ).get("has_disagreement"):
        annotations.append(
            _annotation(
                bucket="context_governors",
                label="balance_model_disagreement",
                behavior="Balance lenses disagree; conviction should stay dampened until one lens wins.",
                evidence=[
                    str(balance.get("reason") or ""),
                    str((pa.get("balance_disagreement") or {}).get("reason") or ""),
                ],
                confirms=["opening/recent/value balance lenses align"],
                invalidates=["continued disagreement across balance lenses"],
                severity="guardrail",
            )
        )
    return annotations


def _suspect_drift_annotations(
    pa: dict[str, Any],
    permission: dict[str, Any],
    setups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    pressure = _score(permission, "pressure_score") or 0
    regime = _score(permission, "regime_score") or 0
    acceptance = _score(permission, "acceptance_score") or 0
    rejection = _score(permission, "rejection_score") or 0
    trap = _score(permission, "trap_score") or 0
    posture = build_range_posture(pa)
    setup_tag = _setup_tag(setups).upper()

    if regime >= 75 and pressure <= 50 and acceptance < 70:
        annotations.append(
            _annotation(
                bucket="suspect_drift_voices",
                label="trend_score_overstating_edge",
                behavior="Regime/trend looks strong, but pressure or acceptance is not proving the edge.",
                evidence=[
                    f"regime_score={regime}: {_reason(permission, 'regime_score')}",
                    f"pressure_score={pressure}: {_reason(permission, 'pressure_score')}",
                    f"acceptance_score={acceptance}: {_reason(permission, 'acceptance_score')}",
                ],
                confirms=["acceptance and pressure join the move"],
                invalidates=["price rotates back into balance"],
                severity="guardrail",
            )
        )

    if "STICKY DAY" in setup_tag and regime >= 75:
        annotations.append(
            _annotation(
                bucket="suspect_drift_voices",
                label="sticky_trend_conflict",
                behavior="Sticky-day gamma label and trend-day regime score are both active; treat the tape as mixed-mode.",
                evidence=[
                    f"setup_tag={_setup_tag(setups)}",
                    f"regime_score={regime}: {_reason(permission, 'regime_score')}",
                ],
                confirms=[
                    "one side wins: either rejection/trap fade or accepted expansion"
                ],
                invalidates=["regime score cools back into balance"],
                severity="watch",
            )
        )

    if (
        bool(posture.get("is_terminal_extreme"))
        and str(posture.get("side")) == "upside"
        and rejection < 60
        and trap < 60
    ):
        annotations.append(
            _annotation(
                bucket="suspect_drift_voices",
                label="upper_edge_exhaustion_watch",
                behavior="Price is at/near highs, but rejection/trap confirmation is not strong enough for a fade token.",
                evidence=[
                    f"range_state={posture.get('range_state')}",
                    f"rejection_score={rejection}",
                    f"trap_score={trap}",
                ],
                confirms=["rejection_score or trap_score rises into confirmation"],
                invalidates=["accepted breakout continuation through the high"],
                token_eligible=True,
                severity="watch",
            )
        )
    return annotations


def annotate_market_behavior(
    pa: dict[str, Any] | None = None,
    op: dict[str, Any] | None = None,
    gp: dict[str, Any] | None = None,
    permission: dict[str, Any] | None = None,
    target_plan: dict[str, Any] | None = None,
    magnitude: dict[str, Any] | None = None,
    setups: list[dict[str, Any]] | None = None,
    edge_token_position: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return behavior annotations without changing any execution decision."""

    pa = pa or {}
    _op = op or {}
    gp = gp or {}
    permission = permission or {}
    target_plan = target_plan or {}
    magnitude = magnitude or {}
    setups = setups or []
    edge_token_position = edge_token_position or {}

    annotations = [
        *_core_spine_annotations(pa, permission),
        *_secondary_annotations(permission),
        *_context_annotations(pa, gp, permission, target_plan, magnitude, setups),
        *_suspect_drift_annotations(pa, permission, setups),
    ]
    buckets = {
        "core_execution_spine": [],
        "secondary_confirmations": [],
        "context_governors": [],
        "suspect_drift_voices": [],
    }
    for item in annotations:
        buckets.setdefault(item["bucket"], []).append(item)

    primary = annotations[0] if annotations else None
    token_labels = [
        item["label"]
        for item in annotations
        if item.get("token_eligible") or item["label"] in TOKEN_ELIGIBLE_LABELS
    ]
    return {
        "schema": SCHEMA,
        "mode": "pure_annotation_no_permission_change",
        "primary_behavior": primary.get("label") if primary else "unclassified_balance",
        "behavior_summary": primary.get("behavior")
        if primary
        else "No specific refinement label fired; preserve the base cockpit read.",
        "buckets": buckets,
        "annotations": annotations,
        "token_annotation": {
            "position_state": edge_token_position.get("position_state", "flat"),
            "suggested_action": edge_token_position.get(
                "suggested_action", "stand_down"
            ),
            "eligible_behavior_labels": token_labels,
            "note": "Annotations can explain token context but do not mint, approve, or execute tokens.",
        },
    }


__all__ = ["SCHEMA", "annotate_market_behavior"]
