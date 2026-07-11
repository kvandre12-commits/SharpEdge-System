"""Live playbook readiness checks for SharpEdge execution grammar.

Current growth doctrine:
- Direct thesis playbooks stay generic: thesis allowed by bucket -> TRIGGER_MATCH.
- Contextual playbooks get dedicated evaluators when readiness requires extra
  phenomena such as edge proximity, confirmation, or defined risk.

`TriggerResult` is the execution-layer sibling of `ScorePart`: a stable contract
for readiness evidence, not a place to compute vector scores or render UI.
"""

from __future__ import annotations

from typing import Any

from range_posture import build_range_posture

TRIGGER_RESULT_FIELDS = (
    "status",
    "permission_role",
    "matched_playbook",
    "matched_evidence",
    "missing_evidence",
    "reason",
    "needs",
)
TRIGGERED_THESES = {
    "failed_breakout_reversal",
    "failed_breakdown_reclaim",
    "accepted_breakout_runner",
    "accepted_breakdown_runner",
}
PIN_MAGNET_BAND_PCT = 0.25
EDGE_PROXIMITY_PCT = 0.20


def _result(
    *,
    status: str,
    permission_role: str,
    matched_playbook: str | None,
    matched_evidence: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    reason: str,
    needs: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "permission_role": permission_role,
        "matched_playbook": matched_playbook,
        "matched_evidence": matched_evidence or [],
        "missing_evidence": missing_evidence or [],
        "reason": reason,
        "needs": needs or [],
        **extra,
    }


def _pct_distance(a: float | None, b: float | None) -> float | None:
    if not a or not b:
        return None
    return abs(a - b) / a * 100


def _numeric_level_items(levels: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": name, "label": name, "price": float(price), "source": "intraday_prior"}
        for name, price in (levels or {}).items()
        if isinstance(price, (int, float))
    ]


def _context_legend_items(pa: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source_name in ("weekly_context", "monthly_context"):
        context = (pa or {}).get(source_name) or {}
        for item in context.get("legend") or []:
            price = item.get("price")
            name = item.get("name") or item.get("label")
            if isinstance(price, (int, float)) and name:
                items.append(
                    {
                        "name": str(name),
                        "label": str(item.get("label") or name),
                        "price": float(price),
                        "source": source_name,
                    }
                )
    return items


def nearest_edge_level(
    spot: float | None,
    levels: dict[str, Any],
    pa: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(spot, (int, float)) or spot <= 0:
        return None
    candidates = [*_numeric_level_items(levels), *_context_legend_items(pa or {})]
    ranked = []
    for candidate in candidates:
        distance = _pct_distance(float(spot), candidate["price"])
        if distance is None:
            continue
        ranked.append({**candidate, "distance_pct": distance})
    if not ranked:
        return None
    nearest = min(ranked, key=lambda item: item["distance_pct"])
    if nearest["distance_pct"] > EDGE_PROXIMITY_PCT:
        return None
    return nearest


def _magnet_fade_context(
    pa: dict[str, Any],
    dealer: dict[str, Any],
    day_bucket: dict[str, Any] | None,
    levels: dict[str, Any],
) -> dict[str, Any] | None:
    bucket = day_bucket or {}
    allowed = set(bucket.get("allowed_playbooks") or [])
    if "magnet_fade" not in allowed:
        return None
    if str(bucket.get("bucket")) != "range_balance_day":
        return None

    spot = (pa or {}).get("spot")
    posture = build_range_posture(pa)
    vs_vwap = float((pa or {}).get("vs_vwap") or 0.0)
    pin_dist = dealer.get("pin_dist_pct")
    near_vwap = bool(posture.get("is_near_value"))
    near_pin = isinstance(pin_dist, (int, float)) and pin_dist <= PIN_MAGNET_BAND_PCT
    positive_gamma = str(dealer.get("regime") or "").lower() == "positive"
    edge = nearest_edge_level(spot, levels, pa)
    matched = ["bucket allows magnet_fade"]
    missing = []
    if positive_gamma:
        matched.append("positive gamma supports mean reversion")
    else:
        missing.append("positive gamma regime")
    if near_vwap:
        matched.append(f"VWAP magnet active ({vs_vwap:+.2f}%)")
    else:
        missing.append("price close enough to VWAP magnet")
    if near_pin:
        matched.append("pin proximity supports gravity")
    else:
        missing.append("pin proximity")
    if edge:
        matched.append(f"near edge {edge['label']} {edge['price']:.2f}")
    elif positive_gamma and near_vwap and near_pin:
        missing.append("nearby support/resistance edge")
        return _result(
            status="WAIT",
            permission_role="missing_edge_context",
            matched_playbook="magnet_fade",
            matched_evidence=matched,
            missing_evidence=missing,
            reason="magnet-fade context has positive gamma, VWAP magnet, and pin proximity, but no nearby support/resistance edge was available",
            needs=[
                "nearby OR/PD support-resistance level or weekly/monthly context-box level",
                "rejection or reclaim confirmation at that edge",
                "defined stop beyond the support/resistance level",
            ],
        )
    else:
        return None

    if not (positive_gamma and near_vwap and near_pin and edge):
        return None

    return _result(
        status="CONTEXT_MATCH",
        permission_role="weighting_context",
        matched_playbook="magnet_fade",
        matched_evidence=matched,
        missing_evidence=[
            "rejection or reclaim confirmation at the edge",
            "defined stop beyond the support/resistance level",
        ],
        location={
            "edge_name": edge["name"],
            "edge_label": edge["label"],
            "edge_price": edge["price"],
            "edge_source": edge["source"],
            "distance_pct": round(edge["distance_pct"], 3),
        },
        reason=(
            "range/balance bucket with positive gamma, VWAP magnet "
            f"({vs_vwap:+.2f}%), pin proximity, and edge {edge['label']} "
            f"{edge['price']:.2f} supports magnet-fade context"
        ),
        needs=[
            "rejection or reclaim confirmation at the edge",
            "defined stop beyond the support/resistance level",
            "do not chase away from the edge",
        ],
    )


def live_trigger_check(
    thesis: str,
    day_bucket: dict[str, Any] | None,
    pa: dict[str, Any],
    dealer: dict[str, Any],
    levels: dict[str, Any],
) -> dict[str, Any]:
    bucket = day_bucket or {}
    allowed = set(bucket.get("allowed_playbooks") or [])
    if thesis not in TRIGGERED_THESES:
        magnet_context = _magnet_fade_context(pa, dealer, day_bucket, levels)
        if magnet_context:
            return magnet_context
        return _result(
            status="WAIT",
            permission_role="none",
            matched_playbook=None,
            missing_evidence=["recognized live trigger or context playbook"],
            reason="no live trigger; battlefield classification does not pull the trigger",
        )
    if thesis not in allowed:
        return _result(
            status="WAIT",
            permission_role="blocked_by_bucket",
            matched_playbook=thesis,
            matched_evidence=[f"live trigger detected: {thesis}"],
            missing_evidence=[f"bucket allows {thesis}"],
            reason=f"live trigger {thesis} is not allowed by bucket {bucket.get('bucket', 'unknown')}",
            needs=["day bucket that explicitly allows this playbook"],
        )
    return _result(
        status="TRIGGER_MATCH",
        permission_role="execution_trigger",
        matched_playbook=thesis,
        matched_evidence=[
            f"live trigger detected: {thesis}",
            f"bucket allows {thesis}",
        ],
        reason=f"live trigger {thesis} matches bucket {bucket.get('bucket', 'unknown')}",
    )


__all__ = ["TRIGGER_RESULT_FIELDS", "live_trigger_check", "nearest_edge_level"]
