"""Market-day bucket brain for SharpEdge.

This classifies the battlefield and constrains legal playbook families. It does
not pull the trigger, determine timing, or replace live readiness evaluation.

Boundary doctrine: if multiple paths enter one bucket, they should describe the
same battlefield. Shared playbooks alone are not enough to prove one bucket.
"""

from __future__ import annotations

from typing import Any

from trade_permission_context import BEARISH, BULLISH
from vwap_posture import build_vwap_posture

TREND_PLAYBOOKS = {"accepted_breakout_runner", "accepted_breakdown_runner"}
FAILED_BREAK_PLAYBOOKS = {"failed_breakout_reversal", "failed_breakdown_reclaim"}
RANGE_PLAYBOOKS = {
    "failed_breakout_reversal",
    "failed_breakdown_reclaim",
    "magnet_fade",
}
NO_TRADE_PLAYBOOKS: set[str] = set()


def _bucket(
    name: str,
    score: int,
    bias: str,
    playbooks: set[str] | list[str],
    risk_posture: str,
    reason: str,
    vwap_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "sharpedge.market_day_bucket.v1",
        "bucket": name,
        "score": score,
        "bias": bias,
        "allowed_playbooks": sorted(playbooks),
        "risk_posture": risk_posture,
        "vwap_context": vwap_context,
        "reason": reason,
    }


def _bias_label(value: int) -> str:
    if value == BULLISH:
        return "CALLS"
    if value == BEARISH:
        return "PUTS"
    return "NEUTRAL"


def _part(parts: dict[str, Any], name: str):
    return parts.get(name)


def _setup_tags(setups: list[dict[str, Any]] | None) -> set[str]:
    return {str(setup.get("tag") or "").upper() for setup in setups or []}


def _score(parts: dict[str, Any], name: str) -> int:
    part = _part(parts, name)
    return int(getattr(part, "score", 0) or 0)


def _bias(parts: dict[str, Any], name: str) -> str:
    part = _part(parts, name)
    return _bias_label(int(getattr(part, "bias", 0) or 0))


def _vwap_context(pa: dict[str, Any]) -> dict[str, Any]:
    return build_vwap_posture(pa)


def _vwap_aligns_with_bias(vwap_context: dict[str, Any], bias: str) -> bool:
    return (bias == "CALLS" and bool(vwap_context.get("has_upside_control"))) or (
        bias == "PUTS" and bool(vwap_context.get("has_downside_control"))
    )


def _range_balance_reason(
    *,
    regime: str,
    balance_score: int,
    near_vwap: bool,
    vwap: dict[str, Any],
) -> str:
    reasons = []
    if regime == "positive":
        reasons.append("positive gamma/OI proxy favors dampening")
    if balance_score <= 45:
        reasons.append("balance context lacks edge confluence")
    if near_vwap:
        reasons.append("spot is near VWAP/value magnet")
    if not reasons:
        reasons.append("battlefield is range-like")
    return (
        f"{', '.join(reasons)}; range/balance bucket caps runner language "
        f"until an explicit edge trigger appears; {vwap['reason']}"
    )


def classify_day_bucket(
    parts: dict[str, Any],
    pa: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the day bucket and allowed playbooks.

    The bucket is a battlefield label. Execution still requires a matching live
    trigger check in execution_grammar.
    """
    tags = _setup_tags(setups)
    regime = str((gp or {}).get("regime") or "unknown").lower()
    atm_iv = float((op or {}).get("atm_iv") or 0)
    volume_score = _score(parts, "volume_score")
    acceptance_score = _score(parts, "acceptance_score")
    trend_score = _score(parts, "trend_score")
    structure_score = _score(parts, "structure_score")
    balance_score = _score(parts, "balance_context_score")
    compression_score = _score(parts, "compression_score")
    trend_bias = _bias(parts, "trend_score")
    acceptance_bias = _bias(parts, "acceptance_score")
    structure_bias = _bias(parts, "structure_score")
    vwap = _vwap_context(pa or {})
    near_vwap = bool(vwap.get("is_range_like"))

    if atm_iv >= 0.32:
        return _bucket(
            "news_vol_shock_day",
            40,
            "NEUTRAL",
            NO_TRADE_PLAYBOOKS,
            "reduce_size_or_no_trade",
            f"ATM IV {atm_iv:.1%} is shock-level; classify first, execute only with explicit operator review; {vwap['reason']}",
            vwap,
        )

    if "FAILED BREAKDOWN" in tags:
        return _bucket(
            "failed_breakdown_long_day",
            82,
            "CALLS",
            ["failed_breakdown_reclaim"],
            "defined_stop_required",
            f"failed-breakdown/reclaim setup is the active battlefield; {vwap['reason']}",
            vwap,
        )

    if "FAILED BREAKOUT" in tags:
        return _bucket(
            "failed_breakout_short_day",
            82,
            "PUTS",
            ["failed_breakout_reversal"],
            "defined_stop_required",
            f"failed-breakout/rejection setup is the active battlefield; {vwap['reason']}",
            vwap,
        )

    if (
        regime == "negative"
        and min(acceptance_score, trend_score, structure_score, volume_score) >= 64
        and len({trend_bias, acceptance_bias, structure_bias} - {"NEUTRAL"}) == 1
    ):
        bias = next(iter({trend_bias, acceptance_bias, structure_bias} - {"NEUTRAL"}))
        if _vwap_aligns_with_bias(vwap, bias):
            return _bucket(
                "a_plus_trend_day",
                88,
                bias,
                TREND_PLAYBOOKS,
                "trend_continuation_only_after_trigger",
                "negative gamma plus aligned structure/acceptance/trend/volume/VWAP classifies a trend battlefield",
                vwap,
            )

    if regime == "positive" or balance_score <= 45 or near_vwap:
        return _bucket(
            "range_balance_day",
            58,
            "NEUTRAL",
            RANGE_PLAYBOOKS,
            "fade_edges_avoid_chasing",
            _range_balance_reason(
                regime=regime,
                balance_score=balance_score,
                near_vwap=near_vwap,
                vwap=vwap,
            ),
            vwap,
        )

    if compression_score <= 45 and volume_score <= 45:
        return _bucket(
            "trap_noise_day",
            35,
            "NEUTRAL",
            NO_TRADE_PLAYBOOKS,
            "stand_down",
            f"tight/noisy tape with weak proof; wait for market to show its hand; {vwap['reason']}",
            vwap,
        )

    return _bucket(
        "unclassified_day",
        45,
        "NEUTRAL",
        FAILED_BREAK_PLAYBOOKS,
        "wait_for_trigger",
        f"battlefield is not clean; only explicit failed-break triggers are allowed; {vwap['reason']}",
        vwap,
    )


__all__ = ["classify_day_bucket"]
