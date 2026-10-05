"""Target selection and reachability planning for SharpEdge.

`infer_target()` produces a TargetPlan-style protocol: objective, direction,
strategic destination, reachable destination, reachability assessment, and
explanation. It estimates travel. It does not authorize execution.

Doctrine: reachable does not mean executable.
"""

from __future__ import annotations

from typing import Any

from gate_workflows import primary_trade_setup


def _valid_targets(
    spot: float, candidates: list[tuple[str, Any]], direction: str | None = None
) -> list[tuple[str, float]]:
    out = []
    for label, price in candidates:
        if not isinstance(price, (int, float)):
            continue
        if direction == "up" and price <= spot:
            continue
        if direction == "down" and price >= spot:
            continue
        out.append((label, float(price)))
    return out


def _target_stats(spot: float, expected: Any, price: Any) -> dict[str, Any]:
    if not isinstance(expected, (int, float)) or not isinstance(price, (int, float)):
        return {"distance": None, "coverage_ratio": None, "status": "unknown"}
    distance = abs(price - spot)
    coverage_ratio = distance / max(expected, 1e-9)
    if coverage_ratio <= 1.0:
        status = "within"
    elif coverage_ratio <= 1.35:
        status = "stretch"
    else:
        status = "beyond"
    return {
        "distance": round(distance, 2),
        "coverage_ratio": round(coverage_ratio, 2),
        "status": status,
    }


def _target_lists(
    spot: float,
    bias: str,
    setup_tag: str,
    setup_bias: str,
    vwap: Any,
    pin: Any,
    call_wall: Any,
    put_wall: Any,
    ch_hi: Any,
    ch_lo: Any,
    exp_high: Any,
    exp_low: Any,
) -> tuple[str, str, str | None, list[tuple[str, Any]], list[tuple[str, Any]]]:
    if setup_tag == "FAILED BREAKDOWN":
        return (
            "mean_reversion_reclaim",
            "failed-breakdown reclaim targets mean reversion first, then overhead magnets/resistance",
            "up",
            [("VWAP", vwap), ("Magnet", pin), ("Call wall", call_wall)],
            [("VWAP", vwap), ("Magnet", pin), ("Call wall", call_wall)],
        )
    if setup_tag == "FAILED BREAKOUT":
        return (
            "mean_reversion_reject",
            "failed-breakout rejection targets mean reversion first, then downside magnets/support",
            "down",
            [("VWAP", vwap), ("Magnet", pin), ("Put wall", put_wall)],
            [("VWAP", vwap), ("Magnet", pin), ("Put wall", put_wall)],
        )
    if setup_tag in {"DOWNSIDE EXHAUSTION", "UPSIDE EXHAUSTION"}:
        direction = "up" if setup_tag == "DOWNSIDE EXHAUSTION" else "down"
        walls = (
            [("Call wall", call_wall)]
            if direction == "up"
            else [("Put wall", put_wall)]
        )
        return (
            "exhaustion_fade",
            "exhaustion setups fade back toward value before reaching outer walls",
            direction,
            [("VWAP", vwap), ("Magnet", pin), *walls],
            [("VWAP", vwap), ("Magnet", pin), *walls],
        )
    if setup_tag == "EXHAUSTION -> RUNNER HANDOFF":
        return (
            "handoff_continuation",
            "downside exhaustion has already reclaimed value; manage for continuation, not a mere VWAP fade",
            "up",
            [("Channel hi", ch_hi), ("Exp high", exp_high), ("Call wall", call_wall)],
            [("Channel hi", ch_hi), ("Exp high", exp_high), ("Call wall", call_wall)],
        )
    if setup_tag == "STICKY DAY (CALM/CHOP)" or "SNAP-BACK TO THE MAGNET" in setup_bias:
        direction = (
            "up"
            if isinstance(pin, (int, float)) and pin > spot
            else "down"
            if isinstance(pin, (int, float)) and pin < spot
            else None
        )
        partials = (
            [
                ("VWAP", vwap),
                ("Channel hi", ch_hi),
                ("Exp high", exp_high),
                ("Magnet", pin),
            ]
            if direction != "down"
            else [
                ("VWAP", vwap),
                ("Channel lo", ch_lo),
                ("Exp low", exp_low),
                ("Magnet", pin),
            ]
        )
        return (
            "mean_reversion",
            "range-fade / sticky-day context uses the magnet as the strategic objective",
            direction,
            [("Magnet", pin), ("VWAP", vwap)],
            partials,
        )
    if setup_tag in {"RUNNER DAY (WHEEE)", "POST-SELLOFF COIL"}:
        direction = "down" if (bias == "PUTS" or "BEARISH" in setup_bias) else "up"
        candidates = (
            [("Channel lo", ch_lo), ("Exp low", exp_low), ("Put wall", put_wall)]
            if direction == "down"
            else [
                ("Channel hi", ch_hi),
                ("Exp high", exp_high),
                ("Call wall", call_wall),
            ]
        )
        return (
            "continuation",
            "continuation setup targets directional expansion, not an automatic magnet snap-back",
            direction,
            candidates,
            candidates,
        )
    if bias == "CALLS":
        return (
            "bullish_fallback",
            "fallback bullish target uses the nearest overhead objective",
            "up",
            [("Call wall", call_wall), ("Magnet", pin), ("Exp high", exp_high)],
            [("Exp high", exp_high), ("Call wall", call_wall), ("Magnet", pin)],
        )
    if bias == "PUTS":
        return (
            "bearish_fallback",
            "fallback bearish target uses the nearest downside objective",
            "down",
            [("Put wall", put_wall), ("Magnet", pin), ("Exp low", exp_low)],
            [("Exp low", exp_low), ("Put wall", put_wall), ("Magnet", pin)],
        )
    return (
        "neutral_fallback",
        "neutral / mixed context defaults to the nearest value anchor",
        None,
        [("Magnet", pin), ("VWAP", vwap)],
        [("VWAP", vwap), ("Magnet", pin)],
    )


def infer_target(
    pa: dict[str, Any],
    op: dict[str, Any],
    permission: dict[str, Any],
    gp: dict[str, Any],
    micro: dict[str, Any] | None = None,
    magnitude: dict[str, Any] | None = None,
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    micro = micro or {}
    magnitude = magnitude or {}
    spot = pa.get("spot")
    bias = str(permission.get("bias", "NEUTRAL")).upper()
    setup = primary_trade_setup(setups)
    setup_tag = str(setup.get("tag", "")).upper()
    setup_bias = str(setup.get("bias", "")).upper()
    pin = (gp or {}).get("pin")
    vwap = pa.get("vwap")
    exp_move = magnitude.get("exp_move_realized_usd")
    exp_high = (
        spot + exp_move
        if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float))
        else None
    )
    exp_low = (
        spot - exp_move
        if isinstance(spot, (int, float)) and isinstance(exp_move, (int, float))
        else None
    )
    if not isinstance(spot, (int, float)):
        return {"label": "No target", "price": None, "reason": "spot unavailable"}

    objective, reason, direction, strategic_raw, reachable_raw = _target_lists(
        spot,
        bias,
        setup_tag,
        setup_bias,
        vwap,
        pin,
        op.get("call_wall"),
        op.get("put_wall"),
        micro.get("ch_hi"),
        micro.get("ch_lo"),
        exp_high,
        exp_low,
    )
    strategic_targets = _valid_targets(spot, strategic_raw, direction)
    if not strategic_targets:
        return {
            "label": "No target",
            "price": None,
            "reason": "no clean target level available",
            "setup_tag": setup_tag,
            "objective": objective,
            "reachable_today": {},
        }
    label, price = strategic_targets[0]
    expected = magnitude.get("exp_move_realized_usd")
    strategic_stats = _target_stats(spot, expected, price)
    reachable_today = {}
    for r_label, r_price in _valid_targets(spot, reachable_raw, direction):
        stats = _target_stats(spot, expected, r_price)
        if stats["status"] in {"within", "stretch"}:
            reachable_today = {"label": r_label, "price": r_price, **stats}
            break
    likely_travel = ""
    if reachable_today and reachable_today["label"] != label:
        likely_travel = (
            "partial reversion only"
            if objective.startswith("mean_reversion") or objective == "exhaustion_fade"
            else "first expansion objective only"
        )
    elif strategic_stats["status"] == "beyond":
        likely_travel = "today probably reaches only an intermediate objective"
    return {
        "label": label,
        "price": price,
        "reason": reason,
        "setup_tag": setup_tag,
        "objective": objective,
        **strategic_stats,
        "reachable_today": reachable_today,
        "likely_travel": likely_travel,
    }


def reachability_context(
    pa: dict[str, Any],
    op: dict[str, Any],
    permission: dict[str, Any],
    magnitude: dict[str, Any],
    gp: dict[str, Any],
    micro: dict[str, Any] | None = None,
    setups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = magnitude.get("exp_move_realized_usd")
    target = infer_target(pa, op, permission, gp, micro, magnitude, setups)
    if target.get("status") == "unknown":
        return {
            "target_label": target.get("label", "No target"),
            "target_price": target.get("price"),
            "distance": target.get("distance"),
            "expected_move": expected,
            "coverage_ratio": target.get("coverage_ratio"),
            "status": "unknown",
            "reason": "expected-move context unavailable",
            "reachable_today": target.get("reachable_today", {}),
            "likely_travel": target.get("likely_travel", ""),
        }
    reason = {
        "within": "target is reachable inside the remaining realized-vol move",
        "stretch": "target is a stretch; needs follow-through, not just noise",
        "beyond": "target sits beyond the remaining expected move",
    }.get(target.get("status"), "expected-move context unavailable")
    return {
        "target_label": target.get("label", "No target"),
        "target_price": target.get("price"),
        "distance": target.get("distance"),
        "expected_move": round(expected, 2)
        if isinstance(expected, (int, float))
        else expected,
        "coverage_ratio": target.get("coverage_ratio"),
        "status": target.get("status", "unknown"),
        "reason": f"{reason} ({target.get('reason', '')})".strip(),
        "reachable_today": target.get("reachable_today", {}),
        "likely_travel": target.get("likely_travel", ""),
    }


__all__ = ["infer_target", "reachability_context"]
