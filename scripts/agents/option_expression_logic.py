from __future__ import annotations

from typing import Any



def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def _classify_gamma_twitchiness(gamma_share_change_per_point: float) -> str:
    magnitude = abs(gamma_share_change_per_point)
    if magnitude >= 15:
        return "high"
    if magnitude >= 7:
        return "moderate"
    return "low"



def _classify_theta_pressure(theta_dollars_per_day: float, debit_dollars: float) -> str:
    if theta_dollars_per_day >= 0 or debit_dollars <= 0:
        return "none"
    burn = abs(theta_dollars_per_day) / debit_dollars
    if burn >= 0.08:
        return "heavy"
    if burn >= 0.03:
        return "moderate"
    return "light"



def _classify_vega_posture(vega_dollars_per_5iv: float) -> str:
    magnitude = abs(vega_dollars_per_5iv)
    if magnitude >= 40:
        return "meaningful"
    if magnitude >= 15:
        return "present"
    return "light"



def _estimate_delta(leg: dict[str, Any], geometry: dict[str, Any], branch: dict[str, Any]) -> float:
    spot = _as_float(geometry.get("spot"))
    strike = _as_float(leg.get("strike"))
    if spot <= 0 or strike <= 0:
        return 0.0
    option_type = str(leg.get("option_type") or "").lower()
    if not option_type:
        option_type = "call" if str(branch.get("direction") or "").upper() == "CALLS" else "put"
    rel = abs((strike - spot) / spot)
    if rel <= 0.0015:
        base = 0.5
    elif rel <= 0.004:
        base = 0.4
    elif rel <= 0.01:
        base = 0.3
    else:
        base = 0.15
    in_the_money = strike <= spot if option_type == "call" else strike >= spot
    if in_the_money:
        base = min(base + 0.15, 0.85)
    return base if option_type == "call" else -base



def _estimate_gamma(leg: dict[str, Any], geometry: dict[str, Any]) -> float:
    spot = _as_float(geometry.get("spot"))
    strike = _as_float(leg.get("strike"))
    if spot <= 0 or strike <= 0:
        return 0.0
    rel = abs((strike - spot) / spot)
    if rel <= 0.0015:
        return 0.08
    if rel <= 0.004:
        return 0.05
    return 0.02



def _estimate_theta(leg: dict[str, Any]) -> float:
    midpoint = _as_float(leg.get("midpoint"))
    if midpoint <= 0:
        return 0.0
    return -max(round(midpoint * 0.035, 3), 0.02)



def _estimate_vega(leg: dict[str, Any]) -> float:
    midpoint = _as_float(leg.get("midpoint"))
    if midpoint <= 0:
        return 0.0
    return max(round(midpoint * 0.025, 3), 0.01)



def _leg_metric(
    leg: dict[str, Any], metric: str, geometry: dict[str, Any], branch: dict[str, Any]
) -> tuple[float, bool]:
    raw = leg.get(metric)
    if raw not in (None, ""):
        return _as_float(raw), False
    if metric == "delta":
        return _estimate_delta(leg, geometry, branch), True
    if metric == "gamma":
        return _estimate_gamma(leg, geometry), True
    if metric == "theta":
        return _estimate_theta(leg), True
    if metric == "vega":
        return _estimate_vega(leg), True
    return 0.0, True


def _plan_greeks_source(legs: list[dict[str, Any]], used_heuristics: bool) -> str:
    if used_heuristics:
        return "heuristic"
    leg_sources = {str(leg.get("greeks_source") or "").strip() for leg in legs}
    leg_sources.discard("")
    if not leg_sources:
        return "observed"
    if len(leg_sources) == 1:
        return next(iter(leg_sources))
    return "mixed"


def build_branch_greek_dollar_plan(
    branch: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any] | None:
    legs = branch.get("legs") or []
    if not legs:
        return None

    net_delta_shares = 0.0
    net_gamma_share_change_per_point = 0.0
    theta_dollars_per_day = 0.0
    vega_dollars_per_1iv = 0.0
    used_heuristics = False
    for leg in legs:
        sign = 1.0 if str(leg.get("side") or "buy").lower() == "buy" else -1.0
        delta, delta_heuristic = _leg_metric(leg, "delta", geometry, branch)
        gamma, gamma_heuristic = _leg_metric(leg, "gamma", geometry, branch)
        theta, theta_heuristic = _leg_metric(leg, "theta", geometry, branch)
        vega, vega_heuristic = _leg_metric(leg, "vega", geometry, branch)
        used_heuristics = used_heuristics or any(
            [delta_heuristic, gamma_heuristic, theta_heuristic, vega_heuristic]
        )
        net_delta_shares += sign * delta * 100.0
        net_gamma_share_change_per_point += sign * gamma * 100.0
        theta_dollars_per_day += sign * theta * 100.0
        vega_dollars_per_1iv += sign * vega * 100.0

    pricing = branch.get("pricing") or {}
    debit_dollars = round(_as_float(pricing.get("debit")) * 100.0, 2)
    max_loss_dollars = round(_as_float(pricing.get("max_loss")) * 100.0, 2)
    max_gain_value = pricing.get("max_gain")
    max_gain_dollars = (
        round(_as_float(max_gain_value) * 100.0, 2)
        if max_gain_value not in (None, "")
        else None
    )
    vega_dollars_per_5iv = round(vega_dollars_per_1iv * 5.0, 2)
    one_r_move = (
        round(max_loss_dollars / abs(net_delta_shares), 2)
        if abs(net_delta_shares) >= 1.0 and max_loss_dollars > 0
        else None
    )
    theta_days_to_25pct_decay = (
        round((debit_dollars * 0.25) / abs(theta_dollars_per_day), 1)
        if theta_dollars_per_day < 0 and debit_dollars > 0
        else None
    )
    levels = branch.get("levels") or {}
    trigger_level = _as_float(levels.get("trigger_level"))
    spot = _as_float(geometry.get("spot"))
    invalidation_level = _as_float(levels.get("invalidation_level"))
    return {
        "greeks_source": _plan_greeks_source(legs, used_heuristics),
        "net_delta_shares": round(net_delta_shares, 1),
        "net_gamma_share_change_per_1pt": round(net_gamma_share_change_per_point, 1),
        "theta_dollars_per_day": round(theta_dollars_per_day, 2),
        "vega_dollars_per_1iv": round(vega_dollars_per_1iv, 2),
        "vega_dollars_per_5iv": vega_dollars_per_5iv,
        "approx_pnl_if_underlying_up_1": round(net_delta_shares, 2),
        "approx_pnl_if_underlying_down_1": round(-net_delta_shares, 2),
        "approx_stock_move_for_1r": one_r_move,
        "entry_debit_dollars": debit_dollars,
        "max_loss_dollars": max_loss_dollars,
        "max_gain_dollars": max_gain_dollars,
        "theta_days_to_25pct_decay": theta_days_to_25pct_decay,
        "iv_up_5pt_pnl": vega_dollars_per_5iv,
        "iv_down_5pt_pnl": round(-vega_dollars_per_5iv, 2),
        "gamma_twitchiness": _classify_gamma_twitchiness(net_gamma_share_change_per_point),
        "theta_pressure": _classify_theta_pressure(theta_dollars_per_day, debit_dollars),
        "vega_posture": _classify_vega_posture(vega_dollars_per_5iv),
        "delta_interpretation": (
            f"acts like {round(net_delta_shares, 1):g} shares; roughly "
            f"${round(net_delta_shares, 1):g} per $1 underlying move before gamma"
        ),
        "stock_move_to_trigger": round(trigger_level - spot, 2) if trigger_level and spot else None,
        "trigger_to_invalidation_move": (
            round(invalidation_level - trigger_level, 2)
            if trigger_level and invalidation_level
            else None
        ),
    }


__all__ = ["build_branch_greek_dollar_plan"]
