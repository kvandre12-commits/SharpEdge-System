from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import mean, median
from typing import Any

try:
    from scripts.nerv.greeks import estimate_option_greeks
except ModuleNotFoundError:  # pragma: no cover
    from nerv.greeks import estimate_option_greeks


TRADING_DAY_HOURS = 6.5
CHECKPOINTS = [
    ("10:30", 1.0),
    ("11:30", 2.0),
    ("12:30", 3.0),
    ("13:30", 4.0),
    ("14:30", 5.0),
    ("15:30", 6.0),
    ("16:00", 6.5),
]
U_SHAPE_SEGMENTS = [
    ("10:30", 1.0, 1.60),
    ("11:30", 1.0, 1.15),
    ("12:30", 1.0, 0.95),
    ("13:30", 1.0, 0.85),
    ("14:30", 1.0, 0.85),
    ("15:30", 1.0, 0.95),
    ("16:00", 0.5, 1.50),
]



def _utc_now() -> str:
    return datetime.now(UTC).isoformat()



def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)



def _favorable_sign(option_type: str) -> int:
    return 1 if str(option_type).lower() == "call" else -1



def _pnl_per_share(delta: float, gamma: float, move: float, theta_carry: float) -> float:
    return delta * move + 0.5 * gamma * (move**2) - theta_carry



def _solve_break_even_move(delta_abs: float, gamma: float, theta_carry: float) -> float | None:
    if theta_carry <= 0:
        return 0.0
    if delta_abs <= 0 and gamma <= 0:
        return None
    if abs(gamma) < 1e-12:
        return theta_carry / delta_abs if delta_abs > 0 else None
    a = 0.5 * gamma
    b = delta_abs
    c = -theta_carry
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    root = (-b + math.sqrt(discriminant)) / (2 * a)
    if root >= 0:
        return root
    alt_root = (-b - math.sqrt(discriminant)) / (2 * a)
    return alt_root if alt_root >= 0 else None



def _sigma_day_dollars(spot: float, iv: float) -> float:
    return spot * iv / math.sqrt(252.0)



def _theta_carry_per_share(theta_per_share_per_day: float, hours: float) -> float:
    return abs(theta_per_share_per_day) * (hours / 24.0)



def _flat_bands(
    spot: float,
    iv: float,
    theta_per_share_per_day: float,
    delta: float,
    gamma: float,
) -> list[dict[str, Any]]:
    sigma_day = _sigma_day_dollars(spot, iv)
    rows: list[dict[str, Any]] = []
    for time_et, elapsed_hours in CHECKPOINTS:
        sigma = sigma_day * math.sqrt(elapsed_hours / TRADING_DAY_HOURS)
        theta_carry = _theta_carry_per_share(theta_per_share_per_day, elapsed_hours)
        be_move = _solve_break_even_move(abs(delta), gamma, theta_carry)
        rows.append(
            {
                "time_et": time_et,
                "theta_carry_contract": _round(theta_carry * 100.0, 2),
                "break_even_move_dollars": _round(be_move, 2),
                "break_even_move_pct": _round((be_move / spot) * 100.0 if be_move else 0.0, 3),
                "band_1sigma_dollars": _round(sigma, 2),
                "band_90_dollars": _round(sigma * 1.645, 2),
                "band_95_dollars": _round(sigma * 1.96, 2),
            }
        )
    return rows



def _u_shape_rows(
    spot: float,
    iv: float,
    theta_per_share_per_day: float,
    delta: float,
    gamma: float,
) -> list[dict[str, Any]]:
    sigma_day = _sigma_day_dollars(spot, iv)
    scalar = TRADING_DAY_HOURS / sum(duration * weight for _, duration, weight in U_SHAPE_SEGMENTS)
    effective_hours = 0.0
    rows: list[dict[str, Any]] = []
    for time_et, duration, weight in U_SHAPE_SEGMENTS:
        effective_hours += duration * weight * scalar
        elapsed_hours = next(hours for label, hours in CHECKPOINTS if label == time_et)
        sigma = sigma_day * math.sqrt(effective_hours / TRADING_DAY_HOURS)
        theta_carry = _theta_carry_per_share(theta_per_share_per_day, elapsed_hours)
        be_move = _solve_break_even_move(abs(delta), gamma, theta_carry)
        plus_pnl = _pnl_per_share(delta, gamma, sigma, theta_carry) * 100.0
        minus_pnl = _pnl_per_share(delta, gamma, -sigma, theta_carry) * 100.0
        sigma_95 = sigma * 1.96
        plus_pnl_95 = _pnl_per_share(delta, gamma, sigma_95, theta_carry) * 100.0
        minus_pnl_95 = _pnl_per_share(delta, gamma, -sigma_95, theta_carry) * 100.0
        rows.append(
            {
                "time_et": time_et,
                "theta_carry_contract": _round(theta_carry * 100.0, 2),
                "break_even_move_dollars": _round(be_move, 2),
                "break_even_move_pct": _round((be_move / spot) * 100.0 if be_move else 0.0, 3),
                "band_1sigma_dollars": _round(sigma, 2),
                "band_90_dollars": _round(sigma * 1.645, 2),
                "band_95_dollars": _round(sigma_95, 2),
                "effective_hours": _round(effective_hours, 3),
                "pnl_plus_1sigma_contract": _round(plus_pnl, 2),
                "pnl_minus_1sigma_contract": _round(minus_pnl, 2),
                "pnl_plus_95_contract": _round(plus_pnl_95, 2),
                "pnl_minus_95_contract": _round(minus_pnl_95, 2),
            }
        )
    return rows



def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight



def _gap_stats(
    gap_rows: list[dict[str, Any]],
    *,
    threshold_pct: float,
    spot: float,
    target_open_weekday: int | None,
    favorable_sign: int,
) -> dict[str, Any]:
    if not gap_rows:
        return {"available": False, "reason": "no_gap_history"}
    all_gaps = [float(row["gap_pct"]) for row in gap_rows if row.get("gap_pct") is not None]
    if not all_gaps:
        return {"available": False, "reason": "no_gap_values"}
    weekday_rows = gap_rows
    if target_open_weekday is not None:
        weekday_rows = [
            row
            for row in gap_rows
            if datetime.fromisoformat(str(row.get("session_date"))[:10]).weekday()
            == target_open_weekday
        ]
    weekday_gaps = [float(row["gap_pct"]) for row in weekday_rows if row.get("gap_pct") is not None]

    def probability(values: list[float], predicate) -> float | None:
        if not values:
            return None
        return round(sum(1 for value in values if predicate(value)) / len(values) * 100.0, 2)

    magnitudes = [abs(value) for value in all_gaps]
    return {
        "available": True,
        "sample_size": len(all_gaps),
        "weekday_sample_size": len(weekday_gaps),
        "threshold_pct": _round(threshold_pct * 100.0, 3),
        "probability_favorable_gap_pct": probability(
            all_gaps,
            lambda value: favorable_sign * value >= threshold_pct,
        ),
        "probability_adverse_gap_pct": probability(
            all_gaps,
            lambda value: favorable_sign * value <= -threshold_pct,
        ),
        "probability_abs_gap_pct": probability(
            all_gaps,
            lambda value: abs(value) >= threshold_pct,
        ),
        "weekday_probability_favorable_gap_pct": probability(
            weekday_gaps,
            lambda value: favorable_sign * value >= threshold_pct,
        ),
        "magnitude_stats": {
            "median_gap_pct": _round(median(magnitudes) * 100.0, 3),
            "median_gap_dollars": _round(median(magnitudes) * spot, 2),
            "p75_gap_pct": _round((_percentile(magnitudes, 0.75) or 0.0) * 100.0, 3),
            "p75_gap_dollars": _round((_percentile(magnitudes, 0.75) or 0.0) * spot, 2),
            "p90_gap_pct": _round((_percentile(magnitudes, 0.90) or 0.0) * 100.0, 3),
            "p90_gap_dollars": _round((_percentile(magnitudes, 0.90) or 0.0) * spot, 2),
            "p95_gap_pct": _round((_percentile(magnitudes, 0.95) or 0.0) * 100.0, 3),
            "p95_gap_dollars": _round((_percentile(magnitudes, 0.95) or 0.0) * spot, 2),
            "mean_gap_pct": _round(mean(magnitudes) * 100.0, 3),
            "mean_gap_dollars": _round(mean(magnitudes) * spot, 2),
        },
    }



def _open_gap_pnl_scenarios(
    magnitudes_dollars: list[tuple[str, float]],
    *,
    spot: float,
    delta: float,
    gamma: float,
    overnight_theta_per_share: float,
) -> list[dict[str, Any]]:
    rows = []
    for label, move in magnitudes_dollars:
        up = _pnl_per_share(delta, gamma, move, overnight_theta_per_share) * 100.0
        down = _pnl_per_share(delta, gamma, -move, overnight_theta_per_share) * 100.0
        rows.append(
            {
                "label": label,
                "move_dollars": _round(move, 2),
                "move_pct": _round(move / spot * 100.0, 3),
                "net_pnl_up_contract": _round(up, 2),
                "net_pnl_down_contract": _round(down, 2),
            }
        )
    return rows



def _estimate_missing_vega(contract: dict[str, Any]) -> tuple[float | None, str | None]:
    provided = _safe_float(contract.get("vega"))
    if provided is not None:
        return provided, "provided"
    estimated = estimate_option_greeks(
        underlying=str(contract.get("symbol") or "QQQ"),
        option_type=str(contract.get("option_type") or "call"),
        spot=_safe_float(contract.get("spot")),
        strike=_safe_float(contract.get("strike")),
        implied_volatility=_safe_float(contract.get("iv")),
        expiration=str(contract.get("expiration") or ""),
        as_of=str(contract.get("close_timestamp") or _utc_now()),
    )
    if not estimated:
        return None, None
    return _safe_float(estimated.get("vega")), "estimated"



def _context_weekday(close_timestamp: str) -> int | None:
    try:
        close_weekday = datetime.fromisoformat(close_timestamp.replace("Z", "+00:00")).weekday()
    except ValueError:
        return None
    return (close_weekday + 1) % 5 if close_weekday < 4 else 0



def _filter_rows_by_context(
    history_rows: list[dict[str, Any]], conditioning_context: dict[str, Any] | None
) -> list[dict[str, Any]]:
    session_dates = set((conditioning_context or {}).get("session_dates") or [])
    if not session_dates:
        return []
    return [row for row in history_rows if str(row.get("session_date"))[:10] in session_dates]



def _stats_context_packet(
    *,
    history_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    source: str | None,
    reason: str | None,
    conditioning_context: dict[str, Any] | None = None,
    label: str = "",
) -> dict[str, Any]:
    packet = {"label": label, "source": source, "reason": reason, **stats}
    if conditioning_context is not None:
        packet["proxy_symbol"] = conditioning_context.get("proxy_symbol")
        packet["filters"] = conditioning_context.get("filters") or {}
        packet["context_match_count"] = conditioning_context.get("match_count") or 0
        packet["overlap_sample_size"] = len(history_rows)
    return packet



def _scenario_moves_from_stats(
    stats: dict[str, Any], be_move: float | None
) -> list[tuple[str, float]]:
    moves = [("break_even", be_move or 0.0)]
    magnitude_stats = stats.get("magnitude_stats") or {}
    for key, label in (
        ("median_gap_dollars", "median_abs_gap"),
        ("p75_gap_dollars", "p75_abs_gap"),
        ("p90_gap_dollars", "p90_abs_gap"),
        ("p95_gap_dollars", "p95_abs_gap"),
    ):
        value = _safe_float(magnitude_stats.get(key))
        if value and value > 0:
            moves.append((label, value))
    return moves



def build_payload(
    contract: dict[str, Any],
    gap_history: dict[str, Any] | None = None,
    conditioning_context: dict[str, Any] | None = None,
    comparison_contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    symbol = str(contract.get("symbol") or "QQQ").upper()
    spot = float(contract["spot"])
    delta = float(contract["delta"])
    gamma = float(contract["gamma"])
    theta = float(contract["theta"])
    iv = float(contract["iv"])
    close_to_open_hours = float(contract.get("close_to_open_hours") or 17.5)
    option_type = str(contract.get("option_type") or "call").lower()
    favorable_sign = _favorable_sign(option_type)
    overnight_theta_per_share = _theta_carry_per_share(theta, close_to_open_hours)
    be_move = _solve_break_even_move(abs(delta), gamma, overnight_theta_per_share)
    be_pct = (be_move / spot) if be_move else 0.0
    flat_rows = _flat_bands(spot, iv, theta, delta, gamma)
    u_rows = _u_shape_rows(spot, iv, theta, delta, gamma)
    history_rows = list((gap_history or {}).get("rows") or [])
    close_timestamp = str(contract.get("close_timestamp") or _utc_now())
    weekday = _context_weekday(close_timestamp)

    empirical = _gap_stats(
        history_rows,
        threshold_pct=be_pct,
        spot=spot,
        target_open_weekday=weekday,
        favorable_sign=favorable_sign,
    )
    unconditioned_moves = _scenario_moves_from_stats(empirical, be_move)

    conditioned_rows = _filter_rows_by_context(history_rows, conditioning_context)
    conditioned_stats = _gap_stats(
        conditioned_rows,
        threshold_pct=be_pct,
        spot=spot,
        target_open_weekday=weekday,
        favorable_sign=favorable_sign,
    )
    conditioned_moves = _scenario_moves_from_stats(conditioned_stats, be_move)

    comparison_packets = []
    for context in comparison_contexts or []:
        comparison_rows = _filter_rows_by_context(history_rows, context)
        comparison_stats = _gap_stats(
            comparison_rows,
            threshold_pct=be_pct,
            spot=spot,
            target_open_weekday=weekday,
            favorable_sign=favorable_sign,
        )
        comparison_packets.append(
            {
                "label": str(context.get("label") or "preset"),
                "gap_context": _stats_context_packet(
                    history_rows=comparison_rows,
                    stats=comparison_stats,
                    source=context.get("source"),
                    reason=context.get("reason"),
                    conditioning_context=context,
                    label=str(context.get("label") or "preset"),
                ),
                "open_gap_pnl_scenarios": _open_gap_pnl_scenarios(
                    _scenario_moves_from_stats(comparison_stats, be_move),
                    spot=spot,
                    delta=delta,
                    gamma=gamma,
                    overnight_theta_per_share=overnight_theta_per_share,
                ),
            }
        )

    vega, vega_source = _estimate_missing_vega(contract)
    iv_shocks = []
    if vega is not None:
        for points in (-5, -3, -1, 1, 3, 5):
            iv_shocks.append(
                {
                    "iv_points": points,
                    "estimated_pnl_contract": _round(vega * points * 100.0, 2),
                }
            )

    return {
        "schema": "sharpedge.overnight_carry_brief.v1",
        "generated_at_utc": _utc_now(),
        "contract": {
            "symbol": symbol,
            "spot": spot,
            "strike": float(contract["strike"]),
            "option_type": option_type,
            "expiration": str(contract["expiration"]),
            "close_timestamp": close_timestamp,
            "delta": delta,
            "gamma": gamma,
            "theta_per_share_per_day": theta,
            "iv": iv,
            "vega": vega,
            "vega_source": vega_source,
        },
        "assumptions": {
            "theta_convention": "calendar_day",
            "close_to_open_hours": close_to_open_hours,
            "trading_day_hours": TRADING_DAY_HOURS,
            "u_shape_segments": [
                {"time_et": time_et, "duration_hours": duration, "weight": weight}
                for time_et, duration, weight in U_SHAPE_SEGMENTS
            ],
            "u_shape_normalization_scalar": _round(
                TRADING_DAY_HOURS / sum(duration * weight for _, duration, weight in U_SHAPE_SEGMENTS),
                5,
            ),
        },
        "overnight_open": {
            "theta_carry_per_share": _round(overnight_theta_per_share, 4),
            "theta_carry_contract": _round(overnight_theta_per_share * 100.0, 2),
            "break_even_move_dollars": _round(be_move, 3),
            "break_even_move_pct": _round(be_pct * 100.0, 3),
            "empirical_gap_context": _stats_context_packet(
                history_rows=history_rows,
                stats=empirical,
                source=(gap_history or {}).get("source"),
                reason=(gap_history or {}).get("reason"),
                label="unconditional",
            ),
            "conditioned_gap_context": _stats_context_packet(
                history_rows=conditioned_rows,
                stats=conditioned_stats,
                source=(conditioning_context or {}).get("source"),
                reason=(conditioning_context or {}).get("reason"),
                conditioning_context=conditioning_context,
                label=str((conditioning_context or {}).get("label") or "conditioned"),
            ),
            "open_gap_pnl_scenarios": _open_gap_pnl_scenarios(
                unconditioned_moves,
                spot=spot,
                delta=delta,
                gamma=gamma,
                overnight_theta_per_share=overnight_theta_per_share,
            ),
            "conditioned_open_gap_pnl_scenarios": _open_gap_pnl_scenarios(
                conditioned_moves,
                spot=spot,
                delta=delta,
                gamma=gamma,
                overnight_theta_per_share=overnight_theta_per_share,
            ),
            "comparison_presets": comparison_packets,
            "open_iv_shock_scenarios": iv_shocks,
        },
        "intraday": {
            "flat_bands": flat_rows,
            "u_shape_bands": u_rows,
        },
        "research_only_warning": (
            "This is a local Greek + gap-distribution research artifact. "
            "It is not execution authority and does not model open spread/slippage."
        ),
    }


__all__ = ["build_payload"]
