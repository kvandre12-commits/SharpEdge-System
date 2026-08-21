from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt_money(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _flag_tokens(item: dict[str, Any]) -> set[str]:
    raw = str(item.get("flags") or item.get("rejection_flags") or "")
    return {token.strip().lower() for token in raw.split(";") if token.strip()}


def _quote_quality_multiplier(item: dict[str, Any]) -> float:
    flags = _flag_tokens(item)
    mid = _safe_float(item.get("mid"))
    bid = _safe_float(item.get("bid"))
    ask = _safe_float(item.get("ask"))
    width_pct = _safe_float(item.get("width_pct"))
    priority = str(
        item.get("priority") or item.get("manual_validation_priority") or ""
    ).lower()

    quality = 1.0
    if mid is None or mid <= 0:
        quality *= 0.2
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        quality *= 0.4
    if "missing_midpoint" in flags:
        quality *= 0.5
    if "zero_or_tiny_market" in flags:
        quality *= 0.45
    if priority == "reject":
        quality *= 0.7
    if width_pct is None:
        quality *= 0.9
    elif width_pct >= 0.25:
        quality *= 0.55
    elif width_pct >= 0.12:
        quality *= 0.75

    return max(0.05, min(quality, 1.0))


def _pressure_score(item: dict[str, Any]) -> float:
    volume = _safe_int(item.get("volume"))
    open_interest = _safe_int(item.get("open_interest"))
    return (volume + 0.2 * open_interest) * _quote_quality_multiplier(item)


def describe_contract(item: dict[str, Any] | None) -> str:
    if item is None:
        return "No clean liquid contract selected yet."
    return (
        f"{item.get('expiration') or ''} {str(item.get('option_type') or '').upper()} "
        f"{_safe_float(item.get('strike')) or 0.0:g} mid {_fmt_money(item.get('mid'))} "
        f"bid/ask {_fmt_money(item.get('bid'))}/{_fmt_money(item.get('ask'))} "
        f"vol {_safe_int(item.get('volume'))} OI {_safe_int(item.get('open_interest'))} "
        f"({item.get('role') or 'flow'})."
    )


def _preferred_roles_for_side(option_type: str, downside_bias: bool) -> set[str]:
    if option_type == "put":
        return {"downside-hedge", "near-money-flow"}
    if downside_bias:
        return {"near-money-flow", "target-call"}
    return {"target-call", "near-money-flow"}


def _sorted_side_rows(
    focus_contracts: list[dict[str, Any]],
    *,
    option_type: str,
    preferred_roles: set[str],
) -> list[dict[str, Any]]:
    rows = [
        item
        for item in focus_contracts
        if str(item.get("option_type") or "").lower() == option_type
    ]
    rows.sort(
        key=lambda item: (
            item.get("role") not in preferred_roles,
            -_pressure_score(item),
            -_safe_int(item.get("volume")),
            -_safe_int(item.get("open_interest")),
        )
    )
    return rows


def _total_pressure(rows: list[dict[str, Any]]) -> float:
    return sum(_pressure_score(item) for item in rows)


def _average_quality(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(_quote_quality_multiplier(item) for item in rows) / len(rows)


def _quality_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.35:
        return "fragile"
    if value < 0.7:
        return "mixed"
    return "usable"


def _pressure_split(put_pressure: float, call_pressure: float) -> dict[str, Any]:
    total = put_pressure + call_pressure
    if total <= 0:
        return {
            "put_pressure_score": 0.0,
            "call_pressure_score": 0.0,
            "put_pressure_pct": 50,
            "call_pressure_pct": 50,
            "dominant_side": "unknown",
        }
    put_pressure_pct = round((put_pressure / total) * 100)
    call_pressure_pct = max(0, 100 - put_pressure_pct)
    if put_pressure > call_pressure:
        dominant_side = "put"
    elif call_pressure > put_pressure:
        dominant_side = "call"
    else:
        dominant_side = "balanced"
    return {
        "put_pressure_score": round(put_pressure, 2),
        "call_pressure_score": round(call_pressure, 2),
        "put_pressure_pct": put_pressure_pct,
        "call_pressure_pct": call_pressure_pct,
        "dominant_side": dominant_side,
    }


def _summarize_option_side(
    focus_contracts: list[dict[str, Any]], *, option_type: str, downside_bias: bool
) -> dict[str, Any]:
    preferred_roles = _preferred_roles_for_side(option_type, downside_bias)
    rows = _sorted_side_rows(
        focus_contracts,
        option_type=option_type,
        preferred_roles=preferred_roles,
    )
    top_rows = rows[:2]
    pressure = _total_pressure(rows[:3])
    quality = _average_quality(rows[:3])
    label = _quality_label(quality)
    return {
        "option_type": option_type,
        "rows": rows,
        "top_rows": top_rows,
        "pressure": pressure,
        "quality": quality,
        "quality_label": label,
        "contracts": [describe_contract(item) for item in top_rows],
        "summary": (
            f"{option_type.upper()} side quality {label}; strongest visible lines are "
            f"{' | '.join(describe_contract(item) for item in top_rows) if top_rows else 'none yet.'}"
        ),
    }


def _summarize_balance(
    *,
    downside_bias: bool,
    put_side: dict[str, Any],
    call_side: dict[str, Any],
) -> dict[str, str]:
    supporting_side = "put" if downside_bias else "call"
    opposing_side = "call" if downside_bias else "put"
    supporting_pressure = (
        put_side["pressure"] if downside_bias else call_side["pressure"]
    )
    opposing_pressure = call_side["pressure"] if downside_bias else put_side["pressure"]
    ratio = (
        supporting_pressure / opposing_pressure
        if supporting_pressure and opposing_pressure
        else None
    )

    if supporting_pressure < 1 and opposing_pressure < 1:
        return {
            "flow_balance": "No clean quote-weighted focus-line pressure yet.",
            "bias_alignment": "unknown",
        }
    if (
        opposing_pressure < 1
        or supporting_pressure >= max(opposing_pressure, 1.0) * 1.5
    ):
        return {
            "flow_balance": (
                f"{supporting_side.upper()}-led tape; quote-weighted focus-line pressure is {_fmt_ratio(ratio)} "
                f"in favor of {supporting_side.upper()}s."
            ),
            "bias_alignment": "aligned",
        }
    if (
        supporting_pressure < 1
        or opposing_pressure >= max(supporting_pressure, 1.0) * 1.5
    ):
        reverse_ratio = (
            opposing_pressure / supporting_pressure if supporting_pressure else None
        )
        return {
            "flow_balance": (
                f"Crosswired tape; {opposing_side.upper()}s are leading quote-weighted focus-line pressure "
                f"by {_fmt_ratio(reverse_ratio)} against the stated bias."
            ),
            "bias_alignment": "crosswired",
        }
    return {
        "flow_balance": (
            "Mixed tape; PUT and CALL quote-weighted focus-line pressure are close enough that price still has to do the talking."
        ),
        "bias_alignment": "mixed",
    }


def _quote_quality_context(put_side: dict[str, Any], call_side: dict[str, Any]) -> str:
    put_quality = put_side["quality"]
    call_quality = call_side["quality"]
    if put_quality is None and call_quality is None:
        return "No usable quote-quality read yet."
    if (put_quality or 0.0) < 0.35 and (call_quality or 0.0) < 0.35:
        return (
            "Both sides are running on ugly quotes, so treat the flow read as fragile."
        )
    if (put_quality or 0.0) < 0.35:
        return "PUT flow is being down-weighted by weak quote quality."
    if (call_quality or 0.0) < 0.35:
        return "CALL flow looks loud on raw volume, but bad quotes are muting it."
    return "Both sides have usable enough quotes for the flow comparison to matter."


def _confirm_invalidate_lines(
    *, downside_bias: bool, nearest_level: Any, spot: Any
) -> tuple[list[str], list[str]]:
    nearest = _safe_float(nearest_level)
    current_spot = _safe_float(spot)
    nearest_text = _fmt_money(nearest)
    spot_text = _fmt_money(current_spot)
    if downside_bias:
        rejection_line = (
            f"Puts keep leading calls while spot stays rejected under {nearest_text}."
            if nearest is not None
            else "Puts keep leading calls while price remains below accepted intraday resistance."
        )
        reclaim_line = (
            f"Spot reclaims and accepts back above {nearest_text}."
            if nearest is not None
            else "Price reclaims accepted intraday resistance; no validated numeric level is available yet."
        )
        stabilization_line = (
            f"Calls retake tape leadership while spot stabilizes around {spot_text}."
            if current_spot is not None
            else "Calls retake tape leadership while price stabilizes."
        )
        confirms = [
            rejection_line,
            "Supporting put mids stay bid instead of collapsing after the first downtick.",
            "Opposing calls fail to regain tape leadership on bounces.",
        ]
        invalidates = [
            reclaim_line,
            stabilization_line,
            "Put volume spikes once and then vanishes into hedge noise.",
        ]
        return confirms, invalidates

    acceptance_line = (
        f"Calls keep leading puts while spot accepts above {nearest_text}."
        if nearest is not None
        else "Calls keep leading puts while price holds accepted intraday support."
    )
    loss_line = (
        f"Spot loses {nearest_text} and cannot reclaim it cleanly."
        if nearest is not None
        else "Price loses accepted intraday support; no validated numeric level is available yet."
    )
    stall_line = (
        f"Puts retake tape leadership while spot stalls around {spot_text}."
        if current_spot is not None
        else "Puts retake tape leadership while price stalls."
    )
    confirms = [
        acceptance_line,
        "Supporting call mids stay bid instead of bleeding on every pause.",
        "Opposing puts fail to retake tape leadership on pullbacks.",
    ]
    invalidates = [
        loss_line,
        stall_line,
        "Call volume prints once and then fades into premium harvest.",
    ]
    return confirms, invalidates


def _plain_english_summary(
    *,
    downside_bias: bool,
    setup_bias: str,
    target_strike: Any,
    heat_label: str,
    iv_rv_ratio_text: str,
    flow_balance: str,
) -> str:
    if downside_bias:
        return (
            f"SharpEdge is leaning {setup_bias}. NERV says the liquid put neighborhood is sitting in the downside hedge lines "
            f"instead of upside chase calls. IV/RV13 is {iv_rv_ratio_text} ({heat_label}), so you still need real rejection before trusting the bearish tape. "
            f"{flow_balance}"
        )
    return (
        f"SharpEdge is leaning {setup_bias}. NERV says the liquid call neighborhood is around {_fmt_money(target_strike)} "
        f"with IV/RV13 {iv_rv_ratio_text} ({heat_label}). That means the idea is real enough to watch, but premium is not free; "
        f"wait for price acceptance or post-event harvest. {flow_balance}"
    )


def _primary_contract(
    focus_contracts: list[dict[str, Any]], *, downside_bias: bool
) -> dict[str, Any] | None:
    if downside_bias:
        return focus_contracts[0] if focus_contracts else None
    return next(
        (item for item in focus_contracts if item.get("role") == "target-call"),
        focus_contracts[0] if focus_contracts else None,
    )


def build_hey_guy_summary(
    *,
    headline: str,
    stance: str,
    setup_bias: str,
    target_strike: Any,
    heat_label: str,
    iv_rv_ratio_text: str,
    downside_bias: bool,
    focus_contracts: list[dict[str, Any]],
    nearest_level: Any,
    spot: Any,
) -> dict[str, Any]:
    put_side = _summarize_option_side(
        focus_contracts,
        option_type="put",
        downside_bias=downside_bias,
    )
    call_side = _summarize_option_side(
        focus_contracts,
        option_type="call",
        downside_bias=downside_bias,
    )
    balance = _summarize_balance(
        downside_bias=downside_bias,
        put_side=put_side,
        call_side=call_side,
    )
    confirms, invalidates = _confirm_invalidate_lines(
        downside_bias=downside_bias,
        nearest_level=nearest_level,
        spot=spot,
    )
    pressure_split = _pressure_split(put_side["pressure"], call_side["pressure"])
    blockers = []
    if not focus_contracts:
        blockers.append("no_focus_contracts")
    if put_side["quality"] is None and call_side["quality"] is None:
        blockers.append("quote_quality_unavailable")
    status = "ready" if not blockers else "degraded"
    return {
        "status": status,
        "blockers": blockers,
        "title": "Hey guy — SharpEdge/NERV read",
        "one_liner": headline,
        "plain_english": _plain_english_summary(
            downside_bias=downside_bias,
            setup_bias=setup_bias,
            target_strike=target_strike,
            heat_label=heat_label,
            iv_rv_ratio_text=iv_rv_ratio_text,
            flow_balance=balance["flow_balance"],
        ),
        "liquidity_spot": describe_contract(
            _primary_contract(focus_contracts, downside_bias=downside_bias)
        ),
        "put_flow": put_side["contracts"],
        "call_flow": call_side["contracts"],
        "put_side_summary": put_side["summary"],
        "call_side_summary": call_side["summary"],
        "near_money_tape": [*put_side["contracts"], *call_side["contracts"]][:4],
        "supporting_flow": put_side["contracts"]
        if downside_bias
        else call_side["contracts"],
        "opposing_flow": call_side["contracts"]
        if downside_bias
        else put_side["contracts"],
        "flow_balance": balance["flow_balance"],
        "bias_alignment": balance["bias_alignment"],
        "quote_quality_context": _quote_quality_context(put_side, call_side),
        **pressure_split,
        "confirms": confirms,
        "invalidates": invalidates,
        "operator_note": "Descriptive only. Use broker-fresh quotes; approval_decision is still the authority object.",
        "stance": stance,
    }
