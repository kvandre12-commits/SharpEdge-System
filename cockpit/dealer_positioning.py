"""Canonical aggregate dealer/options positioning for SharpEdge.

SINGLE SOURCE OF TRUTH for gamma-flip + dealer-state logic, shared between the
historical batch table builder (scripts/aggregate_options_positioning_metrics.py)
and the LIVE cockpit read. The two callers prepare per-strike aggregates from
different data sources (batch = SQLite chain/OI tables; cockpit = live CBOE book)
but funnel through the SAME compute_flip() and compute_dealer_state() so labels
cannot diverge.

DTE window is CONFIGURABLE. The live cockpit defaults to a 0-1 DTE blend
(DEALER_DTE_MIN / DEALER_DTE_MAX env), which captures the short-dated dealer
positioning that actually drives SPY intraday pin/flip behavior — instead of the
single nearest expiry the cockpit gamma card uses today.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Optional

import numpy as np

# Configurable live DTE window (default 0-1). Batch uses its own DTE_MIN/MAX.
DEALER_DTE_MIN = int(os.getenv("DEALER_DTE_MIN", "0"))
DEALER_DTE_MAX = int(os.getenv("DEALER_DTE_MAX", "1"))
PIN_THRESH_PCT = float(os.getenv("PIN_THRESH_PCT", "0.0025"))
ACTIONABLE_OI_WALL_MAX_DIST_PCT = float(
    os.getenv("ACTIONABLE_OI_WALL_MAX_DIST_PCT", "0.05")
)


# --------------------------- canonical shared math ---------------------------
def compute_flip(
    strikes: list[float], net: list[float], spot: Optional[float]
) -> Optional[float]:
    """Interpolate the price where net dealer gamma crosses zero (the flip).

    Spot-anchored: among ALL sign changes, return the interpolated crossing
    NEAREST to spot, rather than the first crossing scanning up from the lowest
    strike. This avoids picking a meaningless zero-crossing out in the low-gamma
    tails (the classic naive-flip artifact). Returns None when spot is
    unavailable (cannot anchor) or no sign change exists.
    """
    if spot is None or len(strikes) < 2:
        return None
    s = np.array(strikes, dtype=float)
    g = np.array(net, dtype=float)
    order = np.argsort(s)
    s, g = s[order], g[order]

    crossings: list[float] = []
    for a in range(len(s) - 1):
        if np.sign(g[a]) != np.sign(g[a + 1]):
            x1, x2 = float(s[a]), float(s[a + 1])
            g1, g2 = float(g[a]), float(g[a + 1])
            crossings.append(
                x1 if g2 == g1 else float(x1 + (0.0 - g1) * (x2 - x1) / (g2 - g1))
            )

    if not crossings:
        return None
    return min(crossings, key=lambda x: abs(x - float(spot)))


def compute_dealer_state(
    spot: Optional[float],
    gamma_flip: Optional[float],
    max_total_oi_strike: Optional[float],
    pcr_oi: Optional[float],
    pcr_vol: Optional[float],
    *,
    pin_thresh_pct: float = PIN_THRESH_PCT,
) -> tuple[Optional[float], Optional[str]]:
    """Canonical dealer-state machine (verbatim rules from the batch builder)."""
    if spot is None:
        return None, None

    gamma_proxy = None
    dealer_hint = "NEUTRAL"

    if gamma_flip is not None:
        gamma_proxy = float(spot - gamma_flip)
        if gamma_proxy > 0:
            dealer_hint = "LONG_GAMMA"
        elif gamma_proxy < 0:
            dealer_hint = "SHORT_GAMMA"

    if max_total_oi_strike is not None:
        wall_distance_pct = abs(spot - max_total_oi_strike) / spot
        if wall_distance_pct <= pin_thresh_pct:
            dealer_hint = "PINNED"

    if pcr_oi is not None:
        if pcr_oi > 1.4:
            dealer_hint = "DEFENSIVE"
        elif pcr_oi < 0.7 and dealer_hint != "PINNED":
            dealer_hint = "CHASE"

    if pcr_vol is not None and pcr_vol > 1.8:
        dealer_hint = "UNWIND_RISK"

    return gamma_proxy, dealer_hint


def _argmax_positive(d: dict[float, float]) -> Optional[float]:
    if not d:
        return None
    k = max(d.keys(), key=lambda x: d[x])
    return float(k) if d[k] > 0 else None


def _argmax_positive_near_spot(
    d: dict[float, float],
    spot: Optional[float],
    *,
    max_dist_pct: float = ACTIONABLE_OI_WALL_MAX_DIST_PCT,
) -> Optional[float]:
    if spot is None or spot <= 0:
        return _argmax_positive(d)
    near = {
        strike: value
        for strike, value in d.items()
        if abs(float(strike) - float(spot)) / float(spot) <= max_dist_pct
    }
    return _argmax_positive(near)


def _distance_pct(strike: Optional[float], spot: Optional[float]) -> Optional[float]:
    if strike is None or spot is None or spot <= 0:
        return None
    return (float(strike) - float(spot)) / float(spot)


def _safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------- live builder (CBOE book) ---------------------------
_DEALER_STORY = {
    "LONG_GAMMA": "Gamma/OI proxy is long-gamma above the flip — moves may dampen. Pin/chop bias; fade edges.",
    "SHORT_GAMMA": "Gamma/OI proxy is short-gamma below the flip — moves may amplify. Trend/runner risk after trigger.",
    "PINNED": "Spot is near the largest OI strike — magnet/pin risk into expiry.",
    "DEFENSIVE": "Put-heavy OI proxy (PCR > 1.4) — defensive positioning / downside hedge pressure.",
    "CHASE": "Call-heavy OI proxy (PCR < 0.7) — upside chase positioning risk.",
    "UNWIND_RISK": "Very put-heavy volume proxy (PCR_vol > 1.8) — hedge-unwind / squeeze risk.",
    "NEUTRAL": "No decisive gamma/OI proxy skew in this DTE window.",
}


def build_dealer_positioning_live(
    book: dict[Any, dict[float, dict[str, dict[str, Any]]]],
    spot: Optional[float],
    *,
    dte_min: int = DEALER_DTE_MIN,
    dte_max: int = DEALER_DTE_MAX,
    symbol: str = "SPY",
    today: Optional[dt.date] = None,
) -> dict[str, Any]:
    """Aggregate dealer positioning across a CONFIGURABLE DTE window from the live CBOE book.

    book: {expiry_date: {strike: {"C": opt, "P": opt}}}. Blends every expiry with
    dte in [dte_min, dte_max]. Returns a sharpedge.dealer_positioning.v1 packet.
    """
    today = today or dt.date.today()
    if not book:
        return {
            "schema": "sharpedge.dealer_positioning.v1",
            "available": False,
            "dealer_state": None,
            "reason": "no options book",
        }

    included_expiries = [
        exp
        for exp in book
        if isinstance(exp, dt.date) and dte_min <= (exp - today).days <= dte_max
    ]
    if not included_expiries:
        return {
            "schema": "sharpedge.dealer_positioning.v1",
            "available": False,
            "dealer_state": None,
            "dte_window": [dte_min, dte_max],
            "reason": f"no expiries in {dte_min}-{dte_max} DTE window",
        }

    net_gamma_by: dict[float, float] = {}
    call_oi_by: dict[float, float] = {}
    put_oi_by: dict[float, float] = {}
    total_call_vol = 0.0
    total_put_vol = 0.0

    for exp in included_expiries:
        chain = book[exp]
        for strike, legs in chain.items():
            k = float(strike)
            c = legs.get("C", {}) or {}
            p = legs.get("P", {}) or {}
            cg = _safe_float(c.get("gamma"))
            pg = _safe_float(p.get("gamma"))
            coi = _safe_float(c.get("open_interest"))
            poi = _safe_float(p.get("open_interest"))
            net_gamma_by[k] = net_gamma_by.get(k, 0.0) + (cg * coi - pg * poi)
            call_oi_by[k] = call_oi_by.get(k, 0.0) + coi
            put_oi_by[k] = put_oi_by.get(k, 0.0) + poi
            total_call_vol += _safe_float(c.get("volume"))
            total_put_vol += _safe_float(p.get("volume"))

    strikes = sorted(net_gamma_by.keys())
    if not strikes:
        return {
            "schema": "sharpedge.dealer_positioning.v1",
            "available": False,
            "dealer_state": None,
            "dte_window": [dte_min, dte_max],
            "reason": "no strikes in window",
        }

    net = [net_gamma_by[k] for k in strikes]
    gamma_wall = max(strikes, key=lambda k: abs(net_gamma_by[k]))
    gamma_pos_wall = max(strikes, key=lambda k: net_gamma_by[k])
    gamma_neg_wall = min(strikes, key=lambda k: net_gamma_by[k])
    gamma_flip = compute_flip(strikes, net, spot)

    total_call_oi = sum(call_oi_by.values())
    total_put_oi = sum(put_oi_by.values())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else None
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol else None

    total_oi_by = {k: call_oi_by.get(k, 0.0) + put_oi_by.get(k, 0.0) for k in strikes}
    max_total_oi_strike = _argmax_positive(total_oi_by)
    max_call_oi_strike = _argmax_positive(call_oi_by)
    max_put_oi_strike = _argmax_positive(put_oi_by)
    actionable_oi_wall_strike = _argmax_positive_near_spot(total_oi_by, spot)
    actionable_call_wall_strike = _argmax_positive_near_spot(call_oi_by, spot)
    actionable_put_wall_strike = _argmax_positive_near_spot(put_oi_by, spot)

    gamma_proxy, dealer_state = compute_dealer_state(
        spot=spot,
        gamma_flip=gamma_flip,
        max_total_oi_strike=actionable_oi_wall_strike,
        pcr_oi=pcr_oi,
        pcr_vol=pcr_vol,
    )

    net_gamma_total = float(sum(net))
    return {
        "schema": "sharpedge.dealer_positioning.v1",
        "symbol": symbol,
        "available": True,
        "dte_window": [dte_min, dte_max],
        "expiries_used": sorted(e.isoformat() for e in included_expiries),
        "spot": spot,
        "dealer_state": dealer_state,
        "gamma_regime": "positive" if net_gamma_total >= 0 else "negative",
        "net_gamma_total": net_gamma_total,
        "gamma_flip_strike": gamma_flip,
        "gamma_proxy": gamma_proxy,
        "gamma_wall_strike": gamma_wall,
        "gamma_pos_wall_strike": gamma_pos_wall,
        "gamma_neg_wall_strike": gamma_neg_wall,
        "max_total_oi_strike": max_total_oi_strike,
        "max_call_oi_strike": max_call_oi_strike,
        "max_put_oi_strike": max_put_oi_strike,
        "actionable_oi_wall_strike": actionable_oi_wall_strike,
        "actionable_call_wall_strike": actionable_call_wall_strike,
        "actionable_put_wall_strike": actionable_put_wall_strike,
        "actionable_oi_wall_max_dist_pct": ACTIONABLE_OI_WALL_MAX_DIST_PCT,
        "actionable_oi_wall_distance_pct": _distance_pct(
            actionable_oi_wall_strike, spot
        ),
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "story": _DEALER_STORY.get(str(dealer_state), _DEALER_STORY["NEUTRAL"]),
        "source": "live:cboe_book",
        "classifier": "canonical:dealer_positioning",
    }


__all__ = [
    "compute_flip",
    "compute_dealer_state",
    "build_dealer_positioning_live",
    "DEALER_DTE_MIN",
    "DEALER_DTE_MAX",
    "PIN_THRESH_PCT",
]
