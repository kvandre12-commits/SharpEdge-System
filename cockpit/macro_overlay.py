"""Live macro overlay for the SharpEdge cockpit.

Computes the FRED macro-overlay signals (VIX level, VIX term structure, 10Y rate
impulse) LIVE from Yahoo index feeds (^VIX, ^VIX3M, ^TNX) instead of the
credential-gated, end-of-day-lagged FRED API. Same z-score + z_to_strength math
as scripts/ingest_fred_overlays.py so the live read and the batch overlay agree.

Term structure is the headline: VIX/VIX3M < 1 = contango (calm, normal), >= 1 =
backwardation (acute fear / risk-off). This changes the playbook intraday.
"""

from __future__ import annotations

import math
from typing import Any, Optional

# Yahoo symbols (caret URL-encoded for the chart endpoint).
VIX_SYMBOL = "%5EVIX"
VIX3M_SYMBOL = "%5EVIX3M"
TNX_SYMBOL = "%5ETNX"

ZSCORE_WINDOW = 252
ZSCORE_MIN_PERIODS = 40


def z_to_strength(value: Any) -> float:
    """Map a z-score to a [0,1] overlay strength (canonical FRED-overlay math)."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return float(max(0.0, min(1.0, (v - 1.0) / 1.5)))


def current_z(
    values: list[float], win: int = ZSCORE_WINDOW, min_periods: int = ZSCORE_MIN_PERIODS
) -> float | None:
    """Trailing z-score of the most recent value over up-to-`win` samples."""
    if not values:
        return None
    sample = values[-win:]
    if len(sample) < min_periods:
        return None
    mean = sum(sample) / len(sample)
    var = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return (sample[-1] - mean) / std


def _closes_by_date(daily_bars: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(b["date"]): float(b["close"])
        for b in (daily_bars or [])
        if b.get("date") is not None and b.get("close") is not None
    }


def _term_regime(term_ratio: float | None) -> tuple[str, str]:
    if term_ratio is None:
        return "UNKNOWN", "VIX term structure unavailable."
    if term_ratio >= 1.0:
        return (
            "BACKWARDATION",
            "VIX above 3M VIX — backwardation. Acute near-term fear / risk-off; "
            "expect sharp, trend-prone tape and unstable pins.",
        )
    if term_ratio >= 0.95:
        return (
            "FLATTENING",
            "VIX term structure flattening toward 1.0 — stress building; "
            "calm-day assumptions are getting fragile.",
        )
    return (
        "CONTANGO",
        "VIX below 3M VIX — normal contango. Calm regime; mean-reversion / pin "
        "behavior is more reliable.",
    )


def build_macro_overlay_live(
    vix_bars: list[dict[str, Any]],
    vix3m_bars: list[dict[str, Any]],
    tnx_bars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the live macro overlay packet from ^VIX / ^VIX3M / ^TNX daily bars."""
    vix_by = _closes_by_date(vix_bars)
    vix3m_by = _closes_by_date(vix3m_bars)
    tnx_by = _closes_by_date(tnx_bars)

    if not vix_by:
        return {
            "schema": "sharpedge.macro_overlay.v1",
            "available": False,
            "reason": "no VIX data",
            "source": "live:yahoo_indices",
        }

    vix_dates = sorted(vix_by)
    vix_series = [vix_by[d] for d in vix_dates]
    vix_level = vix_series[-1]
    vix_z = current_z(vix_series)
    vix3m_level = vix3m_by[sorted(vix3m_by)[-1]] if vix3m_by else None

    # VIX term structure over the intersection of VIX / VIX3M dates.
    term_series: list[float] = []
    common_dates = sorted(set(vix_by) & set(vix3m_by))
    for d in common_dates:
        v3 = vix3m_by[d]
        if v3:
            term_series.append(vix_by[d] / v3)
    vix_term = term_series[-1] if term_series else None
    vix_term_z = current_z(term_series) if term_series else None
    term_regime, term_story = _term_regime(vix_term)

    # 10Y rate impulse: 5-day change of ^TNX, z-scored.
    tnx_dates = sorted(tnx_by)
    tnx_series = [tnx_by[d] for d in tnx_dates]
    rates_10y = tnx_series[-1] if tnx_series else None
    impulse_series: list[float] = []
    if len(tnx_series) > 5:
        impulse_series = [tnx_series[i] - tnx_series[i - 5] for i in range(5, len(tnx_series))]
    rates_impulse_z = current_z(impulse_series) if impulse_series else None

    # Overall macro headline.
    risk_off = term_regime == "BACKWARDATION" or (isinstance(vix_z, float) and vix_z >= 1.5)
    risk_on = term_regime == "CONTANGO" and (isinstance(vix_z, float) and vix_z <= 0.0)
    if risk_off:
        macro_state = "RISK_OFF"
    elif risk_on:
        macro_state = "RISK_ON_CALM"
    else:
        macro_state = "NEUTRAL"

    rate_note = ""
    if isinstance(rates_impulse_z, float):
        if rates_impulse_z >= 1.0:
            rate_note = " 10Y yields impulsing UP (rate pressure)."
        elif rates_impulse_z <= -1.0:
            rate_note = " 10Y yields impulsing DOWN (rate relief)."

    return {
        "schema": "sharpedge.macro_overlay.v1",
        "available": True,
        "macro_state": macro_state,
        "vix": round(vix_level, 2),
        "vix_z": round(vix_z, 2) if isinstance(vix_z, float) else None,
        "vix3m": round(vix3m_level, 2) if isinstance(vix3m_level, (int, float)) else None,
        "vix_term": round(vix_term, 3) if isinstance(vix_term, float) else None,
        "vix_term_z": round(vix_term_z, 2) if isinstance(vix_term_z, float) else None,
        "term_regime": term_regime,
        "rates_10y": round(rates_10y, 2) if isinstance(rates_10y, (int, float)) else None,
        "rates_impulse_z": round(rates_impulse_z, 2) if isinstance(rates_impulse_z, float) else None,
        "vix_strength": z_to_strength(vix_z),
        "vix_term_strength": z_to_strength(vix_term_z),
        "story": term_story + rate_note,
        "source": "live:yahoo_indices",
        "classifier": "canonical:macro_overlay",
    }


__all__ = [
    "TNX_SYMBOL",
    "VIX3M_SYMBOL",
    "VIX_SYMBOL",
    "build_macro_overlay_live",
    "current_z",
    "z_to_strength",
]
