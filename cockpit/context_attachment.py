"""Safe higher-timeframe context attachment for cockpit execution analysis."""

from __future__ import annotations

from typing import Any

from monthly_context_chart import (
    derive_monthly_levels,
    fetch_monthly_context_rows,
    summarize_monthly_context,
)
from weekly_context_chart import (
    derive_today_carry_levels,
    fetch_weekly_context_rows,
    summarize_weekly_context,
)


def _unavailable_weekly_context(spot: float, error: Exception) -> dict[str, Any]:
    context = summarize_weekly_context([], [], spot=spot, symbol="SPY", lookback_days=5)
    return {
        **context,
        "context_available": False,
        "error": str(error),
        "detail": f"Weekly context unavailable; edge checks can still use OR/PD levels. Error: {error}",
    }


def _unavailable_monthly_context(spot: float, error: Exception) -> dict[str, Any]:
    context = summarize_monthly_context(
        [], [], spot=spot, symbol="SPY", lookback_months=6
    )
    return {
        **context,
        "context_available": False,
        "error": str(error),
        "detail": f"Monthly context unavailable; edge checks can still use OR/PD levels. Error: {error}",
    }


def build_context_attachment(
    session_rows: list[tuple[int, float, float, float, float, int]],
    *,
    spot: float,
) -> dict[str, Any]:
    """Return context packets before permission scoring runs.

    The returned weekly/monthly context dicts are always present. If source fetch
    fails, they carry context_available=False and an explicit error instead of
    silently weakening edge/playbook checks.
    """
    carry_levels = derive_today_carry_levels(session_rows)
    weekly_rows: list[dict[str, Any]] = []
    weekly_source: dict[str, Any] = {}
    try:
        weekly_rows, weekly_source = fetch_weekly_context_rows()
        weekly_context = summarize_weekly_context(
            weekly_rows,
            carry_levels,
            spot=spot,
            symbol="SPY",
            lookback_days=5,
        )
        weekly_context = {**weekly_context, "context_available": True}
    except Exception as exc:  # pragma: no cover - defensive live-data fallback
        weekly_context = _unavailable_weekly_context(spot, exc)

    monthly_rows: list[dict[str, Any]] = []
    monthly_source: dict[str, Any] = {}
    monthly_levels: list[dict[str, Any]] = []
    try:
        monthly_rows, monthly_source = fetch_monthly_context_rows()
        monthly_levels = derive_monthly_levels(monthly_rows)
        monthly_context = summarize_monthly_context(
            monthly_rows,
            monthly_levels,
            spot=spot,
            symbol="SPY",
            lookback_months=6,
        )
        monthly_context = {**monthly_context, "context_available": True}
    except Exception as exc:  # pragma: no cover - defensive live-data fallback
        monthly_context = _unavailable_monthly_context(spot, exc)

    return {
        "weekly_context": weekly_context,
        "monthly_context": monthly_context,
        "weekly_rows": weekly_rows,
        "monthly_rows": monthly_rows,
        "carry_levels": carry_levels,
        "monthly_levels": monthly_levels,
        "weekly_source": weekly_source,
        "monthly_source": monthly_source,
    }


__all__ = ["build_context_attachment"]
