from __future__ import annotations

from typing import Any

from market_data_sources import fetch_cboe_options_book


def fetch_cboe_options_book_fail_soft(
    symbol: str = "SPY",
    *,
    timeout: int = 30,
) -> tuple[float, dict[Any, dict[float, dict[str, dict[str, Any]]]], dict[str, Any]]:
    """Fetch CBOE options, returning an explicit unavailable source on failure.

    The live cockpit can still publish price action when the public CBOE endpoint
    denies or times out a request. Strict ingestion callers continue to use
    ``fetch_cboe_options_book`` so data-pipeline failures remain visible.
    """
    try:
        return fetch_cboe_options_book(symbol, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the dashboard's fail-soft boundary
        return 0.0, {}, {
            "provider": "cboe",
            "endpoint": "delayed_quotes/options",
            "symbol": symbol,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
