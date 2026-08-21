"""Credential-seam checks for NERV providers.

This intentionally does not authenticate or fetch. It only tells the adapter CLI
which official routes appear configured so operators can see why a run used the
free yfinance path instead of Alpaca/Tradier.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    configured: bool
    available: bool
    status: str
    blockers: tuple[str, ...]
    data_mode: str
    note: str

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "configured": self.configured,
            "available": self.available,
            "status": self.status,
            "blockers": list(self.blockers),
            "data_mode": self.data_mode,
            "note": self.note,
        }


def alpaca_status() -> ProviderStatus:
    has_key = bool(os.getenv("ALPACA_API_KEY", "").strip())
    has_secret = bool(os.getenv("ALPACA_API_SECRET", "").strip())
    configured = has_key and has_secret
    blockers = () if configured else ("credentials_missing",)
    return ProviderStatus(
        name="alpaca",
        configured=configured,
        available=configured,
        status="ready" if configured else "credentials_missing",
        blockers=blockers,
        data_mode="indicative_options_feed_basic_or_configured_plan",
        note="Requires ALPACA_API_KEY and ALPACA_API_SECRET.",
    )


def tradier_status() -> ProviderStatus:
    has_token = bool(os.getenv("TRADIER_TOKEN", "").strip())
    has_account = bool(os.getenv("TRADIER_ACCOUNT_ID", "").strip())
    blockers = () if has_token else ("credentials_missing",)
    return ProviderStatus(
        name="tradier",
        configured=has_token,
        available=has_token,
        status="ready" if has_token else "credentials_missing",
        blockers=blockers,
        data_mode="brokerage_realtime_or_sandbox_delayed_by_token_environment",
        note=(
            "Requires TRADIER_TOKEN for market data. TRADIER_ACCOUNT_ID is only "
            f"present={has_account}; do not use NERV for order submission."
        ),
    )


def yfinance_status() -> ProviderStatus:
    missing = tuple(
        f"dependency_missing:{module}"
        for module in ("yfinance", "pandas")
        if importlib.util.find_spec(module) is None
    )
    return ProviderStatus(
        name="yfinance",
        configured=True,
        available=not missing,
        status="ready" if not missing else "dependency_missing",
        blockers=missing,
        data_mode="unofficial_yahoo_personal_research_delayed_or_unknown",
        note="No credentials. Bulk discovery/fallback only; not source of record.",
    )


def provider_statuses() -> list[ProviderStatus]:
    return [tradier_status(), alpaca_status(), yfinance_status()]
