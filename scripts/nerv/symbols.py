"""OCC option-symbol helpers for NERV."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

_OCC_TAIL_RE = re.compile(r"(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def normalize_underlying(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "")


def _date_from_any(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def format_occ_symbol(
    underlying: str,
    expiration: str | date | datetime,
    option_type: str,
    strike: float,
) -> str:
    """Format a compact OCC-style symbol like ``SPY250117C00500000``."""
    root = normalize_underlying(underlying)
    expiry = _date_from_any(expiration).strftime("%y%m%d")
    cp = option_type.strip().upper()[0]
    if cp not in {"C", "P"}:
        raise ValueError(f"option_type must be call/put or C/P, got {option_type!r}")
    strike_int = int(round(float(strike) * 1000))
    return f"{root}{expiry}{cp}{strike_int:08d}"


def parse_occ_symbol(symbol: str) -> dict[str, Any] | None:
    """Parse a compact OCC-style symbol.

    Returns ``None`` instead of raising for non-matches because upstream vendors
    occasionally hand us weird corporate-action goblins. We quarantine those at
    adapter boundaries instead of detonating the whole nightly run.
    """
    compact = symbol.strip().replace(" ", "").upper()
    match = _OCC_TAIL_RE.search(compact)
    if not match:
        return None
    root = compact[: match.start()].strip()
    if not root:
        return None
    expiry_raw = match.group("expiry")
    cp = match.group("cp")
    strike = int(match.group("strike")) / 1000.0
    expiry = datetime.strptime(expiry_raw, "%y%m%d").date().isoformat()
    return {
        "underlying": root,
        "expiration": expiry,
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
        "contract_symbol": compact,
    }
