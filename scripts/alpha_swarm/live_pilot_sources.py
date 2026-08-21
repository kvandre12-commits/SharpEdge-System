"""Yahoo/CBOE provider capture adapters for the Alpha Swarm live worker."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
COCKPIT = ROOT / "cockpit"
if str(COCKPIT) not in sys.path:
    sys.path.insert(0, str(COCKPIT))

import market_data_sources as market  # noqa: E402
from scripts.alpha_swarm.contracts import canonical_json  # noqa: E402
from scripts.alpha_swarm.snapshot_acquirer import ACQUISITION_SCHEMA  # noqa: E402

NY = ZoneInfo("America/New_York")


def _artifact_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _bar_timestamp(row: dict[str, Any]) -> datetime:
    midnight = datetime.fromisoformat(row["date"]).replace(tzinfo=NY)
    return midnight + timedelta(minutes=int(row["minute_of_day"]))


def _is_terminal_placeholder(bar: dict[str, Any]) -> bool:
    return (
        int(bar["volume"]) == 0
        and bar["open"] == bar["high"] == bar["low"] == bar["close"]
    )


def _normalize_price_rows(
    rows: list[dict[str, Any]], session_date: str, observed_at: datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session_rows = [row for row in rows if row.get("date") == session_date]
    raw_hash = _artifact_sha256(session_rows)
    by_timestamp: dict[str, dict[str, Any]] = {}
    dropped = 0
    for row in session_rows:
        timestamp = _bar_timestamp(row)
        if timestamp > observed_at:
            continue
        bar = {
            "timestamp": timestamp.isoformat(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        existing = by_timestamp.get(bar["timestamp"])
        if existing is None:
            by_timestamp[bar["timestamp"]] = bar
            continue
        if _is_terminal_placeholder(bar) and bar["close"] == existing["close"]:
            dropped += 1
            continue
        if _is_terminal_placeholder(existing) and existing["close"] == bar["close"]:
            by_timestamp[bar["timestamp"]] = bar
            dropped += 1
            continue
        raise ValueError(f"conflicting Yahoo bars at {bar['timestamp']}")
    bars = list(by_timestamp.values())
    if not bars:
        raise ValueError("Yahoo returned no completed bars for the locked session")
    return bars, {
        "raw_session_row_count": len(session_rows),
        "normalized_bar_count": len(bars),
        "terminal_placeholders_dropped": dropped,
        "raw_session_rows_sha256": raw_hash,
    }


def fetch_price_capture(
    symbol: str, session_date: str, *, observed_at: datetime
) -> dict[str, Any]:
    rows, source = market.fetch_yahoo_regular_session_chart_rows(symbol)
    bars, normalization = _normalize_price_rows(rows, session_date, observed_at)
    return {
        "schema": ACQUISITION_SCHEMA,
        "provider": "yahoo_chart_1m",
        "source_ref": f"yahoo://chart/{symbol}/{observed_at.isoformat()}",
        "symbol": symbol,
        "session_date": session_date,
        "observed_at": observed_at.isoformat(),
        "latest_data_ts": bars[-1]["timestamp"],
        "source_metadata": {**source, "normalization": normalization},
        "bars": bars,
    }


def _numeric(value: Any) -> float | None:
    try:
        if value in (None, "", "."):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_options_capture(
    symbol: str, session_date: str, *, observed_at: datetime
) -> dict[str, Any]:
    spot, book, source = market.fetch_cboe_options_book(symbol)
    contracts = []
    for expiration, strikes in book.items():
        for strike, sides in strikes.items():
            for code, raw in sides.items():
                bid = _numeric(raw.get("bid"))
                ask = _numeric(raw.get("ask"))
                oi = _numeric(raw.get("open_interest"))
                volume = _numeric(raw.get("volume"))
                contract_symbol = str(raw.get("option") or "").strip()
                if not contract_symbol or None in (bid, ask, oi, volume):
                    continue
                contracts.append(
                    {
                        "contract_symbol": contract_symbol,
                        "option_type": "call" if code == "C" else "put",
                        "expiration": expiration.isoformat(),
                        "strike": float(strike),
                        "bid": bid,
                        "ask": ask,
                        "quote_ts": observed_at.isoformat(),
                        "open_interest": int(oi),
                        "volume": int(volume),
                    }
                )
    if spot <= 0 or not contracts:
        raise ValueError("CBOE returned no usable delayed option chain")
    return {
        "schema": ACQUISITION_SCHEMA,
        "provider": "cboe_delayed_options_observed",
        "source_ref": f"cboe://delayed-options/{symbol}/{observed_at.isoformat()}",
        "symbol": symbol,
        "session_date": session_date,
        "observed_at": observed_at.isoformat(),
        "latest_data_ts": observed_at.isoformat(),
        "declared_feed_delay_minutes": 15,
        "source_metadata": source,
        "spot": spot,
        "contracts": contracts,
    }
