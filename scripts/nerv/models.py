"""Typed records for NERV option-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


RESEARCH_ONLY_WARNING = (
    "Research-only free/public data. Confirm final bid/ask, size, buying-power "
    "effect, assignment/dividend status, and order details at the execution broker."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN, tiny goblin.
        return None
    return parsed


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed


@dataclass(frozen=True)
class NERVOptionQuote:
    underlying: str
    contract_symbol: str
    option_type: str
    expiration: str
    strike: float
    source: str
    data_mode: str
    fetch_timestamp: str
    quote_timestamp: str | None = None
    underlying_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    greeks_source: str | None = None
    in_the_money: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 and self.ask <= 0:
            return None
        return round((self.bid + self.ask) / 2, 6)

    @property
    def bid_ask_width(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return round(max(self.ask - self.bid, 0), 6)

    @property
    def moneyness(self) -> float | None:
        if self.underlying_price in (None, 0):
            return None
        return round(self.strike / self.underlying_price, 6)

    @property
    def quote_age_seconds(self) -> int | None:
        quote_dt = parse_datetime(self.quote_timestamp)
        fetch_dt = parse_datetime(self.fetch_timestamp)
        if not quote_dt or not fetch_dt:
            return None
        return max(int((fetch_dt - quote_dt).total_seconds()), 0)

    def to_record(self, include_raw: bool = False) -> dict[str, Any]:
        record = {
            "underlying": self.underlying,
            "contract_symbol": self.contract_symbol,
            "option_type": self.option_type,
            "expiration": self.expiration,
            "strike": self.strike,
            "underlying_price": self.underlying_price,
            "bid": self.bid,
            "ask": self.ask,
            "midpoint": self.midpoint,
            "bid_ask_width": self.bid_ask_width,
            "last": self.last,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "greeks_source": self.greeks_source,
            "in_the_money": self.in_the_money,
            "moneyness": self.moneyness,
            "source": self.source,
            "data_mode": self.data_mode,
            "quote_timestamp": self.quote_timestamp,
            "fetch_timestamp": self.fetch_timestamp,
            "quote_age_seconds": self.quote_age_seconds,
            "research_only_warning": RESEARCH_ONLY_WARNING,
        }
        if include_raw:
            record["raw"] = self.raw
        return record


@dataclass(frozen=True)
class NERVSnapshot:
    symbols: list[str]
    source: str
    data_mode: str
    fetch_timestamp: str
    quotes: list[NERVOptionQuote]
    errors: list[dict[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        by_symbol: dict[str, int] = {}
        expirations: set[str] = set()
        for quote in self.quotes:
            by_symbol[quote.underlying] = by_symbol.get(quote.underlying, 0) + 1
            expirations.add(quote.expiration)
        return {
            "source": self.source,
            "data_mode": self.data_mode,
            "fetch_timestamp": self.fetch_timestamp,
            "requested_symbols": self.symbols,
            "quote_count": len(self.quotes),
            "symbols_with_quotes": sorted(by_symbol),
            "quotes_by_symbol": by_symbol,
            "expiration_count": len(expirations),
            "error_count": len(self.errors),
            "errors": self.errors,
            "research_only_warning": RESEARCH_ONLY_WARNING,
        }

    def to_payload(self, include_raw: bool = False) -> dict[str, Any]:
        return {
            "schema": "sharpedge.nerv_options_snapshot.v1",
            "summary": self.summary(),
            "quotes": [quote.to_record(include_raw=include_raw) for quote in self.quotes],
        }
