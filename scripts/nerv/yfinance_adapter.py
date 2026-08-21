"""yfinance-backed NERV discovery adapter.

This is a broad discovery/fallback adapter, not a source of record. Yahoo data
via yfinance is unofficial and intended for personal research use.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .greeks import merge_observed_and_estimated_greeks
from .models import NERVOptionQuote, NERVSnapshot, safe_float, safe_int, utc_now_iso
from .symbols import normalize_underlying, parse_occ_symbol

SOURCE = "yfinance"
DATA_MODE = "unofficial_yahoo_personal_research_delayed_or_unknown"


class YFinanceOptionsAdapter:
    def __init__(self, ticker_factory: Any | None = None) -> None:
        self._ticker_factory = ticker_factory

    def fetch(
        self,
        symbols: Iterable[str],
        *,
        max_expirations: int | None = 2,
        expirations: Iterable[str] | None = None,
    ) -> NERVSnapshot:
        fetch_ts = utc_now_iso()
        normalized_symbols = [normalize_underlying(symbol) for symbol in symbols if symbol]
        requested_expirations = list(expirations or [])
        quotes: list[NERVOptionQuote] = []
        errors: list[dict[str, str]] = []

        for symbol in normalized_symbols:
            try:
                ticker = self._build_ticker(symbol)
                underlying_price = _extract_underlying_price(ticker)
                selected_expirations = _select_expirations(
                    ticker,
                    requested_expirations=requested_expirations,
                    max_expirations=max_expirations,
                )
                for expiration in selected_expirations:
                    chain = ticker.option_chain(expiration)
                    quotes.extend(
                        _normalize_option_frame(
                            frame=chain.calls,
                            underlying=symbol,
                            option_type="call",
                            expiration=expiration,
                            underlying_price=underlying_price,
                            fetch_ts=fetch_ts,
                        )
                    )
                    quotes.extend(
                        _normalize_option_frame(
                            frame=chain.puts,
                            underlying=symbol,
                            option_type="put",
                            expiration=expiration,
                            underlying_price=underlying_price,
                            fetch_ts=fetch_ts,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - vendor discovery is best-effort
                errors.append({"symbol": symbol, "source": SOURCE, "error": str(exc)})

        return NERVSnapshot(
            symbols=normalized_symbols,
            source=SOURCE,
            data_mode=DATA_MODE,
            fetch_timestamp=fetch_ts,
            quotes=quotes,
            errors=errors,
        )

    def _build_ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # lazy import: tests and non-yfinance paths stay light

        return yf.Ticker(symbol)


def _select_expirations(
    ticker: Any,
    *,
    requested_expirations: list[str],
    max_expirations: int | None,
) -> list[str]:
    available = list(getattr(ticker, "options", []) or [])
    if requested_expirations:
        available_set = set(available)
        return [expiry for expiry in requested_expirations if expiry in available_set]
    if max_expirations is None or max_expirations <= 0:
        return available
    return available[:max_expirations]


def _extract_underlying_price(ticker: Any) -> float | None:
    fast_info = getattr(ticker, "fast_info", None)
    for attr in ("last_price", "regular_market_price", "previous_close"):
        value = _read_obj_or_dict(fast_info, attr)
        parsed = safe_float(value)
        if parsed is not None:
            return parsed

    info = getattr(ticker, "info", None)
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        value = _read_obj_or_dict(info, key)
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _read_obj_or_dict(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_option_frame(
    *,
    frame: Any,
    underlying: str,
    option_type: str,
    expiration: str,
    underlying_price: float | None,
    fetch_ts: str,
) -> list[NERVOptionQuote]:
    rows = [] if frame is None else frame.to_dict("records")
    quotes: list[NERVOptionQuote] = []
    for row in rows:
        symbol = str(row.get("contractSymbol") or "").strip()
        if not symbol:
            continue
        parsed = parse_occ_symbol(symbol)
        strike = safe_float(row.get("strike"))
        if parsed and strike is None:
            strike = safe_float(parsed["strike"])
        if strike is None:
            continue
        quote_timestamp = _timestamp_to_iso(row.get("lastTradeDate"))
        implied_volatility = safe_float(row.get("impliedVolatility"))
        greeks, greeks_source = merge_observed_and_estimated_greeks(
            underlying=underlying,
            option_type=option_type,
            spot=underlying_price,
            strike=strike,
            implied_volatility=implied_volatility,
            expiration=expiration,
            as_of=fetch_ts,
            observed_delta=safe_float(row.get("delta")),
            observed_gamma=safe_float(row.get("gamma")),
            observed_theta=safe_float(row.get("theta")),
            observed_vega=safe_float(row.get("vega")),
        )
        quotes.append(
            NERVOptionQuote(
                underlying=underlying,
                contract_symbol=symbol,
                option_type=option_type,
                expiration=expiration,
                strike=strike,
                underlying_price=underlying_price,
                bid=safe_float(row.get("bid")),
                ask=safe_float(row.get("ask")),
                last=safe_float(row.get("lastPrice")),
                volume=safe_int(row.get("volume")),
                open_interest=safe_int(row.get("openInterest")),
                implied_volatility=implied_volatility,
                delta=greeks.get("delta"),
                gamma=greeks.get("gamma"),
                theta=greeks.get("theta"),
                vega=greeks.get("vega"),
                greeks_source=greeks_source,
                in_the_money=_safe_bool(row.get("inTheMoney")),
                source=SOURCE,
                data_mode=DATA_MODE,
                quote_timestamp=quote_timestamp,
                fetch_timestamp=fetch_ts,
                raw=row,
            )
        )
    return quotes


def _timestamp_to_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _safe_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)
