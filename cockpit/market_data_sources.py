from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from typing import Any

from http_utils import request_json_with_backoff

UA = {"User-Agent": "Mozilla/5.0"}
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
CBOE_OPTIONS_URL = (
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
)
SYM_RE = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_utc_from_epoch(epoch: Any) -> str | None:
    if epoch in (None, ""):
        return None
    try:
        return dt.datetime.fromtimestamp(float(epoch), tz=dt.UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def fetch_yahoo_chart_result(
    symbol: str = "SPY",
    *,
    interval: str,
    range_: str,
    timeout: int,
) -> dict[str, Any]:
    payload = request_json_with_backoff(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"interval": interval, "range": range_},
        headers=UA,
        timeout=timeout,
        attempts=4,
        base_sleep_seconds=1.0,
    )
    return payload["chart"]["result"][0]


def describe_yahoo_chart_source(
    result: dict[str, Any],
    *,
    symbol: str,
    interval: str,
    range_: str,
) -> dict[str, Any]:
    meta = result.get("meta", {})
    ts = result.get("timestamp") or []
    return {
        "provider": "yahoo",
        "endpoint": "chart",
        "symbol": symbol,
        "interval": interval,
        "range": range_,
        "currency": meta.get("currency"),
        "exchange_timezone_name": meta.get("exchangeTimezoneName"),
        "market_state": meta.get("marketState"),
        "chart_previous_close": _safe_float(meta.get("chartPreviousClose")),
        "regular_market_price": _safe_float(meta.get("regularMarketPrice")),
        "regular_market_time_utc": _iso_utc_from_epoch(meta.get("regularMarketTime")),
        "last_bar_utc": _iso_utc_from_epoch(ts[-1]) if ts else None,
    }


def fetch_yahoo_chart_result_with_source(
    symbol: str = "SPY",
    *,
    interval: str,
    range_: str,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = fetch_yahoo_chart_result(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    return result, describe_yahoo_chart_source(
        result,
        symbol=symbol,
        interval=interval,
        range_=range_,
    )


def fetch_yahoo_regular_session_chart_rows(
    symbol: str = "SPY",
    *,
    interval: str = "1m",
    range_: str = "1d",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result, source = fetch_yahoo_chart_result_with_source(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    meta = result.get("meta", {})
    gmt = int(meta.get("gmtoffset") or 0)
    ts = result.get("timestamp") or []
    q = result["indicators"]["quote"][0]
    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    volumes = q.get("volume") or []

    rows = []
    for idx, (epoch, close, volume) in enumerate(zip(ts, closes, volumes)):
        if close is None:
            continue
        local = dt.datetime.utcfromtimestamp(epoch + gmt)
        minute = local.hour * 60 + local.minute
        if 570 <= minute <= 960:  # regular session 09:30-16:00 ET
            open_ = opens[idx] if idx < len(opens) and opens[idx] is not None else close
            high = highs[idx] if idx < len(highs) and highs[idx] is not None else close
            low = lows[idx] if idx < len(lows) and lows[idx] is not None else close
            rows.append(
                {
                    "date": local.date().isoformat(),
                    "minute_of_day": minute,
                    "session_minute": minute - 570,
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": int(volume or 0),
                }
            )

    return rows, {**source, "bar_count": len(rows)}


def fetch_yahoo_intraday_session_rows(
    symbol: str = "SPY",
    *,
    interval: str = "1m",
    range_: str = "1d",
    timeout: int = 20,
) -> tuple[list[tuple[int, float, float, float, float, int]], dict[str, Any]]:
    regular_rows, source = fetch_yahoo_regular_session_chart_rows(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    rows = [
        (
            int(row["session_minute"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
        )
        for row in regular_rows
    ]
    return rows, source


def fetch_yahoo_prior_day_levels(
    symbol: str = "SPY",
    *,
    interval: str = "1d",
    range_: str = "5d",
    timeout: int = 15,
) -> tuple[dict[str, float], dict[str, Any]]:
    result, source = fetch_yahoo_chart_result_with_source(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    q = result["indicators"]["quote"][0]
    highs = [value for value in (q.get("high") or []) if value is not None]
    lows = [value for value in (q.get("low") or []) if value is not None]
    closes = [value for value in (q.get("close") or []) if value is not None]
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return {}, {"provider": "yahoo", "endpoint": "chart", "symbol": symbol}

    return {"PDH": highs[-2], "PDL": lows[-2], "PDC": closes[-2]}, source


def fetch_yahoo_daily_bars(
    symbol: str = "SPY",
    *,
    interval: str = "1d",
    range_: str = "2y",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result, source = fetch_yahoo_chart_result_with_source(
        symbol,
        interval=interval,
        range_=range_,
        timeout=timeout,
    )
    timestamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or [{}]
    quote = quotes[0] if quotes else {}
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    rows = []
    for index, epoch in enumerate(timestamps):
        open_ = opens[index] if index < len(opens) else None
        high = highs[index] if index < len(highs) else None
        low = lows[index] if index < len(lows) else None
        close = closes[index] if index < len(closes) else None
        if None in {open_, high, low, close}:
            continue
        volume = (
            volumes[index] if index < len(volumes) and volumes[index] is not None else 0
        )
        rows.append(
            {
                "date": dt.datetime.fromtimestamp(epoch, tz=dt.UTC).date().isoformat(),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )

    first_date = rows[0]["date"] if rows else None
    last_date = rows[-1]["date"] if rows else None
    return rows, {
        **source,
        "bar_count": len(rows),
        "first_date": first_date,
        "last_date": last_date,
    }


def read_options_surface(
    spot: float, book: dict[Any, dict[float, dict[str, dict[str, Any]]]]
) -> dict[str, Any]:
    def maybe_float(value: Any) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    if not book:
        return {
            "exp": None,
            "call_wall": None,
            "put_wall": None,
            "pcr": 0.0,
            "atm_iv": 0.0,
        }

    today = dt.date.today()
    future = sorted(expiry for expiry in book if expiry >= today) or sorted(book)
    expiry = future[0]
    strikes = sorted(book[expiry].keys())
    call_oi = {
        strike: (
            maybe_float(book[expiry][strike].get("C", {}).get("open_interest")) or 0.0
        )
        for strike in strikes
    }
    put_oi = {
        strike: (
            maybe_float(book[expiry][strike].get("P", {}).get("open_interest")) or 0.0
        )
        for strike in strikes
    }
    call_volume = {
        strike: maybe_float(book[expiry][strike].get("C", {}).get("volume")) or 0.0
        for strike in strikes
    }
    put_volume = {
        strike: maybe_float(book[expiry][strike].get("P", {}).get("volume")) or 0.0
        for strike in strikes
    }

    calls_above = {strike: value for strike, value in call_oi.items() if strike >= spot}
    puts_below = {strike: value for strike, value in put_oi.items() if strike <= spot}
    call_wall = max(calls_above, key=calls_above.get) if calls_above else None
    put_wall = max(puts_below, key=puts_below.get) if puts_below else None
    total_call_oi = sum(call_oi.values()) or 1.0
    total_put_oi = sum(put_oi.values())
    total_call_volume = sum(call_volume.values())
    total_put_volume = sum(put_volume.values())

    atm = min(strikes, key=lambda strike: abs(strike - spot))
    atm_call = book[expiry][atm].get("C", {})
    atm_put = book[expiry][atm].get("P", {})
    atm_call_iv = maybe_float(atm_call.get("iv"))
    atm_put_iv = maybe_float(atm_put.get("iv"))
    atm_iv_values = [value for value in (atm_call_iv, atm_put_iv) if value is not None]
    atm_call_bid = maybe_float(atm_call.get("bid"))
    atm_call_ask = maybe_float(atm_call.get("ask"))
    atm_put_bid = maybe_float(atm_put.get("bid"))
    atm_put_ask = maybe_float(atm_put.get("ask"))
    call_mid = (
        (atm_call_bid + atm_call_ask) / 2
        if atm_call_bid is not None and atm_call_ask is not None
        else None
    )
    put_mid = (
        (atm_put_bid + atm_put_ask) / 2
        if atm_put_bid is not None and atm_put_ask is not None
        else None
    )
    atm_call_spread = (
        atm_call_ask - atm_call_bid
        if atm_call_bid is not None and atm_call_ask is not None
        else None
    )
    atm_put_spread = (
        atm_put_ask - atm_put_bid
        if atm_put_bid is not None and atm_put_ask is not None
        else None
    )

    return {
        "exp": expiry.isoformat(),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_volume_wall": max(call_volume, key=call_volume.get)
        if call_volume
        else None,
        "put_volume_wall": max(put_volume, key=put_volume.get) if put_volume else None,
        "pcr": total_put_oi / total_call_oi,
        "pcvr": total_put_volume / (total_call_volume or 1.0),
        "call_volume_total": total_call_volume,
        "put_volume_total": total_put_volume,
        "atm_strike": atm,
        "atm_iv": sum(atm_iv_values) / len(atm_iv_values) if atm_iv_values else 0.0,
        "atm_call_iv": atm_call_iv,
        "atm_put_iv": atm_put_iv,
        "atm_iv_skew": (
            atm_put_iv - atm_call_iv
            if atm_put_iv is not None and atm_call_iv is not None
            else None
        ),
        "atm_call_delta": maybe_float(atm_call.get("delta")),
        "atm_put_delta": maybe_float(atm_put.get("delta")),
        "atm_call_theta": maybe_float(atm_call.get("theta")),
        "atm_put_theta": maybe_float(atm_put.get("theta")),
        "atm_call_vega": maybe_float(atm_call.get("vega")),
        "atm_put_vega": maybe_float(atm_put.get("vega")),
        "atm_call_rho": maybe_float(atm_call.get("rho")),
        "atm_put_rho": maybe_float(atm_put.get("rho")),
        "atm_call_theo": maybe_float(atm_call.get("theo")),
        "atm_put_theo": maybe_float(atm_put.get("theo")),
        "atm_call_last_trade_price": maybe_float(atm_call.get("last_trade_price")),
        "atm_put_last_trade_price": maybe_float(atm_put.get("last_trade_price")),
        "atm_call_bid": atm_call_bid,
        "atm_call_ask": atm_call_ask,
        "atm_put_bid": atm_put_bid,
        "atm_put_ask": atm_put_ask,
        "atm_call_spread": atm_call_spread,
        "atm_put_spread": atm_put_spread,
        "atm_call_spread_pct": (
            atm_call_spread / call_mid
            if atm_call_spread is not None and call_mid
            else None
        ),
        "atm_put_spread_pct": (
            atm_put_spread / put_mid if atm_put_spread is not None and put_mid else None
        ),
        "atm_straddle_mid": (
            call_mid + put_mid if call_mid is not None and put_mid is not None else None
        ),
    }


def fetch_cboe_options_book(
    symbol: str = "SPY",
    *,
    timeout: int = 30,
) -> tuple[float, dict[Any, dict[float, dict[str, dict[str, Any]]]], dict[str, Any]]:
    data = request_json_with_backoff(
        CBOE_OPTIONS_URL.format(symbol=symbol),
        headers=UA,
        timeout=timeout,
        attempts=3,
        base_sleep_seconds=1.0,
    )["data"]
    options = data.get("options") or []
    book: dict[Any, dict[float, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    latest_option_trade = None
    for option in options:
        match = SYM_RE.match(option.get("option", ""))
        if not match:
            continue
        yymmdd, cp, strike8 = match.groups()
        expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
        book[expiry][int(strike8) / 1000.0][cp] = option
        raw_trade_time = option.get("last_trade_time")
        if raw_trade_time and (
            latest_option_trade is None or raw_trade_time > latest_option_trade
        ):
            latest_option_trade = raw_trade_time

    spot = (
        _safe_float(data.get("current_price")) or _safe_float(data.get("close")) or 0.0
    )
    source = {
        "provider": "cboe",
        "endpoint": "delayed_quotes/options",
        "symbol": symbol,
        "option_count": len(options),
        "current_price": _safe_float(data.get("current_price")),
        "close": _safe_float(data.get("close")),
        "open": _safe_float(data.get("open")),
        "high": _safe_float(data.get("high")),
        "low": _safe_float(data.get("low")),
        "prev_day_close": _safe_float(data.get("prev_day_close")),
        "bid": _safe_float(data.get("bid")),
        "ask": _safe_float(data.get("ask")),
        "bid_size": _safe_float(data.get("bid_size")),
        "ask_size": _safe_float(data.get("ask_size")),
        "price_change": _safe_float(data.get("price_change")),
        "price_change_percent": _safe_float(data.get("price_change_percent")),
        "iv30": _safe_float(data.get("iv30")),
        "iv30_change": _safe_float(data.get("iv30_change")),
        "iv30_change_percent": _safe_float(data.get("iv30_change_percent")),
        "last_trade_time_raw": data.get("last_trade_time") or latest_option_trade,
        "latest_option_trade_time_raw": latest_option_trade,
    }
    return spot, book, source
