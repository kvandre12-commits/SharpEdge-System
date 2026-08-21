"""Local Black-Scholes Greek estimation for research-only option snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, time
from math import erf, exp, log, pi, sqrt
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

DEFAULT_RISK_FREE_RATE = 0.045
DIVIDEND_YIELD_BY_UNDERLYING = {
    "SPY": 0.012,
    "QQQ": 0.006,
    "IWM": 0.012,
    "DIA": 0.016,
}
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0



def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt_value = value
    else:
        try:
            dt_value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC)



def _expiration_close_utc(expiration: str) -> datetime | None:
    try:
        expiry_date = datetime.fromisoformat(expiration).date()
    except ValueError:
        return None
    if ZoneInfo is None:
        return datetime.combine(expiry_date, time(21, 0), tzinfo=UTC)
    ny_close = datetime.combine(
        expiry_date,
        time(16, 0),
        tzinfo=ZoneInfo("America/New_York"),
    )
    return ny_close.astimezone(UTC)



def time_to_expiration_years(expiration: str, as_of: str | datetime | None) -> float | None:
    as_of_dt = _parse_datetime(as_of)
    expiry_dt = _expiration_close_utc(expiration)
    if as_of_dt is None or expiry_dt is None:
        return None
    seconds = max((expiry_dt - as_of_dt).total_seconds(), 60.0)
    return seconds / SECONDS_PER_YEAR



def _normal_pdf(value: float) -> float:
    return exp(-(value**2) / 2.0) / sqrt(2.0 * pi)



def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))



def estimate_option_greeks(
    *,
    underlying: str,
    option_type: str,
    spot: float | None,
    strike: float | None,
    implied_volatility: float | None,
    expiration: str,
    as_of: str | datetime | None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float | None = None,
) -> dict[str, float] | None:
    if spot in (None, 0) or strike in (None, 0) or implied_volatility in (None, 0):
        return None
    spot_value = float(spot)
    strike_value = float(strike)
    sigma = float(implied_volatility)
    if spot_value <= 0 or strike_value <= 0 or sigma <= 0:
        return None
    years = time_to_expiration_years(expiration, as_of)
    if years is None or years <= 0:
        return None

    carry = (
        dividend_yield
        if dividend_yield is not None
        else DIVIDEND_YIELD_BY_UNDERLYING.get(str(underlying).upper(), 0.0)
    )
    root_t = sqrt(years)
    vol_term = sigma * root_t
    if vol_term <= 0:
        return None
    d1 = (
        log(spot_value / strike_value)
        + (risk_free_rate - carry + 0.5 * sigma * sigma) * years
    ) / vol_term
    d2 = d1 - vol_term
    discount_div = exp(-carry * years)
    discount_rf = exp(-risk_free_rate * years)
    pdf_d1 = _normal_pdf(d1)

    if str(option_type).lower() == "put":
        delta = discount_div * (_normal_cdf(d1) - 1.0)
        theta_annual = (
            -(spot_value * discount_div * pdf_d1 * sigma) / (2.0 * root_t)
            + risk_free_rate * strike_value * discount_rf * _normal_cdf(-d2)
            - carry * spot_value * discount_div * _normal_cdf(-d1)
        )
    else:
        delta = discount_div * _normal_cdf(d1)
        theta_annual = (
            -(spot_value * discount_div * pdf_d1 * sigma) / (2.0 * root_t)
            - risk_free_rate * strike_value * discount_rf * _normal_cdf(d2)
            + carry * spot_value * discount_div * _normal_cdf(d1)
        )

    gamma = discount_div * pdf_d1 / (spot_value * vol_term)
    vega = spot_value * discount_div * pdf_d1 * root_t / 100.0
    theta = theta_annual / 365.0
    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega": round(vega, 6),
    }



def merge_observed_and_estimated_greeks(
    *,
    underlying: str,
    option_type: str,
    spot: float | None,
    strike: float | None,
    implied_volatility: float | None,
    expiration: str,
    as_of: str | datetime | None,
    observed_delta: Any = None,
    observed_gamma: Any = None,
    observed_theta: Any = None,
    observed_vega: Any = None,
) -> tuple[dict[str, float | None], str | None]:
    observed = {
        "delta": None if observed_delta in (None, "") else float(observed_delta),
        "gamma": None if observed_gamma in (None, "") else float(observed_gamma),
        "theta": None if observed_theta in (None, "") else float(observed_theta),
        "vega": None if observed_vega in (None, "") else float(observed_vega),
    }
    missing = [name for name, value in observed.items() if value is None]
    estimated = (
        estimate_option_greeks(
            underlying=underlying,
            option_type=option_type,
            spot=spot,
            strike=strike,
            implied_volatility=implied_volatility,
            expiration=expiration,
            as_of=as_of,
        )
        if missing
        else None
    )
    merged = {
        name: observed[name] if observed[name] is not None else (estimated or {}).get(name)
        for name in ("delta", "gamma", "theta", "vega")
    }
    observed_count = sum(value is not None for value in observed.values())
    estimated_count = sum(value is not None for value in merged.values()) - observed_count
    if observed_count and estimated_count:
        return merged, "mixed"
    if observed_count:
        return merged, "observed"
    if estimated_count:
        return merged, "estimated"
    return merged, None


__all__ = [
    "estimate_option_greeks",
    "merge_observed_and_estimated_greeks",
    "time_to_expiration_years",
]
