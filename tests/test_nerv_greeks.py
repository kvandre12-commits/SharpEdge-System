from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.greeks import estimate_option_greeks, time_to_expiration_years  # noqa: E402



def test_time_to_expiration_stays_positive_on_expiry_day() -> None:
    years = time_to_expiration_years(
        "2026-01-16",
        "2026-01-16T15:00:00+00:00",
    )

    assert years is not None
    assert years > 0



def test_black_scholes_estimator_returns_sane_call_greeks() -> None:
    greeks = estimate_option_greeks(
        underlying="XYZ",
        option_type="call",
        spot=100.0,
        strike=100.0,
        implied_volatility=0.20,
        expiration="2026-01-31",
        as_of="2026-01-01T15:30:00+00:00",
    )

    assert greeks is not None
    assert 0.5 < greeks["delta"] < 0.6
    assert 0.06 < greeks["gamma"] < 0.08
    assert -0.05 < greeks["theta"] < -0.03
    assert 0.10 < greeks["vega"] < 0.13



def test_black_scholes_put_delta_is_negative() -> None:
    greeks = estimate_option_greeks(
        underlying="XYZ",
        option_type="put",
        spot=100.0,
        strike=100.0,
        implied_volatility=0.20,
        expiration="2026-01-31",
        as_of="2026-01-01T15:30:00+00:00",
    )

    assert greeks is not None
    assert -0.5 < greeks["delta"] < -0.4
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["vega"] > 0
