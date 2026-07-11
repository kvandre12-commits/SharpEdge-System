"""Default weight doctrine for execution-vector scoring."""

from __future__ import annotations

DEFAULT_BASE_WEIGHTS = {
    "structure_score": 0.15,
    "trend_score": 0.15,
    "acceptance_score": 0.15,
    "volume_score": 0.10,
    "location_score": 0.15,
    "pressure_score": 0.08,  # Reduced base weight (was likely higher)
    "time_of_day_score": 0.05,
    "volatility_score": 0.05,
    "opening_auction_score": 0.10,
    "exhaustion_score": 0.07,
    "dealer_gamma_score": 0.05,
    "regime_score": 0.12,
    "compression_score": 0.07,
    "balance_context_score": 0.08,
}

DEFAULT_BASE_BIAS_WEIGHTS = {
    "structure_score": 0.10,
    "acceptance_score": 0.12,
    "rejection_score": 0.10,
    "trend_score": 0.14,
    "pressure_score": 0.05,
    "opening_auction_score": 0.04,
    "exhaustion_score": 0.07,
    "trap_score": 0.09,
    "dealer_gamma_score": 0.06,
    "regime_score": 0.08,
    "compression_score": 0.15,
    "balance_context_score": 0.12,
}

__all__ = ["DEFAULT_BASE_BIAS_WEIGHTS", "DEFAULT_BASE_WEIGHTS"]
