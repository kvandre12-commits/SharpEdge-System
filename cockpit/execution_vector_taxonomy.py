"""Execution-vector evidence taxonomy.

Category says what role a vector part plays in the reasoning stack.
Correlation family says which market phenomenon it primarily measures.
Overlap families make known correlation risk explicit.

This module is intentionally metadata-only: it must not score, gate, cap,
or override trade permission. Its job is to make score-soup risk visible.
"""

from __future__ import annotations

from typing import TypedDict


class VectorPartTaxonomy(TypedDict):
    category: str
    correlation_family: str
    overlap_families: tuple[str, ...]
    note: str


CORE_STRUCTURAL = "core_structural"
TACTICAL_CONFIRMATION = "tactical_confirmation"
CONTEXT_GOVERNOR = "context_governor"
SUSPECT_DRIFT_VOICE = "suspect_drift_voice"
ADVISORY_SURFACE = "advisory_surface"

AUCTION = "auction"
BALANCE = "balance"
DEALER_POSITIONING = "dealer_positioning"
LOCATION = "location"
MOMENTUM = "momentum"
PARTICIPATION = "participation"
PRICE_STRUCTURE = "price_structure"
SESSION_CONTEXT = "session_context"
STRETCH = "stretch"
TACTICAL_CANDLE = "tactical_candle"
VOLATILITY = "volatility"
EXPANSION_FUEL = "expansion_fuel"

VECTOR_PART_TAXONOMY: dict[str, VectorPartTaxonomy] = {
    "structure_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": PRICE_STRUCTURE,
        "overlap_families": (MOMENTUM,),
        "note": "Swing/sequence structure; slower battlefield evidence.",
    },
    "acceptance_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": AUCTION,
        "overlap_families": (),
        "note": "Multi-close acceptance around reference levels.",
    },
    "trend_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": MOMENTUM,
        "overlap_families": (PRICE_STRUCTURE,),
        "note": "VWAP/momentum alignment; core but correlated with regime/pressure.",
    },
    "location_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": LOCATION,
        "overlap_families": (BALANCE, MOMENTUM, STRETCH),
        "note": "Distance/position relative to important levels and balance state.",
    },
    "volume_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": PARTICIPATION,
        "overlap_families": (MOMENTUM,),
        "note": "Whether participation confirms or refuses the move; not a full explanation of why price can travel.",
    },
    "time_of_day_score": {
        "category": CONTEXT_GOVERNOR,
        "correlation_family": SESSION_CONTEXT,
        "overlap_families": (AUCTION,),
        "note": "Session window quality for execution follow-through.",
    },
    "dealer_gamma_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": DEALER_POSITIONING,
        "overlap_families": (VOLATILITY,),
        "note": "Gamma regime, pin, and wall context.",
    },
    "trap_score": {
        "category": TACTICAL_CONFIRMATION,
        "correlation_family": AUCTION,
        "overlap_families": (),
        "note": "Failed break/trap evidence; correlated with acceptance/rejection.",
    },
    "rejection_score": {
        "category": TACTICAL_CONFIRMATION,
        "correlation_family": AUCTION,
        "overlap_families": (TACTICAL_CANDLE,),
        "note": "Immediate wick rejection evidence at price extremes or levels.",
    },
    "opening_auction_score": {
        "category": CONTEXT_GOVERNOR,
        "correlation_family": AUCTION,
        "overlap_families": (SESSION_CONTEXT,),
        "note": "Opening range behavior; decays after the auction window.",
    },
    "exhaustion_score": {
        "category": CONTEXT_GOVERNOR,
        "correlation_family": STRETCH,
        "overlap_families": (LOCATION, MOMENTUM, TACTICAL_CANDLE),
        "note": "Overextension context that can govern chasing risk.",
    },
    "balance_context_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": BALANCE,
        "overlap_families": (LOCATION,),
        "note": "Core adjunct for balance/value confluence or disagreement; overlaps location, so keep the weight modest.",
    },
    "volatility_score": {
        "category": CONTEXT_GOVERNOR,
        "correlation_family": VOLATILITY,
        "overlap_families": (DEALER_POSITIONING,),
        "note": "IV/premium context for follow-through versus whipsaw risk.",
    },
    "compression_score": {
        "category": CONTEXT_GOVERNOR,
        "correlation_family": VOLATILITY,
        "overlap_families": (),
        "note": "Coil/compression context for potential expansion.",
    },
    "pressure_score": {
        "category": CORE_STRUCTURAL,
        "correlation_family": MOMENTUM,
        "overlap_families": (PARTICIPATION, TACTICAL_CANDLE),
        "note": "Core adjunct for short-horizon follow-through pressure; damped when it merely echoes trend.",
    },
    "expansion_fuel_score": {
        "category": ADVISORY_SURFACE,
        "correlation_family": EXPANSION_FUEL,
        "overlap_families": (
            PARTICIPATION,
            DEALER_POSITIONING,
            AUCTION,
            VOLATILITY,
            MOMENTUM,
        ),
        "note": "Advisory read for why price can still travel even when participation confirmation and fuel diverge.",
    },
    "line_authority_score": {
        "category": ADVISORY_SURFACE,
        "correlation_family": LOCATION,
        "overlap_families": (PRICE_STRUCTURE, BALANCE, AUCTION),
        "note": "Advisory read of how the latest candle interacts with visible reference rails (VWAP/OR/PD/balance lines); overlaps location and structure, so keep it advisory-only.",
    },
    "regime_score": {
        "category": SUSPECT_DRIFT_VOICE,
        "correlation_family": MOMENTUM,
        "overlap_families": (LOCATION, PARTICIPATION, BALANCE),
        "note": "Broad session regime; damped when it merely echoes trend.",
    },
}


__all__ = [
    "ADVISORY_SURFACE",
    "CONTEXT_GOVERNOR",
    "CORE_STRUCTURAL",
    "SUSPECT_DRIFT_VOICE",
    "TACTICAL_CONFIRMATION",
    "VECTOR_PART_TAXONOMY",
    "VectorPartTaxonomy",
]
