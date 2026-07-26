"""Quote-quality and liquidity scoring for NERV research rows.

The score answers: "is this contract usable enough to inspect manually?" It does
not answer: "is this a good trade?" Tiny distinction, massive lawsuits avoided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreConfig:
    min_bid: float = 0.01
    max_quote_age_seconds: int = 24 * 60 * 60
    ideal_width_pct: float = 0.05
    max_width_pct: float = 0.25
    target_volume: int = 500
    target_open_interest: int = 1000
    min_open_interest: int = 50


DEFAULT_SCORE_CONFIG = ScoreConfig()


def score_quote_record(
    record: dict[str, Any],
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> dict[str, Any]:
    bid = _as_float(record.get("bid"))
    ask = _as_float(record.get("ask"))
    midpoint = _as_float(record.get("midpoint"))
    if midpoint is None and bid is not None and ask is not None and ask >= bid:
        midpoint = (bid + ask) / 2
    width = _as_float(record.get("bid_ask_width"))
    if width is None and bid is not None and ask is not None:
        width = max(ask - bid, 0.0)
    volume = _as_int(record.get("volume")) or 0
    open_interest = _as_int(record.get("open_interest")) or 0
    age = _as_int(record.get("quote_age_seconds"))

    flags = _rejection_flags(
        bid=bid,
        ask=ask,
        midpoint=midpoint,
        volume=volume,
        open_interest=open_interest,
        age=age,
        config=config,
    )
    width_pct = _width_pct(width, midpoint)

    quote_score = _quote_quality_score(
        bid=bid,
        ask=ask,
        width_pct=width_pct,
        age=age,
        config=config,
    )
    liquidity_score = _liquidity_score(
        volume=volume,
        open_interest=open_interest,
        config=config,
    )
    total_score = round((quote_score * 0.55) + (liquidity_score * 0.45), 2)
    if flags:
        total_score = min(total_score, 49.0)

    return {
        "quote_quality_score": round(quote_score, 2),
        "liquidity_score": round(liquidity_score, 2),
        "nerv_score": total_score,
        "width_pct": round(width_pct, 6) if width_pct is not None else None,
        "manual_validation_priority": _priority(total_score, flags),
        "rejection_flags": ";".join(flags),
        "fresh_quote_required": True,
    }


def enrich_quote_record(
    record: dict[str, Any],
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> dict[str, Any]:
    enriched = dict(record)
    enriched.update(score_quote_record(record, config=config))
    return enriched


def build_liquidity_board(
    records: list[dict[str, Any]],
    *,
    limit: int = 50,
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> list[dict[str, Any]]:
    enriched = [enrich_quote_record(record, config=config) for record in records]
    enriched.sort(
        key=lambda row: (
            row.get("nerv_score") or 0,
            row.get("volume") or 0,
            row.get("open_interest") or 0,
        ),
        reverse=True,
    )
    if limit <= 0:
        return enriched
    return enriched[:limit]


def _rejection_flags(
    *,
    bid: float | None,
    ask: float | None,
    midpoint: float | None,
    volume: int,
    open_interest: int,
    age: int | None,
    config: ScoreConfig,
) -> list[str]:
    flags: list[str] = []
    if bid is None or ask is None:
        flags.append("missing_bid_ask")
    elif ask < bid:
        flags.append("crossed_market")
    elif bid < config.min_bid and ask <= config.min_bid:
        flags.append("zero_or_tiny_market")
    if midpoint in (None, 0):
        flags.append("missing_midpoint")
    if open_interest < config.min_open_interest:
        flags.append("thin_open_interest")
    if age is None:
        flags.append("missing_quote_age")
    elif age > config.max_quote_age_seconds:
        flags.append("stale_quote")
    if volume <= 0 and open_interest <= 0:
        flags.append("no_activity")
    return flags


def _quote_quality_score(
    *,
    bid: float | None,
    ask: float | None,
    width_pct: float | None,
    age: int | None,
    config: ScoreConfig,
) -> float:
    if bid is None or ask is None or ask < bid:
        return 0.0

    if width_pct is None:
        width_component = 10.0
    elif width_pct <= config.ideal_width_pct:
        width_component = 65.0
    elif width_pct >= config.max_width_pct:
        width_component = 15.0
    else:
        span = config.max_width_pct - config.ideal_width_pct
        decay = (width_pct - config.ideal_width_pct) / span
        width_component = 65.0 - (50.0 * decay)

    if age is None:
        age_component = 10.0
    elif age <= 15 * 60:
        age_component = 35.0
    elif age <= config.max_quote_age_seconds:
        age_component = 25.0
    else:
        age_component = 5.0

    return min(width_component + age_component, 100.0)


def _liquidity_score(*, volume: int, open_interest: int, config: ScoreConfig) -> float:
    volume_component = min(volume / config.target_volume, 1.0) * 45.0
    oi_component = min(open_interest / config.target_open_interest, 1.0) * 55.0
    return min(volume_component + oi_component, 100.0)


def _priority(score: float, flags: list[str]) -> str:
    fatal_flags = {"missing_bid_ask", "crossed_market", "missing_midpoint", "no_activity"}
    if any(flag in flags for flag in fatal_flags):
        return "reject"
    if "stale_quote" in flags:
        return "refresh"
    if score >= 80:
        return "high"
    if score >= 65:
        return "medium"
    if score >= 50:
        return "low"
    return "reject"


def _width_pct(width: float | None, midpoint: float | None) -> float | None:
    if width is None or midpoint in (None, 0):
        return None
    return max(width / abs(midpoint), 0.0)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
