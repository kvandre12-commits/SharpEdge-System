#!/usr/bin/env python3
"""Build historical refill-stack interaction surfaces from event CSV output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class CanonicalEvent:
    mode: str
    threshold_pct: float
    event_date: date
    prior_close: float
    move_pct: float
    filled: bool
    fill_date: date | None
    trading_days_to_fill: int | None
    ema200_side: str
    ema200_distance_bucket: str
    ema200_distance_pct: float | None


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_date(value: str | None) -> date | None:
    if value in (None, ""):
        return None
    return date.fromisoformat(value[:10])


def _as_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _read_events(path: Path, mode: str) -> list[CanonicalEvent]:
    raw: list[CanonicalEvent] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if mode != "all" and row.get("mode") != mode:
                continue
            event_date = _as_date(row.get("event_date"))
            threshold = _as_float(row.get("threshold_pct"))
            prior_close = _as_float(row.get("prior_close"))
            move_pct = _as_float(row.get("move_pct"))
            if event_date is None or threshold is None or prior_close is None:
                continue
            raw.append(
                CanonicalEvent(
                    mode=str(row.get("mode") or "unknown"),
                    threshold_pct=threshold,
                    event_date=event_date,
                    prior_close=prior_close,
                    move_pct=(move_pct or 0.0) * 100.0,
                    filled=str(row.get("filled") or "").lower() == "true",
                    fill_date=_as_date(row.get("fill_date")),
                    trading_days_to_fill=_as_int(row.get("trading_days_to_fill")),
                    ema200_side=row.get("ema200_side") or "unknown",
                    ema200_distance_bucket=row.get("ema200_distance_bucket")
                    or "unknown",
                    ema200_distance_pct=_as_float(row.get("ema200_distance_pct")),
                )
            )
    return _canonicalize(raw)


def _canonicalize(events: list[CanonicalEvent]) -> list[CanonicalEvent]:
    strongest: dict[tuple[str, date], CanonicalEvent] = {}
    for event in events:
        key = (event.mode, event.event_date)
        current = strongest.get(key)
        if current is None or event.threshold_pct > current.threshold_pct:
            strongest[key] = event
    return sorted(strongest.values(), key=lambda event: event.event_date)


def _signature(events: list[CanonicalEvent]) -> str:
    return "+".join(
        f"{event.mode}:{event.threshold_pct:g}:{event.ema200_distance_bucket}"
        for event in events
    )


def _event_packet(event: CanonicalEvent) -> dict[str, Any]:
    return {
        "mode": event.mode,
        "threshold_pct": event.threshold_pct,
        "event_date": event.event_date.isoformat(),
        "target": round(event.prior_close, 2),
        "move_pct": round(event.move_pct, 2),
        "filled": event.filled,
        "fill_date": event.fill_date.isoformat() if event.fill_date else None,
        "trading_days_to_fill": event.trading_days_to_fill,
        "ema200_side": event.ema200_side,
        "ema200_distance_bucket": event.ema200_distance_bucket,
        "ema200_distance_pct": event.ema200_distance_pct,
    }


def _stack_refs(events: list[CanonicalEvent]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        open_previous = [
            previous
            for previous in events[:index]
            if previous.fill_date is None or previous.fill_date > event.event_date
        ]
        if not open_previous:
            continue

        stack = [*open_previous, event]
        fill_dates = [item.fill_date for item in stack]
        full_stack_resolved = all(fill_dates)
        resolved_date = max(fill_dates) if full_stack_resolved else None
        previous = open_previous[-1]
        refs.append(
            {
                "event_date": event.event_date.isoformat(),
                "depth": len(stack),
                "signature": _signature(stack),
                "last_pair_bucket": (
                    f"{previous.ema200_distance_bucket} -> "
                    f"{event.ema200_distance_bucket}"
                ),
                "last_pair_side": f"{previous.ema200_side} -> {event.ema200_side}",
                "new_event": _event_packet(event),
                "oldest_open_event": _event_packet(open_previous[0]),
                "open_previous_count": len(open_previous),
                "full_stack_resolved": full_stack_resolved,
                "full_stack_resolved_date": resolved_date.isoformat()
                if resolved_date
                else None,
                "full_stack_resolution_calendar_days_from_new": (
                    (resolved_date - event.event_date).days if resolved_date else None
                ),
            }
        )
    return refs


def _mode_days(values: list[int]) -> tuple[int | None, int, float | None]:
    if not values:
        return None, 0, None
    counts = Counter(values)
    mode, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return mode, count, count / len(values) * 100.0


def _summarize_refs(refs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(refs)
    filled = [ref for ref in refs if ref["new_event"].get("filled")]
    days = [
        int(ref["new_event"]["trading_days_to_fill"])
        for ref in refs
        if ref["new_event"].get("trading_days_to_fill") is not None
    ]
    mode, mode_count, mode_rate = _mode_days(days)
    full_resolved = [ref for ref in refs if ref.get("full_stack_resolved")]
    resolution_days = [
        int(ref["full_stack_resolution_calendar_days_from_new"])
        for ref in full_resolved
        if ref.get("full_stack_resolution_calendar_days_from_new") is not None
    ]
    return {
        "reference_count": count,
        "new_event_fill_rate_pct": len(filled) / count * 100.0 if count else None,
        "new_event_median_trading_days": median(days) if days else None,
        "new_event_mean_trading_days": mean(days) if days else None,
        "new_event_mode_trading_days": mode,
        "new_event_mode_count": mode_count,
        "new_event_mode_rate_pct": mode_rate,
        "new_event_max_trading_days": max(days) if days else None,
        "full_stack_resolved_rate_pct": len(full_resolved) / count * 100.0
        if count
        else None,
        "full_stack_resolution_calendar_median_days": median(resolution_days)
        if resolution_days
        else None,
        "full_stack_resolution_calendar_max_days": max(resolution_days)
        if resolution_days
        else None,
    }


def _group_refs(refs: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        if key.startswith("new_event."):
            value = ref["new_event"].get(key.split(".", 1)[1])
        else:
            value = ref.get(key)
        grouped[str(value or "unknown")].append(ref)
    return {label: _summarize_refs(items) for label, items in sorted(grouped.items())}


def build_stack_surface(
    events_csv: Path,
    *,
    mode: str = "intraday_dip",
    exclude_latest_event_date: bool = True,
) -> dict[str, Any]:
    events = _read_events(events_csv, mode)
    refs = _stack_refs(events)
    latest_event_date = max((event.event_date for event in events), default=None)
    historical_refs = [
        ref
        for ref in refs
        if not exclude_latest_event_date
        or latest_event_date is None
        or ref["event_date"] < latest_event_date.isoformat()
    ]
    return {
        "schema": "sharpedge.historical_refill_stack_surface.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_events_csv": str(events_csv),
        "mode": mode,
        "basis": (
            "Canonical strongest threshold per mode/date; stack reference occurs "
            "when a new event fires while one or more prior refill targets remain open."
        ),
        "caveat": (
            "Daily OHLC cannot prove intraday sequence for same-day low/high fills; "
            "use as diagnostic context, not execution authority."
        ),
        "canonical_event_count": len(events),
        "latest_event_date": latest_event_date.isoformat()
        if latest_event_date
        else None,
        "latest_event_date_excluded_from_stats": exclude_latest_event_date,
        "stack_reference_count_all": len(refs),
        "stack_reference_count_historical": len(historical_refs),
        "overall": _summarize_refs(historical_refs),
        "by_depth": _group_refs(historical_refs, "depth"),
        "by_new_ema_side": _group_refs(historical_refs, "new_event.ema200_side"),
        "by_new_ema_distance_bucket": _group_refs(
            historical_refs, "new_event.ema200_distance_bucket"
        ),
        "by_last_pair_side": _group_refs(historical_refs, "last_pair_side"),
        "by_last_pair_distance_bucket": _group_refs(
            historical_refs, "last_pair_bucket"
        ),
        "exact_signatures": _group_refs(historical_refs, "signature"),
        "recent_references": historical_refs[-20:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-csv",
        default="outputs/spy_gap_dip_fill_surface_5y.events.csv",
    )
    parser.add_argument("--mode", default="intraday_dip")
    parser.add_argument(
        "--include-latest-event-date",
        action="store_true",
        help="Include refs whose new event date equals the latest event date.",
    )
    parser.add_argument(
        "--output",
        default="outputs/historical_refill_stack_surface.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_stack_surface(
        Path(args.events_csv),
        mode=args.mode,
        exclude_latest_event_date=not args.include_latest_event_date,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    print(
        "stack refs:",
        payload["stack_reference_count_historical"],
        "historical /",
        payload["stack_reference_count_all"],
        "all",
    )


if __name__ == "__main__":
    main()
