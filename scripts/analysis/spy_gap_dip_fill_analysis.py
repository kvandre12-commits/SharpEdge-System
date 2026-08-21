#!/usr/bin/env python3
"""Analyze large SPY down gaps/dips and time-to-fill from daily OHLC bars.

Default event definitions:
- gap_down_open: today's open is at least threshold below prior close.
- intraday_dip: today's low is at least threshold below prior close.

Fill definition for both modes:
- The prior close is the fill target.
- A fill occurs when a current/future daily high reaches or exceeds that target.
- For gap_down_open, same-day fill is valid from daily OHLC because the session
  opened below the target and later printed a high at/above it.
- For intraday_dip, same-day fill is path-ambiguous with daily OHLC because the
  high may have happened before the low. The report flags this caveat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COCKPIT = ROOT / "cockpit"
for path in (ROOT, COCKPIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ema_context import (  # noqa: E402
    classify_ema200_context,
    ema_series,
    summarize_ema200_context,
)
from market_data_sources import fetch_yahoo_daily_bars  # noqa: E402
from refill_stack_surface import build_stack_surface  # noqa: E402


@dataclass(frozen=True)
class DailyBar:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class FillEvent:
    mode: str
    threshold_pct: float
    event_date: str
    prior_close: float
    session_open: float
    session_high: float
    session_low: float
    session_close: float
    move_pct: float
    filled: bool
    fill_date: str | None
    trading_days_to_fill: int | None
    calendar_days_to_fill: int | None
    trading_days_since_prior_event: int | None
    calendar_days_since_prior_event: int | None
    ema200: float | None
    ema200_distance_pct: float | None
    ema200_side: str
    ema200_distance_bucket: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def load_yahoo_bars(symbol: str, period: str, timeout: int) -> list[DailyBar]:
    rows, _source = fetch_yahoo_daily_bars(
        symbol,
        interval="1d",
        range_=period,
        timeout=timeout,
    )
    return [
        DailyBar(
            session_date=_parse_date(row["date"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=_float_or_none(row.get("volume")),
        )
        for row in rows
    ]


def load_db_bars(db_path: Path, symbol: str) -> list[DailyBar]:
    uri = f"file:{db_path}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM bars_daily
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            (symbol,),
        ).fetchall()
    finally:
        con.close()

    return [
        DailyBar(
            session_date=_parse_date(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=_float_or_none(row[5]),
        )
        for row in rows
    ]


def is_event(mode: str, bar: DailyBar, prior_close: float, threshold: float) -> bool:
    if mode == "gap_down_open":
        return (bar.open / prior_close) - 1.0 <= -threshold
    if mode == "intraday_dip":
        return (bar.low / prior_close) - 1.0 <= -threshold
    raise ValueError(f"unknown mode: {mode}")


def move_pct_for_mode(mode: str, bar: DailyBar, prior_close: float) -> float:
    if mode == "gap_down_open":
        return (bar.open / prior_close) - 1.0
    if mode == "intraday_dip":
        return (bar.low / prior_close) - 1.0
    raise ValueError(f"unknown mode: {mode}")


def find_fill_index(
    bars: list[DailyBar], start_index: int, target: float
) -> int | None:
    for index in range(start_index, len(bars)):
        if bars[index].high >= target:
            return index
    return None


def analyze_mode(bars: list[DailyBar], mode: str, threshold: float) -> list[FillEvent]:
    events: list[FillEvent] = []
    prior_event_index: int | None = None
    prior_event_date: date | None = None
    ema200_values = ema_series([bar.close for bar in bars])

    for index in range(1, len(bars)):
        bar = bars[index]
        prior_close = bars[index - 1].close
        if prior_close <= 0 or not is_event(mode, bar, prior_close, threshold):
            continue

        fill_index = find_fill_index(bars, index, prior_close)
        fill_bar = bars[fill_index] if fill_index is not None else None
        event_date = bar.session_date
        ema_context = classify_ema200_context(prior_close, ema200_values[index - 1])

        events.append(
            FillEvent(
                mode=mode,
                threshold_pct=threshold * 100,
                event_date=event_date.isoformat(),
                prior_close=prior_close,
                session_open=bar.open,
                session_high=bar.high,
                session_low=bar.low,
                session_close=bar.close,
                move_pct=move_pct_for_mode(mode, bar, prior_close),
                filled=fill_bar is not None,
                fill_date=fill_bar.session_date.isoformat() if fill_bar else None,
                trading_days_to_fill=(fill_index - index)
                if fill_index is not None
                else None,
                calendar_days_to_fill=(fill_bar.session_date - event_date).days
                if fill_bar
                else None,
                trading_days_since_prior_event=(index - prior_event_index)
                if prior_event_index is not None
                else None,
                calendar_days_since_prior_event=(event_date - prior_event_date).days
                if prior_event_date is not None
                else None,
                ema200=ema_context["ema200"],
                ema200_distance_pct=ema_context["distance_pct"],
                ema200_side=ema_context["side"],
                ema200_distance_bucket=ema_context["distance_bucket"],
            )
        )
        prior_event_index = index
        prior_event_date = event_date

    return events


def _number_stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "modes": [],
            "mode_count": 0,
            "mode_rate_pct": None,
            "min": None,
            "max": None,
        }

    sorted_values = sorted(values)
    counts = Counter(sorted_values)
    max_count = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == max_count)
    has_real_mode = max_count > 1
    mode_count = max_count if has_real_mode else 0
    return {
        "count": len(sorted_values),
        "mean": mean(sorted_values),
        "median": median(sorted_values),
        "mode": modes[0] if has_real_mode else None,
        "modes": modes if has_real_mode else [],
        "mode_count": mode_count,
        "mode_rate_pct": (mode_count / len(sorted_values) * 100)
        if has_real_mode
        else None,
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def _horizon_stats(events: list[FillEvent], horizons: list[int]) -> dict[str, Any]:
    if not events:
        return {str(horizon): {"count": 0, "rate_pct": None} for horizon in horizons}

    filled_days = [event.trading_days_to_fill for event in events if event.filled]
    clean_days = [value for value in filled_days if value is not None]
    return {
        str(horizon): {
            "count": sum(value <= horizon for value in clean_days),
            "rate_pct": sum(value <= horizon for value in clean_days)
            / len(events)
            * 100,
        }
        for horizon in horizons
    }


def summarize_events(
    events: list[FillEvent],
    bars: list[DailyBar],
    mode: str,
    threshold: float,
    horizons: list[int],
) -> dict[str, Any]:
    filled = [event for event in events if event.filled]
    unfilled = [event for event in events if not event.filled]
    comparable_sessions = max(len(bars) - 1, 0)
    fill_days = [event.trading_days_to_fill for event in filled]
    fill_calendar_days = [event.calendar_days_to_fill for event in filled]
    interval_days = [event.trading_days_since_prior_event for event in events]
    interval_calendar_days = [event.calendar_days_since_prior_event for event in events]

    clean_fill_days = [value for value in fill_days if value is not None]
    clean_fill_calendar_days = [
        value for value in fill_calendar_days if value is not None
    ]
    clean_interval_days = [value for value in interval_days if value is not None]
    clean_interval_calendar_days = [
        value for value in interval_calendar_days if value is not None
    ]

    return {
        "mode": mode,
        "threshold_pct": threshold * 100,
        "bar_start": bars[0].session_date.isoformat() if bars else None,
        "bar_end": bars[-1].session_date.isoformat() if bars else None,
        "bar_count": len(bars),
        "comparable_sessions": comparable_sessions,
        "event_count": len(events),
        "event_frequency_pct": (len(events) / comparable_sessions * 100)
        if comparable_sessions
        else None,
        "filled_count": len(filled),
        "unfilled_count": len(unfilled),
        "fill_rate_pct": (len(filled) / len(events) * 100) if events else None,
        "time_to_fill_trading_days": _number_stats(clean_fill_days),
        "time_to_fill_calendar_days": _number_stats(clean_fill_calendar_days),
        "event_interval_trading_days": _number_stats(clean_interval_days),
        "event_interval_calendar_days": _number_stats(clean_interval_calendar_days),
        "fill_horizon_trading_days": _horizon_stats(events, horizons),
        "ema200_context": summarize_ema200_context(events),
        "unfilled_events": [event.event_date for event in unfilled],
        "daily_ohlc_caveat": (
            "Same-day intraday_dip fills are path-ambiguous with daily OHLC; "
            "use intraday bars for ordered low-before-fill proof."
            if mode == "intraday_dip"
            else "Gap-down same-day fills are detectable from daily OHLC because open starts below target."
        ),
    }


def write_events_csv(path: Path, events: list[FillEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(event) for event in events]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# SPY Down Gap/Dip Fill Surface", ""]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['mode']}",
                "",
                f"Range: {summary['bar_start']} → {summary['bar_end']} ({summary['bar_count']} daily bars)",
                f"Threshold: {summary['threshold_pct']:.2f}% below prior close",
                f"Events: {summary['event_count']} / {summary['comparable_sessions']} sessions "
                f"({summary['event_frequency_pct']:.2f}%)",
                f"Filled: {summary['filled_count']} | Unfilled: {summary['unfilled_count']} | "
                f"Fill rate: {summary['fill_rate_pct']:.2f}%"
                if summary["fill_rate_pct"] is not None
                else "Fill rate: NA",
                "",
                "### Time to fill — trading days",
                f"`{json.dumps(summary['time_to_fill_trading_days'], sort_keys=True)}`",
                "",
                "### Time to fill — calendar days",
                f"`{json.dumps(summary['time_to_fill_calendar_days'], sort_keys=True)}`",
                "",
                "### Event interval — trading days",
                f"`{json.dumps(summary['event_interval_trading_days'], sort_keys=True)}`",
                "",
                "### Fill horizon rates — trading days",
                f"`{json.dumps(summary['fill_horizon_trading_days'], sort_keys=True)}`",
                "",
                f"Caveat: {summary['daily_ohlc_caveat']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def surface_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        fill_days = summary["time_to_fill_trading_days"]
        event_count = summary["event_count"]
        mode_count = fill_days["mode_count"]
        mode_event_rate = (mode_count / event_count * 100) if event_count else None
        row = {
            "mode": summary["mode"],
            "threshold_pct": summary["threshold_pct"],
            "event_count": event_count,
            "event_frequency_pct": summary["event_frequency_pct"],
            "filled_count": summary["filled_count"],
            "unfilled_count": summary["unfilled_count"],
            "fill_rate_pct": summary["fill_rate_pct"],
            "fill_mean_trading_days": fill_days["mean"],
            "fill_median_trading_days": fill_days["median"],
            "fill_mode_trading_days": fill_days["mode"],
            "fill_mode_count": mode_count,
            "fill_mode_rate_pct": mode_event_rate,
            "fill_max_trading_days": fill_days["max"],
            "ema200_context": summary.get("ema200_context", {}),
        }
        for horizon, stats in summary["fill_horizon_trading_days"].items():
            row[f"fill_within_{horizon}d_count"] = stats["count"]
            row[f"fill_within_{horizon}d_rate_pct"] = stats["rate_pct"]
        rows.append(row)
    return rows


def write_surface_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_surface_json(
    path: Path,
    *,
    symbol: str,
    source: str,
    period: str,
    bars: list[DailyBar],
    horizons: list[int],
    summaries: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": "sharpedge.historical_refill_surface.v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "symbol": symbol,
        "source": source,
        "period": period,
        "bar_start": bars[0].session_date.isoformat() if bars else None,
        "bar_end": bars[-1].session_date.isoformat() if bars else None,
        "bar_count": len(bars),
        "horizons_trading_days": horizons,
        "authority": "diagnostic_only",
        "cockpit_feature": {
            "name": "historical_refill_surface",
            "usage": "Compare current down gap/dip magnitude to historical refill behavior.",
            "not_authority": "Does not override execution permission or final authority gates.",
            "suggested_live_inputs": [
                "open_vs_prior_close_pct",
                "low_vs_prior_close_pct",
                "current_gap_fill_target",
                "ema200_side",
                "ema200_distance_pct",
                "ema200_distance_bucket",
            ],
        },
        "rows": surface_rows(summaries),
        "summaries": summaries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_threshold_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated decimal thresholds, e.g. 0.015,0.02,0.025. Overrides --threshold.",
    )
    parser.add_argument(
        "--horizons",
        default="0,1,3,5,10,20,60,120",
        help="Comma-separated trading-day fill horizons for surface rates.",
    )
    parser.add_argument("--period", default="5y")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--source", choices=("yahoo", "db"), default="yahoo")
    parser.add_argument("--db-path", default="data/spy_truth.db")
    parser.add_argument(
        "--mode",
        choices=("gap_down_open", "intraday_dip", "both"),
        default="both",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/spy_gap_dip_fill_3pct_5y",
    )
    parser.add_argument(
        "--cockpit-surface-path",
        default="outputs/historical_refill_surface.json",
        help="Stable cockpit-ready diagnostic JSON surface path.",
    )
    parser.add_argument(
        "--cockpit-stack-surface-path",
        default="outputs/historical_refill_stack_surface.json",
        help="Stable cockpit-ready JSON surface for active dip-stack interactions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars = (
        load_yahoo_bars(args.symbol, args.period, args.timeout)
        if args.source == "yahoo"
        else load_db_bars(Path(args.db_path), args.symbol)
    )
    if len(bars) < 2:
        raise SystemExit("Need at least 2 daily bars for prior-close analysis.")

    modes = ["gap_down_open", "intraday_dip"] if args.mode == "both" else [args.mode]
    thresholds = (
        parse_threshold_list(args.thresholds) if args.thresholds else [args.threshold]
    )
    horizons = parse_int_list(args.horizons)

    all_events: list[FillEvent] = []
    summaries = []
    for threshold in thresholds:
        for mode in modes:
            events = analyze_mode(bars, mode, threshold)
            all_events.extend(events)
            summaries.append(summarize_events(events, bars, mode, threshold, horizons))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    surface = surface_rows(summaries)
    events_csv_path = prefix.with_suffix(".events.csv")
    write_events_csv(events_csv_path, all_events)
    write_surface_csv(prefix.with_suffix(".surface.csv"), surface)
    prefix.with_suffix(".summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_surface_json(
        prefix.with_suffix(".surface.json"),
        symbol=args.symbol,
        source=args.source,
        period=args.period,
        bars=bars,
        horizons=horizons,
        summaries=summaries,
    )
    write_surface_json(
        Path(args.cockpit_surface_path),
        symbol=args.symbol,
        source=args.source,
        period=args.period,
        bars=bars,
        horizons=horizons,
        summaries=summaries,
    )
    stack_surface = build_stack_surface(events_csv_path, mode="intraday_dip")
    stack_surface_path = Path(args.cockpit_stack_surface_path)
    stack_surface_path.parent.mkdir(parents=True, exist_ok=True)
    stack_surface_path.write_text(
        json.dumps(stack_surface, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(prefix.with_suffix(".md"), summaries)

    for summary in summaries:
        print(
            f"{summary['mode']} {summary['threshold_pct']:.1f}%: "
            f"events={summary['event_count']} "
            f"freq={summary['event_frequency_pct']:.2f}% "
            f"filled={summary['filled_count']} unfilled={summary['unfilled_count']} "
            f"fill_rate={summary['fill_rate_pct']:.2f}%"
            if summary["fill_rate_pct"] is not None
            else f"{summary['mode']} {summary['threshold_pct']:.1f}%: events=0"
        )
        print("  trading_days_to_fill:", summary["time_to_fill_trading_days"])
    print(f"wrote {prefix.with_suffix('.summary.json')}")
    print(f"wrote {prefix.with_suffix('.events.csv')}")
    print(f"wrote {prefix.with_suffix('.surface.csv')}")
    print(f"wrote {prefix.with_suffix('.surface.json')}")
    print(f"wrote {Path(args.cockpit_surface_path)}")
    print(f"wrote {Path(args.cockpit_stack_surface_path)}")
    print(f"wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
