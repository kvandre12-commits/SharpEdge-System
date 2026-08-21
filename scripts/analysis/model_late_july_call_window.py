#!/usr/bin/env python3
"""Late-July/August call-window seasonality model.

Models whether a same-distance upside target is tagged within 20/30 sessions
when the anchor date falls in a late-July/early-August window.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Candle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class SeasonalEpisode:
    date: str
    close: float
    month_day: str
    close_vs_prior_close_pct: float
    red_close: bool
    target_750_distance_pct: float
    target_757_distance_pct: float
    hit_750_10d: bool
    hit_750_20d: bool
    hit_750_30d: bool
    hit_757_10d: bool
    hit_757_20d: bool
    hit_757_30d: bool
    max_high_20d_pct: float | None
    max_high_30d_pct: float | None
    min_low_20d_pct: float | None
    min_low_30d_pct: float | None
    close_return_20d_pct: float | None
    close_return_30d_pct: float | None


def pct(current: float, base: float) -> float:
    return ((current / base) - 1.0) * 100.0


def fetch_yahoo_daily(symbol: str, data_range: str) -> list[Candle]:
    params = urlencode(
        {
            "range": data_range,
            "interval": "1d",
            "events": "div,splits",
            "includePrePost": "false",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 SharpEdge seasonal call-window model"},
    )
    with urlopen(req, timeout=30) as response:  # noqa: S310 - public market data
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo returned no result: {payload}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
    candles: list[Candle] = []
    for index, ts in enumerate(timestamps):
        values = {
            key: (quote.get(key) or [None] * len(timestamps))[index]
            for key in ("open", "high", "low", "close", "volume")
        }
        if any(values[key] is None for key in ("open", "high", "low", "close")):
            continue
        candles.append(
            Candle(
                date=datetime.fromtimestamp(ts, tz=UTC).date().isoformat(),
                open=float(values["open"]),
                high=float(values["high"]),
                low=float(values["low"]),
                close=float(values["close"]),
                volume=int(values["volume"] or 0),
            )
        )
    return candles


def month_day_in_window(month_day: str, start: str, end: str) -> bool:
    return start <= month_day <= end


def max_high_pct(candles: list[Candle], index: int, horizon: int) -> float | None:
    rows = candles[index + 1 : min(len(candles), index + horizon + 1)]
    if not rows:
        return None
    return pct(max(candle.high for candle in rows), candles[index].close)


def min_low_pct(candles: list[Candle], index: int, horizon: int) -> float | None:
    rows = candles[index + 1 : min(len(candles), index + horizon + 1)]
    if not rows:
        return None
    return pct(min(candle.low for candle in rows), candles[index].close)


def close_return(candles: list[Candle], index: int, horizon: int) -> float | None:
    target = index + horizon
    if target >= len(candles):
        return None
    return pct(candles[target].close, candles[index].close)


def hit_target(candles: list[Candle], index: int, distance_pct: float, horizon: int) -> bool:
    target = candles[index].close * (1.0 + distance_pct / 100.0)
    rows = candles[index + 1 : min(len(candles), index + horizon + 1)]
    return any(candle.high >= target for candle in rows)


def build_episodes(
    candles: list[Candle],
    *,
    window_start: str,
    window_end: str,
    distance_750: float,
    distance_757: float,
) -> list[SeasonalEpisode]:
    episodes: list[SeasonalEpisode] = []
    for index in range(1, len(candles) - 30):
        candle = candles[index]
        month_day = candle.date[5:]
        if not month_day_in_window(month_day, window_start, window_end):
            continue
        close_vs_prior = pct(candle.close, candles[index - 1].close)
        episodes.append(
            SeasonalEpisode(
                date=candle.date,
                close=candle.close,
                month_day=month_day,
                close_vs_prior_close_pct=close_vs_prior,
                red_close=close_vs_prior < 0,
                target_750_distance_pct=distance_750,
                target_757_distance_pct=distance_757,
                hit_750_10d=hit_target(candles, index, distance_750, 10),
                hit_750_20d=hit_target(candles, index, distance_750, 20),
                hit_750_30d=hit_target(candles, index, distance_750, 30),
                hit_757_10d=hit_target(candles, index, distance_757, 10),
                hit_757_20d=hit_target(candles, index, distance_757, 20),
                hit_757_30d=hit_target(candles, index, distance_757, 30),
                max_high_20d_pct=max_high_pct(candles, index, 20),
                max_high_30d_pct=max_high_pct(candles, index, 30),
                min_low_20d_pct=min_low_pct(candles, index, 20),
                min_low_30d_pct=min_low_pct(candles, index, 30),
                close_return_20d_pct=close_return(candles, index, 20),
                close_return_30d_pct=close_return(candles, index, 30),
            )
        )
    return episodes


def median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.median(finite)


def hit_rate(events: list[SeasonalEpisode], field: str) -> float | None:
    if not events:
        return None
    return sum(bool(getattr(event, field)) for event in events) / len(events) * 100.0


def positive_rate(events: list[SeasonalEpisode], field: str) -> float | None:
    values = [getattr(event, field) for event in events if getattr(event, field) is not None]
    if not values:
        return None
    return sum(value > 0 for value in values) / len(values) * 100.0


def summarize(events: list[SeasonalEpisode]) -> dict[str, Any]:
    return {
        "count": len(events),
        "hit_750_10d_pct": hit_rate(events, "hit_750_10d"),
        "hit_750_20d_pct": hit_rate(events, "hit_750_20d"),
        "hit_750_30d_pct": hit_rate(events, "hit_750_30d"),
        "hit_757_10d_pct": hit_rate(events, "hit_757_10d"),
        "hit_757_20d_pct": hit_rate(events, "hit_757_20d"),
        "hit_757_30d_pct": hit_rate(events, "hit_757_30d"),
        "median_max_high_20d_pct": median([e.max_high_20d_pct for e in events]),
        "median_max_high_30d_pct": median([e.max_high_30d_pct for e in events]),
        "median_min_low_20d_pct": median([e.min_low_20d_pct for e in events]),
        "median_min_low_30d_pct": median([e.min_low_30d_pct for e in events]),
        "positive_20d_close_pct": positive_rate(events, "close_return_20d_pct"),
        "positive_30d_close_pct": positive_rate(events, "close_return_30d_pct"),
        "median_close_return_20d_pct": median([e.close_return_20d_pct for e in events]),
        "median_close_return_30d_pct": median([e.close_return_30d_pct for e in events]),
    }


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def write_csv(path: Path, episodes: list[SeasonalEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(episodes[0]).keys()))
        writer.writeheader()
        for episode in episodes:
            writer.writerow(rounded(asdict(episode)))


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SPY Late-July Month-Long 750C Context",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Window: `{report['window_start']}` to `{report['window_end']}` by month/day.",
        f"Spot: `{report['spot']}`; 750 distance: `{report['distance_750_pct']}%`; 757.09 distance: `{report['distance_757_pct']}%`.",
        "",
        "## Seasonality / target tag odds",
        "",
        "| Cohort | N | 750 <=10d | 750 <=20d | 750 <=30d | 757 <=20d | 757 <=30d | Median 30d high | Median 30d low | Median 30d close |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name.replace("_", " "),
                    str(stats["count"]),
                    fmt_pct(stats["hit_750_10d_pct"]),
                    fmt_pct(stats["hit_750_20d_pct"]),
                    fmt_pct(stats["hit_750_30d_pct"]),
                    fmt_pct(stats["hit_757_20d_pct"]),
                    fmt_pct(stats["hit_757_30d_pct"]),
                    fmt_pct(stats["median_max_high_30d_pct"]),
                    fmt_pct(stats["median_min_low_30d_pct"]),
                    fmt_pct(stats["median_close_return_30d_pct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- If Apple rejection pushes 757 farther out, 750 becomes the cleaner month-long call strike/target zone.",
            "- The 750C idea should be treated as a time-backed reclaim thesis, not a short-dated squeeze chase.",
            "- Path matters: ideally buy after rejection damage/vol crush or after 747/750 acceptance, not into pre-event IV froth like a raccoon with a Robinhood account.",
            "- AAPL lower-open rejection changes timing: 757 becomes later unfinished business; 750 can still be reachable inside a 20–30 session window.",
            "",
            "Research only. Broker-fresh quotes and operator approval required for any trade.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Late-July/August SPY 750C context model.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--range", default="10y", dest="data_range")
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--target-750", type=float, default=750.0)
    parser.add_argument("--target-757", type=float, default=757.09)
    parser.add_argument("--window-start", default="07-20")
    parser.add_argument("--window-end", default="08-05")
    parser.add_argument("--output-dir", default="outputs/late_july_call_window")
    args = parser.parse_args()

    candles = fetch_yahoo_daily(args.symbol.upper(), args.data_range)
    distance_750 = pct(args.target_750, args.spot)
    distance_757 = pct(args.target_757, args.spot)
    episodes = build_episodes(
        candles,
        window_start=args.window_start,
        window_end=args.window_end,
        distance_750=distance_750,
        distance_757=distance_757,
    )
    if not episodes:
        raise RuntimeError("No seasonal episodes matched the requested window.")
    red = [event for event in episodes if event.red_close]
    report = rounded(
        {
            "schema": "sharpedge.late_july_call_window.v1",
            "symbol": args.symbol.upper(),
            "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "source": "Yahoo Finance daily OHLC",
            "window_start": args.window_start,
            "window_end": args.window_end,
            "spot": args.spot,
            "target_750": args.target_750,
            "target_757": args.target_757,
            "distance_750_pct": distance_750,
            "distance_757_pct": distance_757,
            "summaries": {
                "late_july_early_aug_all": summarize(episodes),
                "late_july_early_aug_red_close": summarize(red),
            },
            "episodes": [asdict(event) for event in episodes],
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spy_late_july_750c_context.json"
    csv_path = output_dir / "spy_late_july_750c_context.csv"
    md_path = output_dir / "spy_late_july_750c_context.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, episodes)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
