#!/usr/bin/env python3
"""Model deadline odds for a stubborn historical refill target.

Given a current spot/close and target, estimate how often SPY historically tags
an equivalent upside distance within N sessions. This is a timing model, not a
trade signal; cockpit context still owns live acceptance/rejection.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
COCKPIT = ROOT / "cockpit"
ANALYSIS = ROOT / "scripts" / "analysis"
for path in (COCKPIT, ANALYSIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_apple_earnings_reaction_dips import AAPL_EARNINGS_DATES  # noqa: E402
from event_calendar import FOMC_DATES  # noqa: E402


HORIZONS = (1, 2, 3, 4, 5, 10, 20)


@dataclass(frozen=True)
class Candle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class TargetEpisode:
    date: str
    close: float
    synthetic_target: float
    target_distance_pct: float
    close_vs_prior_close_pct: float | None
    red_close: bool | None
    near_fomc_calendar_days: int | None
    near_apple_calendar_days: int | None
    hit_1d: bool
    hit_2d: bool
    hit_3d: bool
    hit_4d: bool
    hit_5d: bool
    hit_10d: bool
    hit_20d: bool
    max_high_3d_pct: float | None
    max_high_5d_pct: float | None
    max_high_10d_pct: float | None
    min_low_5d_pct: float | None
    close_return_3d_pct: float | None
    close_return_5d_pct: float | None
    close_return_10d_pct: float | None


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
        headers={"User-Agent": "Mozilla/5.0 SharpEdge target deadline model"},
    )
    with urlopen(req, timeout=30) as response:  # noqa: S310 - public market data
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo returned no result for {symbol}: {payload}")
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


def calendar_days_to_next(current: str, dates: list[str], *, max_days: int = 10) -> int | None:
    current_date = date.fromisoformat(current)
    deltas = [
        (date.fromisoformat(item) - current_date).days
        for item in dates
        if date.fromisoformat(item) >= current_date
    ]
    if not deltas:
        return None
    delta = min(deltas)
    return delta if delta <= max_days else None


def hit_within(candles: list[Candle], start_index: int, target: float, horizon: int) -> bool:
    end = min(len(candles), start_index + horizon + 1)
    return any(candle.high >= target for candle in candles[start_index + 1 : end])


def max_high_pct(candles: list[Candle], start_index: int, base: float, horizon: int) -> float | None:
    end = min(len(candles), start_index + horizon + 1)
    rows = candles[start_index + 1 : end]
    if not rows:
        return None
    return pct(max(candle.high for candle in rows), base)


def min_low_pct(candles: list[Candle], start_index: int, base: float, horizon: int) -> float | None:
    end = min(len(candles), start_index + horizon + 1)
    rows = candles[start_index + 1 : end]
    if not rows:
        return None
    return pct(min(candle.low for candle in rows), base)


def forward_close_pct(candles: list[Candle], start_index: int, horizon: int) -> float | None:
    target = start_index + horizon
    if target >= len(candles):
        return None
    return pct(candles[target].close, candles[start_index].close)


def build_episodes(candles: list[Candle], target_distance_pct: float) -> list[TargetEpisode]:
    episodes: list[TargetEpisode] = []
    multiplier = 1.0 + target_distance_pct / 100.0
    for index in range(1, len(candles) - 20):
        candle = candles[index]
        prior = candles[index - 1]
        target = candle.close * multiplier
        close_vs_prior = pct(candle.close, prior.close)
        hits = {horizon: hit_within(candles, index, target, horizon) for horizon in HORIZONS}
        episodes.append(
            TargetEpisode(
                date=candle.date,
                close=candle.close,
                synthetic_target=target,
                target_distance_pct=target_distance_pct,
                close_vs_prior_close_pct=close_vs_prior,
                red_close=close_vs_prior < 0,
                near_fomc_calendar_days=calendar_days_to_next(candle.date, list(FOMC_DATES)),
                near_apple_calendar_days=calendar_days_to_next(candle.date, AAPL_EARNINGS_DATES),
                hit_1d=hits[1],
                hit_2d=hits[2],
                hit_3d=hits[3],
                hit_4d=hits[4],
                hit_5d=hits[5],
                hit_10d=hits[10],
                hit_20d=hits[20],
                max_high_3d_pct=max_high_pct(candles, index, candle.close, 3),
                max_high_5d_pct=max_high_pct(candles, index, candle.close, 5),
                max_high_10d_pct=max_high_pct(candles, index, candle.close, 10),
                min_low_5d_pct=min_low_pct(candles, index, candle.close, 5),
                close_return_3d_pct=forward_close_pct(candles, index, 3),
                close_return_5d_pct=forward_close_pct(candles, index, 5),
                close_return_10d_pct=forward_close_pct(candles, index, 10),
            )
        )
    return episodes


def median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.median(finite)


def summarize(events: list[TargetEpisode]) -> dict[str, Any]:
    def hit_rate(field: str) -> float | None:
        if not events:
            return None
        return sum(bool(getattr(event, field)) for event in events) / len(events) * 100.0

    return {
        "count": len(events),
        "hit_1d_pct": hit_rate("hit_1d"),
        "hit_2d_pct": hit_rate("hit_2d"),
        "hit_3d_pct": hit_rate("hit_3d"),
        "hit_4d_pct": hit_rate("hit_4d"),
        "hit_5d_pct": hit_rate("hit_5d"),
        "hit_10d_pct": hit_rate("hit_10d"),
        "hit_20d_pct": hit_rate("hit_20d"),
        "median_max_high_3d_pct": median([event.max_high_3d_pct for event in events]),
        "median_max_high_5d_pct": median([event.max_high_5d_pct for event in events]),
        "median_max_high_10d_pct": median([event.max_high_10d_pct for event in events]),
        "median_min_low_5d_pct": median([event.min_low_5d_pct for event in events]),
        "median_close_return_3d_pct": median([event.close_return_3d_pct for event in events]),
        "median_close_return_5d_pct": median([event.close_return_5d_pct for event in events]),
        "median_close_return_10d_pct": median([event.close_return_10d_pct for event in events]),
    }


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def build_report(
    symbol: str,
    candles: list[Candle],
    episodes: list[TargetEpisode],
    spot: float,
    target: float,
) -> dict[str, Any]:
    distance_pct = pct(target, spot)
    red = [event for event in episodes if event.red_close]
    near_fomc = [
        event
        for event in episodes
        if event.near_fomc_calendar_days is not None and event.near_fomc_calendar_days <= 3
    ]
    near_apple = [
        event
        for event in episodes
        if event.near_apple_calendar_days is not None and event.near_apple_calendar_days <= 3
    ]
    red_near_event = [
        event
        for event in episodes
        if event.red_close
        and (
            (event.near_fomc_calendar_days is not None and event.near_fomc_calendar_days <= 3)
            or (event.near_apple_calendar_days is not None and event.near_apple_calendar_days <= 3)
        )
    ]
    latest = candles[-1]
    return rounded(
        {
            "schema": "sharpedge.refill_target_deadline_model.v1",
            "symbol": symbol,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "source": "Yahoo Finance daily OHLC + SharpEdge event calendar + curated AAPL earnings dates",
            "latest_yahoo_date": latest.date,
            "latest_yahoo_close": latest.close,
            "model_spot": spot,
            "target": target,
            "target_distance_pct": distance_pct,
            "sessions_to_fomc_approx": 2,
            "sessions_to_aapl_earnings_approx": 3,
            "sessions_to_aapl_reaction_approx": 4,
            "summaries": {
                "all_days_same_distance": summarize(episodes),
                "red_close_days_same_distance": summarize(red),
                "within_3_calendar_days_of_fomc": summarize(near_fomc),
                "within_3_calendar_days_of_aapl_earnings": summarize(near_apple),
                "red_close_near_fomc_or_aapl": summarize(red_near_event),
            },
            "active_context": {
                "interpretation": "757 is the older, slow/stubborn refill target; 747 is the nearer reclaim target already tested/rejected.",
                "required_move_before_aapl_earnings_pct": distance_pct,
                "required_average_daily_close_move_3_sessions_pct": ((target / spot) ** (1 / 3) - 1) * 100,
                "required_average_daily_close_move_4_sessions_pct": ((target / spot) ** (1 / 4) - 1) * 100,
            },
        }
    )


def write_csv(path: Path, episodes: list[TargetEpisode]) -> None:
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
        f"# {report['symbol']} {report['target']} Refill Target Deadline Model",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Spot used: `{report['model_spot']}`; target: `{report['target']}`; distance: `{report['target_distance_pct']}%`.",
        "",
        "## Deadline odds for same-distance upside tag",
        "",
        "| Cohort | N | <=1d | <=2d | <=3d | <=4d | <=5d | <=10d | Median 5d max high | Median 5d low |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name.replace("_", " "),
                    str(stats["count"]),
                    fmt_pct(stats["hit_1d_pct"]),
                    fmt_pct(stats["hit_2d_pct"]),
                    fmt_pct(stats["hit_3d_pct"]),
                    fmt_pct(stats["hit_4d_pct"]),
                    fmt_pct(stats["hit_5d_pct"]),
                    fmt_pct(stats["hit_10d_pct"]),
                    fmt_pct(stats["median_max_high_5d_pct"]),
                    fmt_pct(stats["median_min_low_5d_pct"]),
                ]
            )
            + " |"
        )
    ctx = report["active_context"]
    lines.extend(
        [
            "",
            "## Context read",
            "",
            f"- {ctx['interpretation']}",
            f"- To tag 757 before AAPL earnings, SPY needs about `{ctx['required_average_daily_close_move_3_sessions_pct']}%` compounded per session for 3 sessions.",
            f"- To tag by the AAPL reaction session, it needs about `{ctx['required_average_daily_close_move_4_sessions_pct']}%` compounded per session for 4 sessions.",
            "- If price is slow/stubborn under the target, treat 757 as a magnet/ceiling model, not an automatic near-term destination.",
            "- Live acceptance over 747/then 750 is the path check; without that, 757 remains older unfinished business, not a trigger.",
            "",
            "Research only. Broker-fresh quotes and operator approval required for any trade.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model deadline odds to a refill target.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--range", default="10y", dest="data_range")
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--output-dir", default="outputs/refill_target_deadline")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    candles = fetch_yahoo_daily(symbol, args.data_range)
    distance_pct = pct(args.target, args.spot)
    episodes = build_episodes(candles, distance_pct)
    report = build_report(symbol, candles, episodes, args.spot, args.target)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{symbol.lower()}_{args.target:g}_deadline_model.json"
    csv_path = output_dir / f"{symbol.lower()}_{args.target:g}_deadline_episodes.csv"
    md_path = output_dir / f"{symbol.lower()}_{args.target:g}_deadline_model.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, episodes)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
