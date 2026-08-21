#!/usr/bin/env python3
"""Analyze lower-open dips after Apple earnings reports.

Apple usually reports after close, so this study treats the next trading session
as the reaction day. It measures whether SPY/NVDA/AAPL open below the prior
close and whether lower-open pressure persists over the next few sessions.

Research-only: earnings dates below are curated and should be verified against
Apple investor-relations/Nasdaq before becoming authority.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


AAPL_EARNINGS_DATES = [
    "2016-01-26", "2016-04-26", "2016-07-26", "2016-10-25",
    "2017-01-31", "2017-05-02", "2017-08-01", "2017-11-02",
    "2018-02-01", "2018-05-01", "2018-07-31", "2018-11-01",
    "2019-01-29", "2019-04-30", "2019-07-30", "2019-10-30",
    "2020-01-28", "2020-04-30", "2020-07-30", "2020-10-29",
    "2021-01-27", "2021-04-28", "2021-07-27", "2021-10-28",
    "2022-01-27", "2022-04-28", "2022-07-28", "2022-10-27",
    "2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02",
    "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
    "2025-01-30", "2025-05-01", "2025-07-31", "2025-10-30",
    "2026-01-29", "2026-04-30", "2026-07-30",
]


@dataclass(frozen=True)
class Candle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class AppleReactionEvent:
    symbol: str
    report_date: str
    reaction_date: str
    prior_close: float
    reaction_open: float
    reaction_close: float
    reaction_open_gap_pct: float
    reaction_close_vs_prior_close_pct: float
    reaction_low_vs_prior_close_pct: float
    opened_below_prior_close: bool
    closed_below_prior_close: bool
    consecutive_lower_opens_from_reaction: int
    lower_open_count_next_3_sessions: int
    lower_open_count_next_5_sessions: int
    return_1d_from_reaction_close_pct: float | None
    return_3d_from_reaction_close_pct: float | None
    return_5d_from_reaction_close_pct: float | None


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
        headers={"User-Agent": "Mozilla/5.0 SharpEdge Apple earnings study"},
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


def next_trading_index(candles: list[Candle], after_date: str) -> int | None:
    target = date.fromisoformat(after_date)
    for index, candle in enumerate(candles):
        if date.fromisoformat(candle.date) > target:
            return index
    return None


def forward_return(candles: list[Candle], index: int, days: int) -> float | None:
    target = index + days
    if target >= len(candles):
        return None
    return pct(candles[target].close, candles[index].close)


def lower_open_streak(candles: list[Candle], start_index: int) -> int:
    streak = 0
    for index in range(start_index, len(candles)):
        if index == 0 or candles[index].open >= candles[index - 1].close:
            break
        streak += 1
    return streak


def lower_open_count(candles: list[Candle], start_index: int, sessions: int) -> int:
    count = 0
    end = min(len(candles), start_index + sessions)
    for index in range(start_index, end):
        if index > 0 and candles[index].open < candles[index - 1].close:
            count += 1
    return count


def build_events(symbol: str, candles: list[Candle]) -> list[AppleReactionEvent]:
    events: list[AppleReactionEvent] = []
    for report_date in AAPL_EARNINGS_DATES:
        reaction_index = next_trading_index(candles, report_date)
        if reaction_index is None or reaction_index <= 0:
            continue
        reaction = candles[reaction_index]
        prior = candles[reaction_index - 1]
        events.append(
            AppleReactionEvent(
                symbol=symbol,
                report_date=report_date,
                reaction_date=reaction.date,
                prior_close=prior.close,
                reaction_open=reaction.open,
                reaction_close=reaction.close,
                reaction_open_gap_pct=pct(reaction.open, prior.close),
                reaction_close_vs_prior_close_pct=pct(reaction.close, prior.close),
                reaction_low_vs_prior_close_pct=pct(reaction.low, prior.close),
                opened_below_prior_close=reaction.open < prior.close,
                closed_below_prior_close=reaction.close < prior.close,
                consecutive_lower_opens_from_reaction=lower_open_streak(candles, reaction_index),
                lower_open_count_next_3_sessions=lower_open_count(candles, reaction_index, 3),
                lower_open_count_next_5_sessions=lower_open_count(candles, reaction_index, 5),
                return_1d_from_reaction_close_pct=forward_return(candles, reaction_index, 1),
                return_3d_from_reaction_close_pct=forward_return(candles, reaction_index, 3),
                return_5d_from_reaction_close_pct=forward_return(candles, reaction_index, 5),
            )
        )
    return events


def median(values: list[float | int | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(statistics.median(finite))


def rate(events: list[AppleReactionEvent], field: str) -> float | None:
    if not events:
        return None
    return sum(bool(getattr(event, field)) for event in events) / len(events) * 100.0


def positive_rate(events: list[AppleReactionEvent], field: str) -> float | None:
    values = [getattr(event, field) for event in events if getattr(event, field) is not None]
    if not values:
        return None
    return sum(value > 0 for value in values) / len(values) * 100.0


def summarize(events: list[AppleReactionEvent]) -> dict[str, Any]:
    lower_open_events = [event for event in events if event.opened_below_prior_close]
    return {
        "count": len(events),
        "opened_below_prior_close_pct": rate(events, "opened_below_prior_close"),
        "closed_below_prior_close_pct": rate(events, "closed_below_prior_close"),
        "median_reaction_open_gap_pct": median([e.reaction_open_gap_pct for e in events]),
        "median_reaction_close_vs_prior_close_pct": median(
            [e.reaction_close_vs_prior_close_pct for e in events]
        ),
        "median_reaction_low_vs_prior_close_pct": median(
            [e.reaction_low_vs_prior_close_pct for e in events]
        ),
        "median_consecutive_lower_opens_from_reaction": median(
            [e.consecutive_lower_opens_from_reaction for e in lower_open_events]
        ),
        "median_lower_open_count_next_3_sessions": median(
            [e.lower_open_count_next_3_sessions for e in events]
        ),
        "median_lower_open_count_next_5_sessions": median(
            [e.lower_open_count_next_5_sessions for e in events]
        ),
        "return_1d_positive_pct": positive_rate(events, "return_1d_from_reaction_close_pct"),
        "return_3d_positive_pct": positive_rate(events, "return_3d_from_reaction_close_pct"),
        "return_5d_positive_pct": positive_rate(events, "return_5d_from_reaction_close_pct"),
        "median_return_1d_pct": median([e.return_1d_from_reaction_close_pct for e in events]),
        "median_return_3d_pct": median([e.return_3d_from_reaction_close_pct for e in events]),
        "median_return_5d_pct": median([e.return_5d_from_reaction_close_pct for e in events]),
    }


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def build_report(symbol_events: dict[str, list[AppleReactionEvent]]) -> dict[str, Any]:
    summaries = {}
    all_events = []
    for symbol, events in symbol_events.items():
        lower = [event for event in events if event.opened_below_prior_close]
        not_lower = [event for event in events if not event.opened_below_prior_close]
        summaries[symbol] = {
            "all_reaction_days": summarize(events),
            "reaction_opened_below_prior_close": summarize(lower),
            "reaction_did_not_open_below_prior_close": summarize(not_lower),
        }
        all_events.extend(asdict(event) for event in events)
    return rounded(
        {
            "schema": "sharpedge.apple_earnings_reaction_dips.v1",
            "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "source": "Yahoo Finance daily OHLC + curated AAPL earnings dates; verify dates before authority use",
            "assumption": "AAPL reports after close; reaction day is next trading session.",
            "symbols": sorted(symbol_events),
            "summaries": summaries,
            "events": all_events,
        }
    )


def write_csv(path: Path, events: list[AppleReactionEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(rounded(asdict(event)))


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Apple Earnings Reaction Lower-Open Study",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "Assumption: AAPL reports after close; reaction day is the next trading session.",
        "Dates are curated research inputs; verify before authority use.",
        "",
        "## Reaction day lower-open summary",
        "",
        "| Symbol | N | Opened below prior close | Closed below prior close | Median open gap | Median low vs prior close | Median lower-open streak when lower | Median 5d return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol in report["symbols"]:
        stats = report["summaries"][symbol]["all_reaction_days"]
        lines.append(
            "| "
            + " | ".join(
                [
                    symbol,
                    str(stats["count"]),
                    fmt_pct(stats["opened_below_prior_close_pct"]),
                    fmt_pct(stats["closed_below_prior_close_pct"]),
                    fmt_pct(stats["median_reaction_open_gap_pct"]),
                    fmt_pct(stats["median_reaction_low_vs_prior_close_pct"]),
                    str(stats["median_consecutive_lower_opens_from_reaction"]),
                    fmt_pct(stats["median_return_5d_pct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conditional: when the Apple reaction day opens lower",
            "",
            "| Symbol | N | Median lower-open streak | Median lower opens next 3 | Median lower opens next 5 | 1d positive | 3d positive | 5d positive | Median 5d return |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in report["symbols"]:
        stats = report["summaries"][symbol]["reaction_opened_below_prior_close"]
        lines.append(
            "| "
            + " | ".join(
                [
                    symbol,
                    str(stats["count"]),
                    str(stats["median_consecutive_lower_opens_from_reaction"]),
                    str(stats["median_lower_open_count_next_3_sessions"]),
                    str(stats["median_lower_open_count_next_5_sessions"]),
                    fmt_pct(stats["return_1d_positive_pct"]),
                    fmt_pct(stats["return_3d_positive_pct"]),
                    fmt_pct(stats["return_5d_positive_pct"]),
                    fmt_pct(stats["median_return_5d_pct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Trading read",
            "",
            "- If Apple is the next catalyst magnet, rejection may wait until the reaction session rather than the pre-FOMC session.",
            "- A reaction-day lower open is the key tell: it marks immediate post-earnings pressure from the prior close.",
            "- Persistence is measured as consecutive lower opens and lower-open counts over the next 3/5 sessions; this is a timing lens, not an execution signal by itself.",
            "- Use live reclaim acceptance/rejection, cockpit macro box, and NERV liquidity for actual timing.",
            "",
            "Research only. Broker-fresh quotes and operator approval required for any trade.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze lower opens after AAPL earnings.")
    parser.add_argument("--symbols", default="SPY,NVDA,AAPL")
    parser.add_argument("--range", default="10y", dest="data_range")
    parser.add_argument("--output-dir", default="outputs/apple_earnings_reaction_dips")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol_events: dict[str, list[AppleReactionEvent]] = {}
    for symbol in parse_symbols(args.symbols):
        candles = fetch_yahoo_daily(symbol, args.data_range)
        events = build_events(symbol, candles)
        if not events:
            raise RuntimeError(f"No Apple reaction events matched for {symbol}.")
        symbol_events[symbol] = events
        write_csv(output_dir / f"{symbol.lower()}_apple_reaction_dips.csv", events)

    report = build_report(symbol_events)
    json_path = output_dir / "apple_earnings_reaction_dips.json"
    md_path = output_dir / "apple_earnings_reaction_dips.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
