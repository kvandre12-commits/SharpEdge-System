#!/usr/bin/env python3
"""Analyze pre-FOMC trading days from Yahoo daily OHLC.

The study is intentionally daily/OHLC only: it does not claim intraday tape truth.
It answers: on the session before FOMC, did SPY open below/above prior close,
reclaim prior close intraday, reject, and what happened on FOMC day?
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
if str(COCKPIT) not in sys.path:
    sys.path.insert(0, str(COCKPIT))

from event_calendar import FOMC_DATES as CANONICAL_FOMC_DATES  # noqa: E402


# Manually curated FOMC decision dates for larger base-rate context.
# Keep this study labeled as "research"; official calendar should be verified.
HISTORICAL_FOMC_DATES = [
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
    "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03",
    "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04",
    "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02",
    "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03",
    "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01",
    "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01",
    "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
    "2024-12-18",
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
class PreFomcEvent:
    fomc_date: str
    pre_date: str
    pre_open_vs_prior_close_pct: float
    pre_low_vs_prior_close_pct: float
    pre_close_vs_prior_close_pct: float
    pre_open_side: str
    pre_reclaimed_prior_close_intraday: bool
    pre_rejected_prior_close: bool
    pre_closed_green: bool
    fomc_open_vs_pre_close_pct: float | None
    fomc_close_vs_pre_close_pct: float | None
    fomc_intraday_high_vs_pre_close_pct: float | None
    fomc_intraday_low_vs_pre_close_pct: float | None
    next_close_vs_pre_close_pct: float | None


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
        headers={"User-Agent": "Mozilla/5.0 SharpEdge FOMC predate study"},
    )
    with urlopen(req, timeout=30) as response:  # noqa: S310 - public market data
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo returned no result: {payload.get('chart', {}).get('error')}")
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


def previous_trading_index(candles: list[Candle], target: str) -> int | None:
    target_date = date.fromisoformat(target)
    candidate: int | None = None
    for index, candle in enumerate(candles):
        candle_date = date.fromisoformat(candle.date)
        if candle_date < target_date:
            candidate = index
        elif candle_date == target_date:
            return index - 1 if index > 0 else None
        else:
            return candidate
    return candidate


def build_events(candles: list[Candle], fomc_dates: list[str]) -> list[PreFomcEvent]:
    events: list[PreFomcEvent] = []
    by_date = {candle.date: index for index, candle in enumerate(candles)}
    for fomc_date in sorted(set(fomc_dates)):
        fomc_index = by_date.get(fomc_date)
        pre_index = previous_trading_index(candles, fomc_date)
        if fomc_index is None or pre_index is None or pre_index <= 0:
            continue
        pre = candles[pre_index]
        prior = candles[pre_index - 1]
        fomc = candles[fomc_index]
        next_close = candles[fomc_index + 1].close if fomc_index + 1 < len(candles) else None
        open_side = "below_prior_close" if pre.open < prior.close else "above_prior_close"
        reclaimed = pre.high >= prior.close
        rejected = reclaimed and pre.close < prior.close
        events.append(
            PreFomcEvent(
                fomc_date=fomc.date,
                pre_date=pre.date,
                pre_open_vs_prior_close_pct=pct(pre.open, prior.close),
                pre_low_vs_prior_close_pct=pct(pre.low, prior.close),
                pre_close_vs_prior_close_pct=pct(pre.close, prior.close),
                pre_open_side=open_side,
                pre_reclaimed_prior_close_intraday=reclaimed,
                pre_rejected_prior_close=rejected,
                pre_closed_green=pre.close > pre.open,
                fomc_open_vs_pre_close_pct=pct(fomc.open, pre.close),
                fomc_close_vs_pre_close_pct=pct(fomc.close, pre.close),
                fomc_intraday_high_vs_pre_close_pct=pct(fomc.high, pre.close),
                fomc_intraday_low_vs_pre_close_pct=pct(fomc.low, pre.close),
                next_close_vs_pre_close_pct=None if next_close is None else pct(next_close, pre.close),
            )
        )
    return events


def median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.median(finite)


def hit_rate(events: list[PreFomcEvent], predicate: str) -> float | None:
    if not events:
        return None
    hits = sum(bool(getattr(event, predicate)) for event in events)
    return hits / len(events) * 100.0


def positive_rate(events: list[PreFomcEvent], field: str) -> float | None:
    values = [getattr(event, field) for event in events if getattr(event, field) is not None]
    if not values:
        return None
    return sum(value > 0 for value in values) / len(values) * 100.0


def summarize(events: list[PreFomcEvent]) -> dict[str, Any]:
    return {
        "count": len(events),
        "pre_reclaim_intraday_pct": hit_rate(events, "pre_reclaimed_prior_close_intraday"),
        "pre_reject_after_reclaim_pct": hit_rate(events, "pre_rejected_prior_close"),
        "pre_green_close_pct": hit_rate(events, "pre_closed_green"),
        "median_pre_open_vs_prior_close_pct": median([e.pre_open_vs_prior_close_pct for e in events]),
        "median_pre_close_vs_prior_close_pct": median([e.pre_close_vs_prior_close_pct for e in events]),
        "fomc_positive_close_pct": positive_rate(events, "fomc_close_vs_pre_close_pct"),
        "median_fomc_close_vs_pre_close_pct": median([e.fomc_close_vs_pre_close_pct for e in events]),
        "median_fomc_high_vs_pre_close_pct": median([e.fomc_intraday_high_vs_pre_close_pct for e in events]),
        "median_fomc_low_vs_pre_close_pct": median([e.fomc_intraday_low_vs_pre_close_pct for e in events]),
        "next_day_positive_close_pct": positive_rate(events, "next_close_vs_pre_close_pct"),
        "median_next_close_vs_pre_close_pct": median([e.next_close_vs_pre_close_pct for e in events]),
    }


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def build_report(symbol: str, candles: list[Candle], events: list[PreFomcEvent]) -> dict[str, Any]:
    below = [event for event in events if event.pre_open_side == "below_prior_close"]
    above = [event for event in events if event.pre_open_side == "above_prior_close"]
    rejected = [event for event in events if event.pre_rejected_prior_close]
    reclaimed = [event for event in events if event.pre_reclaimed_prior_close_intraday]
    return rounded(
        {
            "schema": "sharpedge.fomc_predate_study.v1",
            "symbol": symbol,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "source": "Yahoo Finance daily OHLC + cockpit/event_calendar FOMC dates + manually curated historical FOMC dates",
            "candles": len(candles),
            "start_date": candles[0].date,
            "end_date": candles[-1].date,
            "event_count": len(events),
            "summaries": {
                "all_pre_fomc": summarize(events),
                "pre_open_below_prior_close": summarize(below),
                "pre_open_above_prior_close": summarize(above),
                "pre_reclaimed_prior_close_intraday": summarize(reclaimed),
                "pre_rejected_after_reclaim": summarize(rejected),
            },
            "recent_events": [asdict(event) for event in events[-16:]],
            "all_events": [asdict(event) for event in events],
        }
    )


def write_csv(path: Path, events: list[PreFomcEvent]) -> None:
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
    summaries = report["summaries"]
    lines = [
        f"# {report['symbol']} Pre-FOMC Date Study",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Data: `{report['start_date']}` to `{report['end_date']}`; {report['event_count']} FOMC pre-date events.",
        "",
        "## Cohort summary",
        "",
        "| Cohort | N | Pre reclaim intraday | Pre reject after reclaim | FOMC positive close | Median FOMC close | Median FOMC high | Median FOMC low |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name.replace("_", " "),
                    str(stats["count"]),
                    fmt_pct(stats["pre_reclaim_intraday_pct"]),
                    fmt_pct(stats["pre_reject_after_reclaim_pct"]),
                    fmt_pct(stats["fomc_positive_close_pct"]),
                    fmt_pct(stats["median_fomc_close_vs_pre_close_pct"]),
                    fmt_pct(stats["median_fomc_high_vs_pre_close_pct"]),
                    fmt_pct(stats["median_fomc_low_vs_pre_close_pct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Read for tomorrow",
            "",
            "- Treat the pre-FOMC session as a positioning/attempt day, not a normal random Tuesday.",
            "- If price starts below the reclaim line and rejects again, historical framing favors respecting the failed reclaim instead of waiting for a prettier late signal.",
            "- If price accepts above the reclaim line, bearish defined-risk structures should be reduced or invalidated before FOMC whipsaw.",
            "- This study uses daily OHLC only; live cockpit levels, VWAP, gamma, and NERV liquidity must drive execution timing.",
            "",
            "Research only. Broker-fresh quotes and operator approval required for any trade.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SPY pre-FOMC date behavior.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--range", default="10y", dest="data_range")
    parser.add_argument("--output-dir", default="outputs/fomc_pre_dates")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    candles = fetch_yahoo_daily(symbol, args.data_range)
    fomc_dates = sorted(set(HISTORICAL_FOMC_DATES + list(CANONICAL_FOMC_DATES)))
    events = build_events(candles, fomc_dates)
    if not events:
        raise RuntimeError("No FOMC pre-date events matched the fetched OHLC data.")
    report = build_report(symbol, candles, events)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{symbol.lower()}_fomc_predates.json"
    csv_path = output_dir / f"{symbol.lower()}_fomc_predates.csv"
    md_path = output_dir / f"{symbol.lower()}_fomc_predates.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, events)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
