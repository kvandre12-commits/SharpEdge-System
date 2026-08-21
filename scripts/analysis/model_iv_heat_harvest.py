#!/usr/bin/env python3
"""Model option IV heat and likely harvest windows from a NERV snapshot.

This is a current-surface diagnostic, not a historical vol surface backtest.
It compares option implied volatility to recent Yahoo realized volatility and
labels event-week harvest pressure around known macro/single-name catalysts.
"""

from __future__ import annotations

import argparse
import json
import math
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

TRADING_DAYS = 252


@dataclass(frozen=True)
class Candle:
    date: str
    close: float


@dataclass(frozen=True)
class ExpiryIvRead:
    expiration: str
    dte_calendar: int
    atm_strike: float
    atm_call_iv_pct: float | None
    atm_put_iv_pct: float | None
    atm_iv_pct: float | None
    call_750_mid: float | None
    call_750_iv_pct: float | None
    call_750_width_pct: float | None
    call_750_volume: int | None
    call_750_open_interest: int | None
    iv_rv5_ratio: float | None
    iv_rv10_ratio: float | None
    iv_rv13_ratio: float | None
    iv_rv20_ratio: float | None
    heat_label: str
    nearest_event: str | None
    days_to_nearest_event: int | None
    harvest_window: str
    harvest_read: str


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
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 SharpEdge IV heat model"})
    with urlopen(req, timeout=30) as response:  # noqa: S310 - public market data
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo returned no result for {symbol}: {payload}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    candles: list[Candle] = []
    for index, ts in enumerate(timestamps):
        close = closes[index] if index < len(closes) else None
        if close is None:
            continue
        candles.append(
            Candle(
                date=datetime.fromtimestamp(ts, tz=UTC).date().isoformat(),
                close=float(close),
            )
        )
    return candles


def realized_vol_pct(candles: list[Candle], lookback: int) -> float | None:
    if len(candles) <= lookback:
        return None
    returns = []
    sample = candles[-(lookback + 1) :]
    for prev, current in zip(sample[:-1], sample[1:], strict=True):
        if prev.close <= 0 or current.close <= 0:
            continue
        returns.append(math.log(current.close / prev.close))
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS) * 100.0


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pct_iv(value: Any) -> float | None:
    raw = safe_float(value)
    if raw is None:
        return None
    return raw * 100.0 if raw <= 3 else raw


def ratio(iv_pct: float | None, rv_pct: float | None) -> float | None:
    if iv_pct is None or rv_pct is None or rv_pct <= 0:
        return None
    return iv_pct / rv_pct


def heat_label(iv_rv13: float | None, fallback_ratio: float | None) -> str:
    primary = iv_rv13 if iv_rv13 is not None else fallback_ratio
    if primary is None:
        return "unknown"
    if primary >= 1.75:
        return "very_hot"
    if primary >= 1.35:
        return "hot"
    if primary >= 1.1:
        return "warm"
    if primary >= 0.85:
        return "fair"
    return "cool"


def sorted_events() -> list[tuple[str, date]]:
    events = [("FOMC", date.fromisoformat(item)) for item in FOMC_DATES]
    events.extend(("AAPL earnings", date.fromisoformat(item)) for item in AAPL_EARNINGS_DATES)
    return sorted(events, key=lambda item: item[1])


def nearest_event(anchor: date) -> tuple[str | None, int | None]:
    future = [(name, (event_date - anchor).days) for name, event_date in sorted_events() if event_date >= anchor]
    if not future:
        return None, None
    name, days = min(future, key=lambda item: item[1])
    return name, days


def harvest_window(expiration: date, anchor: date) -> tuple[str, str]:
    events = sorted_events()
    future = [(name, event_date) for name, event_date in events if event_date >= anchor]
    if not future:
        return "no_known_event", "No known near event in calendar; harvest depends on realized path."
    next_name, next_date = future[0]
    days_to = (next_date - anchor).days
    days_after_event = (expiration - next_date).days
    if expiration < next_date:
        return (
            "expires_before_event",
            f"Expires before {next_name}; it can harvest pre-event decay but misses post-event vol crush.",
        )
    if days_to <= 3 and 0 <= days_after_event <= 2:
        return (
            "event_crush_window",
            f"Expires right after {next_name}; IV harvest risk is high immediately after event resolution.",
        )
    if days_to <= 3 and 3 <= days_after_event <= 10:
        return (
            "post_event_harvest_plus_time",
            f"Survives {next_name} with some post-event time; better for waiting on IV harvest than front expiry.",
        )
    if days_to <= 3 and days_after_event > 10:
        return (
            "event_plus_month_time",
            f"Carries through {next_name}; event IV may be harvested while monthly time value remains.",
        )
    return (
        "ordinary_term_decay",
        f"Next known event is {next_name} in {days_to} calendar days; no immediate harvest label.",
    )


def choose_atm(rows: list[dict[str, Any]], underlying: float) -> tuple[float, dict[str, Any] | None, dict[str, Any] | None]:
    strikes = sorted({safe_float(row.get("strike")) for row in rows if safe_float(row.get("strike")) is not None})
    if not strikes:
        return 0.0, None, None
    atm = min(strikes, key=lambda strike: abs(strike - underlying))
    call = next((row for row in rows if row.get("option_type") == "call" and safe_float(row.get("strike")) == atm), None)
    put = next((row for row in rows if row.get("option_type") == "put" and safe_float(row.get("strike")) == atm), None)
    return atm, call, put


def choose_call(rows: list[dict[str, Any]], strike: float) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("option_type") == "call" and safe_float(row.get("strike")) == strike
        ),
        None,
    )


def read_snapshot(snapshot_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = payload.get("quotes") or payload.get("contracts") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No quotes found in snapshot: {snapshot_path}")
    return rows


def build_reads(
    rows: list[dict[str, Any]],
    *,
    anchor_date: date,
    rv5: float | None,
    rv10: float | None,
    rv13: float | None,
    rv20: float | None,
    target_strike: float,
) -> tuple[float, list[ExpiryIvRead]]:
    underlying = safe_float(rows[0].get("underlying_price")) or 0.0
    reads: list[ExpiryIvRead] = []
    expirations = sorted({str(row.get("expiration")) for row in rows if row.get("expiration")})
    event_name, event_days = nearest_event(anchor_date)
    for expiration in expirations:
        expiry_date = date.fromisoformat(expiration)
        exp_rows = [row for row in rows if row.get("expiration") == expiration]
        atm, atm_call, atm_put = choose_atm(exp_rows, underlying)
        target_call = choose_call(exp_rows, target_strike)
        call_iv = pct_iv(atm_call.get("implied_volatility")) if atm_call else None
        put_iv = pct_iv(atm_put.get("implied_volatility")) if atm_put else None
        iv_values = [value for value in (call_iv, put_iv) if value is not None]
        atm_iv = statistics.mean(iv_values) if iv_values else None
        iv_rv5 = ratio(atm_iv, rv5)
        iv_rv10 = ratio(atm_iv, rv10)
        iv_rv13 = ratio(atm_iv, rv13)
        iv_rv20 = ratio(atm_iv, rv20)
        window, read = harvest_window(expiry_date, anchor_date)
        reads.append(
            ExpiryIvRead(
                expiration=expiration,
                dte_calendar=(expiry_date - anchor_date).days,
                atm_strike=atm,
                atm_call_iv_pct=call_iv,
                atm_put_iv_pct=put_iv,
                atm_iv_pct=atm_iv,
                call_750_mid=safe_float(target_call.get("midpoint")) if target_call else None,
                call_750_iv_pct=pct_iv(target_call.get("implied_volatility")) if target_call else None,
                call_750_width_pct=safe_float(target_call.get("width_pct")) if target_call else None,
                call_750_volume=safe_int(target_call.get("volume")) if target_call else None,
                call_750_open_interest=safe_int(target_call.get("open_interest")) if target_call else None,
                iv_rv5_ratio=iv_rv5,
                iv_rv10_ratio=iv_rv10,
                iv_rv13_ratio=iv_rv13,
                iv_rv20_ratio=iv_rv20,
                heat_label=heat_label(iv_rv13, iv_rv20),
                nearest_event=event_name,
                days_to_nearest_event=event_days,
                harvest_window=window,
                harvest_read=read,
            )
        )
    return underlying, reads


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def fmt_ratio(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {report['symbol']} IV Heat / Harvest Read",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Underlying: `{report['underlying_price']}`; target strike: `{report['target_strike']}`.",
        f"Realized vol proxy: RV5 `{report['realized_vol']['rv5_pct']}%`, RV10 `{report['realized_vol']['rv10_pct']}%`, RV13 `{report['realized_vol']['rv13_pct']}%`, RV20 `{report['realized_vol']['rv20_pct']}%`.",
        "",
        "## Expiry heat table",
        "",
        "| Expiry | DTE | ATM IV | IV/RV13 | IV/RV20 | Heat | 750C mid | 750C IV | 750C OI | Harvest window |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in report["expiry_reads"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["expiration"],
                    str(row["dte_calendar"]),
                    fmt_pct(row["atm_iv_pct"]),
                    fmt_ratio(row["iv_rv13_ratio"]),
                    fmt_ratio(row["iv_rv20_ratio"]),
                    row["heat_label"],
                    "n/a" if row["call_750_mid"] is None else str(row["call_750_mid"]),
                    fmt_pct(row["call_750_iv_pct"]),
                    "n/a" if row["call_750_open_interest"] is None else str(row["call_750_open_interest"]),
                    row["harvest_window"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            f"- Overall IV heat: **{report['overall_heat_label']}**.",
            f"- Nearest event: `{report['nearest_event']}` in `{report['days_to_nearest_event']}` calendar days.",
            "- 'Hot' means option IV is elevated versus recent realized volatility; it does not mean direction is wrong.",
            "- IV is usually harvested after event uncertainty resolves: FOMC statement/press conference, then AAPL reaction session in this stack.",
            "- For a month-ish 750C, the cleaner idea is often to let front-event IV bleed first, then buy time if the reclaim path survives.",
            "",
            "Research only. Broker-fresh quotes and operator approval required for any trade.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_iv_heat_report(
    *,
    symbol: str,
    snapshot_path: str | Path,
    data_range: str = "6mo",
    target_strike: float = 750.0,
    anchor_date: str = "",
) -> dict[str, Any]:
    symbol = symbol.upper()
    candles = fetch_yahoo_daily(symbol, data_range)
    anchor = date.fromisoformat(anchor_date or candles[-1].date)
    rv5 = realized_vol_pct(candles, 5)
    rv10 = realized_vol_pct(candles, 10)
    rv13 = realized_vol_pct(candles, 13)
    rv20 = realized_vol_pct(candles, 20)
    rows = read_snapshot(Path(snapshot_path))
    symbol_rows = [row for row in rows if str(row.get("underlying", "")).upper() == symbol]
    rows = symbol_rows or rows
    underlying, reads = build_reads(
        rows,
        anchor_date=anchor,
        rv5=rv5,
        rv10=rv10,
        rv13=rv13,
        rv20=rv20,
        target_strike=target_strike,
    )
    event_name, event_days = nearest_event(anchor)
    heat_values = [read.iv_rv13_ratio for read in reads if read.iv_rv13_ratio is not None]
    median_heat = statistics.median(heat_values) if heat_values else None
    return rounded(
        {
            "schema": "sharpedge.iv_heat_harvest.v1",
            "symbol": symbol,
            "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "snapshot": str(snapshot_path),
            "anchor_date": anchor.isoformat(),
            "underlying_price": underlying,
            "target_strike": target_strike,
            "realized_vol": {"rv5_pct": rv5, "rv10_pct": rv10, "rv13_pct": rv13, "rv20_pct": rv20},
            "nearest_event": event_name,
            "days_to_nearest_event": event_days,
            "median_iv_rv13_ratio": median_heat,
            "overall_heat_label": heat_label(median_heat, None),
            "expiry_reads": [asdict(read) for read in reads],
        }
    )


def write_iv_heat_report(
    *,
    symbol: str,
    snapshot_path: str | Path,
    output_dir: str | Path,
    data_range: str = "6mo",
    target_strike: float = 750.0,
    anchor_date: str = "",
) -> dict[str, str]:
    report = build_iv_heat_report(
        symbol=symbol,
        snapshot_path=snapshot_path,
        data_range=data_range,
        target_strike=target_strike,
        anchor_date=anchor_date,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{symbol.lower()}_iv_heat_harvest.json"
    md_path = out / f"{symbol.lower()}_iv_heat_harvest.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IV heat/harvest read from NERV snapshot.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--range", default="6mo", dest="data_range")
    parser.add_argument("--target-strike", type=float, default=750.0)
    parser.add_argument("--anchor-date", default="")
    parser.add_argument("--output-dir", default="outputs/iv_heat_harvest")
    args = parser.parse_args()
    paths = write_iv_heat_report(
        symbol=args.symbol,
        snapshot_path=args.snapshot,
        output_dir=args.output_dir,
        data_range=args.data_range,
        target_strike=args.target_strike,
        anchor_date=args.anchor_date,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
