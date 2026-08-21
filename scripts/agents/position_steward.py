#!/usr/bin/env python3
"""One-shot, scheduler-friendly position steward with no broker authority."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
COCKPIT = ROOT / "cockpit"
if str(COCKPIT) not in sys.path:
    sys.path.insert(0, str(COCKPIT))

from market_data_sources import (  # noqa: E402
    fetch_cboe_options_book,
    fetch_yahoo_regular_session_chart_rows,
)

try:  # noqa: E402
    from scripts.agents.position_steward_logic import (
        build_payload,
        build_session_snapshot,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from position_steward_logic import build_payload, build_session_snapshot

NY = ZoneInfo("America/New_York")


def _parse_timestamp(
    value: str | None, *, naive_zone: ZoneInfo = NY
) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_zone)
    return parsed.astimezone(UTC)


def _age_minutes(
    value: str | None, now: datetime, *, naive_zone: ZoneInfo = NY
) -> float | None:
    parsed = _parse_timestamp(value, naive_zone=naive_zone)
    if parsed is None:
        return None
    return round((now.astimezone(UTC) - parsed).total_seconds() / 60.0, 1)


def _market_open(now: datetime) -> bool:
    local = now.astimezone(NY)
    return local.weekday() < 5 and time(9, 30) <= local.time() < time(16, 0)


def _flatten_options_book(book: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    quotes: dict[str, dict[str, Any]] = {}
    for strikes in book.values():
        for contracts in strikes.values():
            for quote in contracts.values():
                contract = str(quote.get("option") or "").upper()
                if contract:
                    quotes[contract] = quote
    return quotes


def build_freshness(
    *,
    now: datetime,
    yahoo_source: dict[str, Any],
    cboe_source: dict[str, Any],
    session: dict[str, Any],
) -> dict[str, Any]:
    price_age = _age_minutes(yahoo_source.get("regular_market_time_utc"), now)
    option_age = _age_minutes(cboe_source.get("latest_option_trade_time_raw"), now)
    open_now = _market_open(now)
    local_date = now.astimezone(NY).date().isoformat()
    same_session = session.get("session_date") == local_date
    fresh = bool(
        open_now
        and same_session
        and price_age is not None
        and price_age <= 20
        and option_age is not None
        and option_age <= 45
    )
    return {
        "market_open": open_now,
        "fresh_for_management": fresh,
        "price_age_minutes": price_age,
        "option_age_minutes": option_age,
        "same_regular_session": same_session,
        "policy": "management requires open market, same-session bars, price <=20m, options <=45m",
    }


def render_text(payload: dict[str, Any]) -> str:
    action = payload.get("action") or {}
    recovery = payload.get("recovery") or {}
    position = payload.get("position") or {}
    freshness = payload.get("freshness") or {}
    authority = payload.get("authority") or {}
    lines = [
        f"# {payload.get('symbol')} Position Steward",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"State: {action.get('state')}",
        f"Reason: {action.get('reason')}",
        f"Spot: {payload.get('spot')}",
        f"Recovery: {recovery.get('recovery_pct')}% ({recovery.get('phase')})",
        f"Delta shares: {position.get('net_delta_shares')}",
        f"Theta/day: ${position.get('theta_dollars_per_day')}",
        f"Estimated P/L: ${position.get('estimated_pnl_dollars')}",
        f"Market open: {freshness.get('market_open')}",
        f"Fresh for management: {freshness.get('fresh_for_management')}",
        "",
        f"Authority: {authority.get('notice')}",
    ]
    return "\n".join(lines) + "\n"


def refresh(spec: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    symbol = str(spec.get("symbol") or "").upper()
    if symbol != "GME":
        raise ValueError("this approved steward run is restricted to GME")
    rows, yahoo_source = fetch_yahoo_regular_session_chart_rows(
        symbol,
        interval="5m",
        range_="5d",
        timeout=10,
    )
    cboe_spot, book, cboe_source = fetch_cboe_options_book(symbol, timeout=10)
    session = build_session_snapshot(rows)
    spot = float(cboe_spot or session.get("spot") or 0.0)
    freshness = build_freshness(
        now=now,
        yahoo_source=yahoo_source,
        cboe_source=cboe_source,
        session=session,
    )
    return build_payload(
        spec,
        generated_at=now.astimezone(UTC).isoformat(),
        spot=spot,
        session=session,
        quotes=_flatten_options_book(book),
        freshness=freshness,
        as_of=now.astimezone(NY).date(),
        sources={"price": yahoo_source, "options": cboe_source},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="required safety marker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.once:
        raise SystemExit("only bounded --once refreshes are supported")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    payload = refresh(spec)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_text.write_text(render_text(payload), encoding="utf-8")
    print(
        json.dumps({"state": payload["action"]["state"], "symbol": payload["symbol"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
