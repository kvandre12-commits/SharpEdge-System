"""Macro-catalyst event radar for the SharpEdge cockpit.

Canonical, single-source schedule of high-impact scheduled events:
  - FOMC   : Fed decision days (2pm ET statement)              overlay_type='fomc'
  - JOBS   : Non-Farm Payrolls, first Friday, 8:30am ET        overlay_type='jobs'
  - TREASURY: Quarterly Refunding Announcement (QRA)           overlay_type='treasury'

Both the live cockpit radar (build_event_radar_live) and the batch seeder
(scripts/seed_event_overlays.py) read the SAME lists here, so the cockpit flag
and the overlays_daily event rows can never disagree.

NFP is COMPUTED (first Friday of each month). FOMC and QRA are SCHEDULED dates
that must be maintained by hand — VERIFY the lists below against the official
Fed / US Treasury calendars before trusting them.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

# ---------------------------------------------------------------------------
# VERIFY THESE against federalreserve.gov and treasurydirect.gov before trusting.
# FOMC = second (decision) day; 2pm ET statement.
# ---------------------------------------------------------------------------
FOMC_DATES: list[str] = [
    # 2025 (verify)
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    # 2026 (VERIFY — likely dates, confirm official schedule)
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]

# Treasury Quarterly Refunding Announcement (Wednesday refunding statement).
# 2026 VERIFIED against Treasury schedule. 2025 still verify.
TREASURY_QRA_DATES: list[str] = [
    # 2025 (verify)
    "2025-02-05",
    "2025-05-07",
    "2025-07-30",
    "2025-11-05",
    # 2026 (VERIFIED)
    "2026-02-04",
    "2026-05-06",
    "2026-08-05",
    "2026-11-04",
]

# Optional: big coupon auctions (10Y/30Y). Left empty by default — populate if
# you want auction days on the radar. VERIFY against treasurydirect.gov.
TREASURY_AUCTION_DATES: list[str] = []

# Single-name earnings headliners. VERIFY each quarter (dates shift; confirm the
# session AND whether it's before open (BMO) or after close (AMC)).
#
# Keep the symbol universe broader than the currently verified dates so the
# cockpit can tell us which tracked competitors/watchlist names are still
# missing confirmed entries instead of pretending the list is complete.
MEGA_CAP_EARNINGS_HEADLINERS: tuple[str, ...] = (
    "AAPL",
    "NVDA",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "PLTR",
    "MCHP",
    "RDDT",
)

EARNINGS_DATES: dict[str, list[str]] = {
    "AAPL": [
        "2026-07-30",  # FQ3 — AMC
    ],
    "NVDA": [
        "2026-08-26",  # Q2 FY27 — AMC
    ],
    "MSFT": [],
    "AMZN": [],
    "GOOGL": [],
    "META": [],
    "TSLA": [],
    "AMD": [
        "2026-08-04",  # Nasdaq calendar — after hours
    ],
    "NFLX": [],
    "PLTR": [
        "2026-08-03",  # Nasdaq calendar — after hours
    ],
    "MCHP": [
        "2026-08-06",  # Nasdaq calendar — time not supplied
    ],
    "RDDT": [],
}

EVENT_LABELS = {
    "fomc": "FOMC decision",
    "jobs": "Jobs report (NFP)",
    "treasury": "Treasury refunding (QRA)",
    "treasury_auction": "Treasury auction",
    "earnings": "earnings",
}


def event_label(event: dict[str, str]) -> str:
    """Human label for an event, specializing earnings by ticker."""
    etype = event.get("type", "")
    if etype == "earnings":
        ticker = event.get("ticker", "")
        return f"{ticker} earnings".strip()
    return EVENT_LABELS.get(etype, etype)


def _first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    # weekday(): Mon=0 .. Fri=4
    offset = (4 - d.weekday()) % 7
    return d + dt.timedelta(days=offset)


def nfp_dates_in_range(start: dt.date, end: dt.date) -> list[str]:
    """First Friday of each month in [start, end] (NFP approximation)."""
    out: list[str] = []
    y, m = start.year, start.month
    while dt.date(y, m, 1) <= end:
        ff = _first_friday(y, m)
        if start <= ff <= end:
            out.append(ff.isoformat())
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def all_events_in_range(start: dt.date, end: dt.date) -> list[dict[str, str]]:
    """Return every scheduled event (type,date[,ticker]) within [start, end], sorted."""
    lo, hi = start.isoformat(), end.isoformat()
    events: list[dict[str, str]] = []
    for d in FOMC_DATES:
        if lo <= d <= hi:
            events.append({"type": "fomc", "date": d})
    for d in TREASURY_QRA_DATES:
        if lo <= d <= hi:
            events.append({"type": "treasury", "date": d})
    for d in TREASURY_AUCTION_DATES:
        if lo <= d <= hi:
            events.append({"type": "treasury_auction", "date": d})
    for ticker, dates in EARNINGS_DATES.items():
        for d in dates:
            if lo <= d <= hi:
                events.append({"type": "earnings", "date": d, "ticker": ticker})
    for d in nfp_dates_in_range(start, end):
        events.append({"type": "jobs", "date": d})
    events.sort(key=lambda e: (e["date"], e["type"]))
    return events


def earnings_headliner_status() -> dict[str, list[str]]:
    """Return which mega-cap earnings headliners have verified dates loaded."""
    tracked: list[str] = []
    missing: list[str] = []
    for ticker in MEGA_CAP_EARNINGS_HEADLINERS:
        if EARNINGS_DATES.get(ticker):
            tracked.append(ticker)
        else:
            missing.append(ticker)
    return {"tracked": tracked, "missing": missing}


def build_event_radar_live(
    today: dt.date | None = None,
    *,
    lookahead_days: int = 45,
    include_social: bool = False,
) -> dict[str, Any]:
    """Compute the macro-catalyst radar around `today`.

    Returns events happening today plus the nearest upcoming events, with a
    high-impact risk-window flag (event today or tomorrow).
    """
    today = today or dt.date.today()
    window_start = today - dt.timedelta(days=1)
    window_end = today + dt.timedelta(days=lookahead_days)
    events = all_events_in_range(window_start, window_end)

    today_iso = today.isoformat()
    events_today = [e for e in events if e["date"] == today_iso]
    upcoming = [e for e in events if e["date"] >= today_iso]

    def _decorate(e: dict[str, str]) -> dict[str, Any]:
        d = dt.date.fromisoformat(e["date"])
        days_to = (d - today).days
        return {
            "type": e["type"],
            "date": e["date"],
            "days_to": days_to,
            "ticker": e.get("ticker"),
            "label": event_label(e),
        }

    upcoming_dec = [_decorate(e) for e in upcoming][:5]
    next_event = upcoming_dec[0] if upcoming_dec else None

    # High-impact window: any tracked event today or tomorrow.
    risk_window = any(0 <= e["days_to"] <= 1 for e in upcoming_dec)
    social_catalyst: dict[str, Any] = {}
    if include_social:
        try:
            from truth_social_scanner import build_truth_social_event_scan

            social_catalyst = build_truth_social_event_scan()
        except Exception as exc:
            social_catalyst = {
                "schema": "sharpedge.truth_social_event_scan.v1",
                "available": False,
                "headline": "Trump Truth scanner unavailable",
                "story": f"Truth Social scanner failed: {exc}",
                "source_status": {"ok": False, "status": "scanner_error"},
            }
    latest_social = social_catalyst.get("latest_relevant") or {}
    if latest_social and latest_social.get("impact") in {"medium", "high"}:
        risk_window = True

    social_story = ""
    if latest_social:
        social_story = f" Social catalyst watch: {social_catalyst.get('story') or latest_social.get('text')}"
    elif social_catalyst.get("source_status"):
        status = (social_catalyst.get("source_status") or {}).get("status")
        if status == "ok":
            count = int(social_catalyst.get("status_count") or 0)
            social_story = f" Social scanner: no market-relevant Trump Truth in latest {count} posts."
        elif status in {"source_blocked", "request_failed"}:
            social_story = f" Social scanner status: {status}."

    earnings_status = earnings_headliner_status()

    if events_today:
        labels = ", ".join(event_label(e) for e in events_today)
        headline = f"{labels} TODAY"
        story = (
            f"{labels} hits today — expect compression into the release, then an "
            "expansion/volatility window. Size down pre-event; respect the whipsaw."
            f"{social_story}"
        )
    elif next_event and next_event["days_to"] <= 1:
        headline = f"{next_event['label']} TOMORROW"
        story = f"{next_event['label']} on {next_event['date']} — pre-event drift/compression risk.{social_story}"
    elif next_event:
        headline = f"{next_event['label']} in {next_event['days_to']}d"
        story = f"Next macro catalyst: {next_event['label']} on {next_event['date']}.{social_story}"
    else:
        headline = "No scheduled catalyst in window"
        story = f"No FOMC / jobs / Treasury refunding within the lookahead window.{social_story}"

    return {
        "schema": "sharpedge.event_radar.v1",
        "available": True,
        "headline": headline,
        "risk_window": risk_window,
        "events_today": [_decorate(e) for e in events_today],
        "next_event": next_event,
        "upcoming": upcoming_dec,
        "story": story,
        "earnings_headliners": {
            "tracked": earnings_status["tracked"],
            "missing_verified_dates": earnings_status["missing"],
        },
        "social_catalyst": social_catalyst,
        "source": "canonical:event_calendar+truth_social"
        if include_social
        else "canonical:event_calendar",
    }


__all__ = [
    "EARNINGS_DATES",
    "EVENT_LABELS",
    "FOMC_DATES",
    "MEGA_CAP_EARNINGS_HEADLINERS",
    "TREASURY_AUCTION_DATES",
    "TREASURY_QRA_DATES",
    "all_events_in_range",
    "build_event_radar_live",
    "earnings_headliner_status",
    "event_label",
    "nfp_dates_in_range",
]
