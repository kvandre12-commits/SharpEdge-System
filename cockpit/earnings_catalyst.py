"""Advisory earnings-catalyst verification for the hand-maintained radar.

`event_calendar.py` is the canonical, deterministic, network-free schedule.
This module does NOT mutate it. Instead it automates the "VERIFY each quarter"
toil: for each mega-cap earnings headliner it reports what the network says
(SEC latest filing + newness, and an IR-page-confirmed next date when the
conservative rule fires) versus the hardcoded list, and flags what is stale or
mismatched. The operator reviews the report and updates `EARNINGS_DATES` by hand
so the canonical source stays deterministic and reviewable.

Interpretation only. Fetch failures degrade to hardcoded-only status; nothing
here raises on a bad network.
"""

from __future__ import annotations

import datetime as dt

from catalyst_sources import (
    diff_filing_newness,
    fetch_issuer_earnings_date,
    fetch_latest_sec_filing,
    fetch_sec_ticker_cik_map,
)
from event_calendar import EARNINGS_DATES, MEGA_CAP_EARNINGS_HEADLINERS

# Per-ticker investor-relations sources for next-earnings-date confirmation.
# Optional and operator-extensible; VERIFY urls/keywords when adding a name.
# Tickers without an IR source still get SEC filing context via CIK resolution.
HEADLINER_IR_SOURCES: dict[str, dict] = {
    "AAPL": {
        "ir_url": "https://investor.apple.com/investor-relations/default.aspx",
        "keywords": ["earnings", "financial results", "quarter results"],
    },
    "NVDA": {
        "ir_url": "https://investor.nvidia.com/events-and-presentations/events-and-presentations/default.aspx",
        "keywords": ["earnings", "financial results", "quarter results"],
    },
}


def _upcoming(dates: list[str], today: dt.date) -> list[str]:
    floor = today.isoformat()
    return sorted(d for d in dates if d >= floor)


def build_earnings_catalyst_report(
    today: dt.date | None = None,
    *,
    headliners: tuple[str, ...] = MEGA_CAP_EARNINGS_HEADLINERS,
    ir_sources: dict[str, dict] | None = None,
    cik_map: dict[str, str] | None = None,
    previous_accessions: dict[str, str] | None = None,
) -> dict:
    """Build an advisory verification report over the earnings headliners.

    Never mutates `EARNINGS_DATES`. Returns a per-ticker breakdown plus rollups:
    ``needs_attention`` (no upcoming hardcoded date, none confirmed),
    ``suggestions`` (ticker -> IR-confirmed date when hardcoded is empty),
    ``mismatches`` (IR-confirmed date absent from the hardcoded upcoming list),
    and ``new_filings`` (accession changed vs ``previous_accessions``).
    """
    today = today or dt.date.today()  # noqa: DTZ011 - earnings are calendar dates, tz-naive by design
    ir_sources = HEADLINER_IR_SOURCES if ir_sources is None else ir_sources
    if cik_map is None:
        cik_map = fetch_sec_ticker_cik_map()
    previous_accessions = previous_accessions or {}

    tickers: dict[str, dict] = {}
    current_accessions: dict[str, str] = {}
    needs_attention: list[str] = []
    suggestions: dict[str, str] = {}
    mismatches: list[str] = []

    for ticker in headliners:
        hardcoded_upcoming = _upcoming(EARNINGS_DATES.get(ticker, []), today)

        ir_confirmed: str | None = None
        source = ir_sources.get(ticker)
        if source and source.get("ir_url"):
            ev = fetch_issuer_earnings_date(
                source["ir_url"],
                source.get("keywords", ["earnings"]),
                today=today,
            )
            if ev.get("confirmed"):
                ir_confirmed = ev.get("date")

        latest_filing: dict | None = None
        cik = cik_map.get(ticker.upper())
        if cik:
            filing = fetch_latest_sec_filing(cik)
            if filing.get("ok"):
                latest_filing = filing
                accession = filing.get("accession") or ""
                if accession:
                    current_accessions[ticker] = accession

        if hardcoded_upcoming:
            if ir_confirmed and ir_confirmed not in hardcoded_upcoming:
                status = "mismatch"
                mismatches.append(ticker)
            else:
                status = "ok"
        elif ir_confirmed:
            status = "suggestion"
            suggestions[ticker] = ir_confirmed
        else:
            status = "needs_attention"
            needs_attention.append(ticker)

        tickers[ticker] = {
            "status": status,
            "hardcoded_upcoming": hardcoded_upcoming,
            "ir_confirmed_date": ir_confirmed,
            "latest_filing": latest_filing,
            "cik": cik,
        }

    new_filings = diff_filing_newness(previous_accessions, current_accessions)

    return {
        "generated": today.isoformat(),
        "tickers": tickers,
        "needs_attention": needs_attention,
        "suggestions": suggestions,
        "mismatches": mismatches,
        "new_filings": new_filings,
        "current_accessions": current_accessions,
    }


def summarize(report: dict) -> list[str]:
    """Compact operator-readable lines from a catalyst report."""
    lines: list[str] = [f"Earnings catalyst report {report.get('generated', '')}"]
    if report.get("suggestions"):
        lines += [f"  SUGGEST {t}: confirm {d}" for t, d in report["suggestions"].items()]
    if report.get("mismatches"):
        lines.append(f"  MISMATCH (verify): {', '.join(report['mismatches'])}")
    if report.get("needs_attention"):
        lines.append(f"  NEEDS DATE: {', '.join(report['needs_attention'])}")
    if report.get("new_filings"):
        lines.append(f"  NEW SEC FILING (re-underwrite): {', '.join(report['new_filings'])}")
    if len(lines) == 1:
        lines.append("  all headliners have upcoming dates; no new filings")
    return lines


__all__ = [
    "HEADLINER_IR_SOURCES",
    "build_earnings_catalyst_report",
    "summarize",
]
