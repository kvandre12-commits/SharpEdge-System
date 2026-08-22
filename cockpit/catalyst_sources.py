"""Catalyst discovery: issuer earnings dates + SEC filing newness.

Two additive, network-backed reads that keep event radars honest without
hand-maintained calendars:

* ``fetch_issuer_earnings_date`` scrapes an official investor-relations page and
  promotes a date only under a conservative rule (exactly one future date in a
  keyword-local window), so noise never becomes a confirmed catalyst.
* ``fetch_latest_sec_filing`` reads the public SEC EDGAR submissions API for a
  CIK's most recent filing; ``diff_filing_newness`` flags when an accession has
  changed since a prior run (a re-underwrite trigger).

Interpretation only — nothing here authorizes a trade. HTTP goes through
``http_utils`` for shared retry/backoff and test mockability. The approach was
inspired by the N-S Terminal handoff and re-implemented here against the public
SEC EDGAR and IR-page contracts.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import re
from html.parser import HTMLParser

from http_utils import request_json_with_backoff, request_text_with_backoff

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# data.sec.gov requires a descriptive User-Agent identifying the caller.
DEFAULT_SEC_USER_AGENT_ENV = "SHARPEDGE_SEC_USER_AGENT"
_FALLBACK_SEC_USER_AGENT = "SharpEdge catalyst refresh (contact via repo owner)"

_EARNINGS_WINDOW_DAYS = 180

_MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_MONTH_ABBR = {name[:3]: index for name, index in _MONTHS.items()}


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text nodes, dropping tags/scripts markup."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        chunk = data.strip()
        if chunk:
            self._parts.append(chunk)

    def text(self) -> str:
        return " ".join(self._parts)


def strip_html(raw: bytes | str) -> str:
    """Reduce an HTML document to collapsed, entity-unescaped visible text."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(raw)
        text = extractor.text()
    except Exception:  # noqa: BLE001 - malformed markup falls back to raw text, never crashes
        text = raw
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def candidate_future_dates(
    text: str,
    keywords: list[str],
    today: dt.date,
    *,
    window_days: int = _EARNINGS_WINDOW_DAYS,
) -> list[dt.date]:
    """Return sorted future dates near any keyword, within ``window_days``.

    Only text windowed around a keyword occurrence is scanned, so a date that is
    not adjacent to earnings/results language is ignored. Matches both
    ``Month DD, YYYY`` and ``DD Month YYYY`` forms.
    """
    lowered = text.lower()
    windows: list[str] = []
    for keyword in keywords:
        start = 0
        needle = keyword.lower()
        while True:
            pos = lowered.find(needle, start)
            if pos < 0:
                break
            windows.append(text[max(0, pos - 220): pos + 320])
            start = pos + len(needle)
    if not windows:
        return []

    month_names = "|".join(_MONTHS) + "|" + "|".join(m.title() for m in _MONTH_ABBR)
    patterns = [
        re.compile(
            rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b",
            re.IGNORECASE,
        ),
    ]
    horizon = today + dt.timedelta(days=window_days)
    found: set[dt.date] = set()
    for window in windows:
        for pattern in patterns:
            for match in pattern.finditer(window):
                groups = match.groups()
                try:
                    if groups[0].isdigit():
                        day, month_token, year = int(groups[0]), groups[1], int(groups[2])
                    else:
                        month_token, day, year = groups[0], int(groups[1]), int(groups[2])
                    month = _MONTH_ABBR[month_token[:3].lower()]
                    candidate = dt.date(year, month, day)
                except (KeyError, ValueError):
                    continue
                if today <= candidate <= horizon:
                    found.add(candidate)
    return sorted(found)


def fetch_issuer_earnings_date(
    url: str,
    keywords: list[str],
    *,
    today: dt.date | None = None,
    timeout: int = 20,
) -> dict:
    """Discover a confirmable next-earnings date from an official IR page.

    Confirmation rule: exactly one keyword-local future date may be promoted.
    Returns ``ok``/``confirmed``/``date``/``freshness`` and never raises for a
    fetch failure (reports ``ok=False`` instead).
    """
    today = today or dt.date.today()  # noqa: DTZ011 - earnings are calendar dates, tz-naive by design
    try:
        raw = request_text_with_backoff(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - network/HTTP faults are reported as data (ok=False), not raised
        return {
            "ok": False,
            "confirmed": False,
            "date": None,
            "freshness": f"FETCH FAILED: {exc}",
            "note": str(exc),
        }

    text = strip_html(raw)
    dates = candidate_future_dates(text, keywords, today)
    confirmed = len(dates) == 1
    lowered = text.lower()
    if "no events scheduled" in lowered:
        freshness = "CURRENT: NO EVENTS SCHEDULED"
    elif "more events" in lowered and "coming" in lowered:
        freshness = "CURRENT: MORE EVENTS COMING SOON"
    elif confirmed:
        freshness = "CURRENT: ONE KEYWORD-LOCAL FUTURE DATE FOUND"
    else:
        freshness = "REFRESHED: NO UNIQUE FUTURE EARNINGS DATE IDENTIFIED"
    return {
        "ok": True,
        "confirmed": confirmed,
        "date": dates[0].isoformat() if confirmed else None,
        "candidates": [d.isoformat() for d in dates],
        "freshness": freshness,
        "note": "Confirmation requires one unique future date adjacent to earnings/results language.",
    }


def _sec_user_agent(user_agent: str | None) -> str:
    return user_agent or os.environ.get(
        DEFAULT_SEC_USER_AGENT_ENV, _FALLBACK_SEC_USER_AGENT
    )


def fetch_latest_sec_filing(
    cik: str,
    *,
    user_agent: str | None = None,
    timeout: int = 20,
) -> dict:
    """Return the most recent SEC filing for ``cik`` from the submissions API.

    ``cik`` is the zero-padded 10-digit CIK string. Never raises for a fetch
    failure (reports ``ok=False``).
    """
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    headers = {"User-Agent": _sec_user_agent(user_agent), "Accept": "application/json"}
    try:
        data = request_json_with_backoff(url, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - network/HTTP faults are reported as data (ok=False), not raised
        return {"ok": False, "error": str(exc)}

    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    if not forms:
        return {"ok": True, "form": "", "date": "", "accession": "", "document": ""}
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    return {
        "ok": True,
        "form": forms[0],
        "date": dates[0] if dates else "",
        "accession": accessions[0] if accessions else "",
        "document": documents[0] if documents else "",
    }


def fetch_sec_ticker_cik_map(
    *,
    user_agent: str | None = None,
    timeout: int = 20,
) -> dict[str, str]:
    """Return an upper-ticker -> zero-padded-10-digit-CIK map from SEC EDGAR.

    Robust CIK resolution so callers never hardcode (and mistype) a CIK.
    Returns an empty dict on fetch failure rather than raising.
    """
    headers = {"User-Agent": _sec_user_agent(user_agent), "Accept": "application/json"}
    try:
        data = request_json_with_backoff(SEC_TICKER_MAP_URL, headers=headers, timeout=timeout)
    except Exception:  # noqa: BLE001 - resolution failure degrades to empty map, never crashes
        return {}
    mapping: dict[str, str] = {}
    rows = data.values() if isinstance(data, dict) else data
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = str(cik).zfill(10)
    return mapping


def diff_filing_newness(
    previous: dict[str, str],
    current: dict[str, str],
) -> list[str]:
    """Return keys whose accession changed vs a prior run (re-underwrite flags).

    A key absent from ``previous`` is treated as a baseline, not a change, so the
    first run never floods every name as "new".
    """
    changed: list[str] = []
    for key, accession in current.items():
        if not accession:
            continue
        prior = previous.get(key)
        if prior is not None and prior != accession:
            changed.append(key)
    return sorted(changed)


__all__ = [
    "SEC_SUBMISSIONS_URL",
    "candidate_future_dates",
    "diff_filing_newness",
    "fetch_issuer_earnings_date",
    "fetch_latest_sec_filing",
    "fetch_sec_ticker_cik_map",
    "strip_html",
]
