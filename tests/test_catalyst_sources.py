from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import catalyst_sources as cat

TODAY = dt.date(2026, 8, 21)


def test_strip_html_collapses_and_unescapes():
    raw = b"<html><body><h1>Q3   Results</h1>\n<p>Oct&nbsp;30, 2026</p></body></html>"
    assert cat.strip_html(raw) == "Q3 Results Oct 30, 2026"


def test_candidate_future_dates_promotes_single_keyword_local_date():
    text = "The board will host its earnings call on October 30, 2026 in New York."
    assert cat.candidate_future_dates(text, ["earnings"], TODAY) == [dt.date(2026, 10, 30)]


def test_candidate_future_dates_handles_day_first_form():
    text = "Financial results are due 5 November 2026 per the notice."
    assert cat.candidate_future_dates(text, ["financial results"], TODAY) == [
        dt.date(2026, 11, 5)
    ]


def test_candidate_future_dates_ignores_dates_far_from_keyword():
    text = "Unrelated milestone December 1, 2026." + (" filler" * 80) + " earnings soon."
    assert cat.candidate_future_dates(text, ["earnings"], TODAY) == []


def test_candidate_future_dates_drops_past_and_out_of_window():
    text = "earnings January 1, 2026 and earnings January 1, 2030"
    # past date and >180d date both excluded from an Aug 2026 anchor
    assert cat.candidate_future_dates(text, ["earnings"], TODAY) == []


def test_fetch_issuer_earnings_confirms_single_date():
    page = "<p>Next quarterly earnings: October 30, 2026.</p>"
    with patch.object(cat, "request_text_with_backoff", return_value=page):
        result = cat.fetch_issuer_earnings_date("http://ir", ["earnings"], today=TODAY)
    assert result["ok"] is True
    assert result["confirmed"] is True
    assert result["date"] == "2026-10-30"


def test_fetch_issuer_earnings_does_not_confirm_multiple_dates():
    page = "earnings October 30, 2026 ... earnings November 2, 2026"
    with patch.object(cat, "request_text_with_backoff", return_value=page):
        result = cat.fetch_issuer_earnings_date("http://ir", ["earnings"], today=TODAY)
    assert result["confirmed"] is False
    assert result["date"] is None
    assert len(result["candidates"]) == 2


def test_fetch_issuer_earnings_reports_fetch_failure_without_raising():
    with patch.object(cat, "request_text_with_backoff", side_effect=RuntimeError("boom")):
        result = cat.fetch_issuer_earnings_date("http://ir", ["earnings"], today=TODAY)
    assert result["ok"] is False
    assert "FETCH FAILED" in result["freshness"]


def test_fetch_latest_sec_filing_returns_most_recent():
    payload = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K"],
                "filingDate": ["2026-08-01", "2026-07-15"],
                "accessionNumber": ["0000851310-26-000045", "0000851310-26-000044"],
                "primaryDocument": ["hlit-20260801.htm", "hlit-8k.htm"],
            }
        }
    }
    with patch.object(cat, "request_json_with_backoff", return_value=payload):
        result = cat.fetch_latest_sec_filing("0000851310")
    assert result == {
        "ok": True,
        "form": "10-Q",
        "date": "2026-08-01",
        "accession": "0000851310-26-000045",
        "document": "hlit-20260801.htm",
    }


def test_fetch_latest_sec_filing_handles_empty_filings():
    with patch.object(cat, "request_json_with_backoff", return_value={"filings": {"recent": {}}}):
        result = cat.fetch_latest_sec_filing("0000851310")
    assert result == {"ok": True, "form": "", "date": "", "accession": "", "document": ""}


def test_fetch_latest_sec_filing_reports_failure_without_raising():
    with patch.object(cat, "request_json_with_backoff", side_effect=RuntimeError("429")):
        result = cat.fetch_latest_sec_filing("0000851310")
    assert result["ok"] is False
    assert "429" in result["error"]


def test_fetch_sec_ticker_cik_map_zero_pads_and_uppercases():
    payload = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA Corp"},
    }
    with patch.object(cat, "request_json_with_backoff", return_value=payload):
        mapping = cat.fetch_sec_ticker_cik_map()
    assert mapping == {"AAPL": "0000320193", "NVDA": "0001045810"}


def test_fetch_sec_ticker_cik_map_empty_on_failure():
    with patch.object(cat, "request_json_with_backoff", side_effect=RuntimeError("boom")):
        assert cat.fetch_sec_ticker_cik_map() == {}


def test_diff_filing_newness_baseline_flags_nothing():
    current = {"HLIT": "acc-1", "EXTR": "acc-2"}
    assert cat.diff_filing_newness({}, current) == []


def test_diff_filing_newness_flags_changed_accession_only():
    previous = {"HLIT": "acc-1", "EXTR": "acc-2"}
    current = {"HLIT": "acc-1", "EXTR": "acc-9", "SILC": "acc-3"}
    # EXTR changed; HLIT unchanged; SILC is new-baseline (not flagged)
    assert cat.diff_filing_newness(previous, current) == ["EXTR"]
