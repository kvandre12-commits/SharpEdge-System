from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import earnings_catalyst as ec

TODAY = dt.date(2026, 8, 21)


def _ir(confirmed_date):
    return {"ok": True, "confirmed": bool(confirmed_date), "date": confirmed_date}


def _filing(accession):
    return {
        "ok": True,
        "form": "8-K",
        "date": "2026-08-01",
        "accession": accession,
        "document": "d.htm",
    }


def test_status_ok_when_hardcoded_upcoming_present():
    with patch.object(ec, "EARNINGS_DATES", {"AAPL": ["2026-09-30"]}), patch.object(
        ec, "fetch_latest_sec_filing", return_value=_filing("acc-1")
    ):
        report = ec.build_earnings_catalyst_report(
            TODAY, headliners=("AAPL",), ir_sources={}, cik_map={"AAPL": "0000320193"}
        )
    assert report["tickers"]["AAPL"]["status"] == "ok"
    assert report["needs_attention"] == []


def test_suggestion_when_empty_and_ir_confirms():
    with patch.object(ec, "EARNINGS_DATES", {"MSFT": []}), patch.object(
        ec, "fetch_issuer_earnings_date", return_value=_ir("2026-10-28")
    ), patch.object(ec, "fetch_latest_sec_filing", return_value=_filing("acc-2")):
        report = ec.build_earnings_catalyst_report(
            TODAY,
            headliners=("MSFT",),
            ir_sources={"MSFT": {"ir_url": "http://ir", "keywords": ["earnings"]}},
            cik_map={"MSFT": "0000789019"},
        )
    assert report["tickers"]["MSFT"]["status"] == "suggestion"
    assert report["suggestions"] == {"MSFT": "2026-10-28"}


def test_needs_attention_when_empty_and_no_ir_source():
    with patch.object(ec, "EARNINGS_DATES", {"TSLA": []}), patch.object(
        ec, "fetch_latest_sec_filing", return_value=_filing("acc-3")
    ):
        report = ec.build_earnings_catalyst_report(
            TODAY, headliners=("TSLA",), ir_sources={}, cik_map={"TSLA": "0001318605"}
        )
    assert report["tickers"]["TSLA"]["status"] == "needs_attention"
    assert report["needs_attention"] == ["TSLA"]


def test_mismatch_when_ir_date_differs_from_hardcoded():
    with patch.object(ec, "EARNINGS_DATES", {"AAPL": ["2026-09-30"]}), patch.object(
        ec, "fetch_issuer_earnings_date", return_value=_ir("2026-10-30")
    ), patch.object(ec, "fetch_latest_sec_filing", return_value=_filing("acc-4")):
        report = ec.build_earnings_catalyst_report(
            TODAY,
            headliners=("AAPL",),
            ir_sources={"AAPL": {"ir_url": "http://ir", "keywords": ["earnings"]}},
            cik_map={"AAPL": "0000320193"},
        )
    assert report["tickers"]["AAPL"]["status"] == "mismatch"
    assert report["mismatches"] == ["AAPL"]


def test_new_filing_flag_from_previous_accessions():
    with patch.object(ec, "EARNINGS_DATES", {"AMD": ["2026-09-01"]}), patch.object(
        ec, "fetch_latest_sec_filing", return_value=_filing("acc-NEW")
    ):
        report = ec.build_earnings_catalyst_report(
            TODAY,
            headliners=("AMD",),
            ir_sources={},
            cik_map={"AMD": "0000002488"},
            previous_accessions={"AMD": "acc-OLD"},
        )
    assert report["new_filings"] == ["AMD"]
    assert report["current_accessions"] == {"AMD": "acc-NEW"}


def test_past_hardcoded_dates_are_not_counted_upcoming():
    with patch.object(ec, "EARNINGS_DATES", {"AAPL": ["2026-01-01"]}), patch.object(
        ec, "fetch_latest_sec_filing", return_value=_filing("a")
    ):
        report = ec.build_earnings_catalyst_report(
            TODAY, headliners=("AAPL",), ir_sources={}, cik_map={"AAPL": "0000320193"}
        )
    assert report["tickers"]["AAPL"]["hardcoded_upcoming"] == []
    assert report["tickers"]["AAPL"]["status"] == "needs_attention"


def test_degrades_when_no_cik_and_no_ir():
    # No network reachable: no cik, no ir source -> hardcoded-only, no crash.
    with patch.object(ec, "EARNINGS_DATES", {"NFLX": ["2026-10-15"]}):
        report = ec.build_earnings_catalyst_report(
            TODAY, headliners=("NFLX",), ir_sources={}, cik_map={}
        )
    assert report["tickers"]["NFLX"]["status"] == "ok"
    assert report["tickers"]["NFLX"]["latest_filing"] is None
    assert report["new_filings"] == []


def test_summarize_surfaces_actionable_lines():
    report = {
        "generated": "2026-08-21",
        "suggestions": {"MSFT": "2026-10-28"},
        "mismatches": ["AAPL"],
        "needs_attention": ["TSLA"],
        "new_filings": ["AMD"],
    }
    lines = ec.summarize(report)
    assert any("SUGGEST MSFT" in ln for ln in lines)
    assert any("MISMATCH" in ln for ln in lines)
    assert any("NEEDS DATE" in ln for ln in lines)
    assert any("NEW SEC FILING" in ln for ln in lines)
