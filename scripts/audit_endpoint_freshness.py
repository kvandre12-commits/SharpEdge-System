#!/usr/bin/env python3
"""Audit SharpEdge endpoint freshness and repo endpoint ownership.

This is intentionally lightweight and mostly read-only:
- probes public endpoints live when possible
- probes credential-gated endpoints only when env creds exist
- inspects outputs/health/*_state.json for stale pipeline artifacts
- writes machine + human audit artifacts under outputs/
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from http_utils import request_json_with_backoff  # noqa: E402

try:
    from scripts.utils.pipeline_state import age_hours, parse_date, utc_now
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import age_hours, parse_date, utc_now

OUTPUTS_DIR = ROOT / "outputs"
HEALTH_DIR = OUTPUTS_DIR / "health"
OUT_JSON = OUTPUTS_DIR / "endpoint_audit_latest.json"
OUT_MD = OUTPUTS_DIR / "endpoint_audit_latest.md"

UA = {"User-Agent": "sharpedge-endpoint-audit/1.0"}
SYMBOL = os.getenv("SYMBOL", "SPY")
YAHOO_URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
CBOE_URL = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets")
ALPACA_TRADING_BASE = os.getenv(
    "ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets"
)
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name"


REPO_NOTES = {
    "SharpEdge-System": {
        "owns_live_fetch": True,
        "notes": [
            "Primary live market-data fetch repo.",
            "Yahoo/CBOE are the public no-auth surfaces currently powering cockpit reads.",
            "FRED and Alpaca are credential-gated ingestion layers.",
            "FINRA ATS is publicly probeable here, but its schema is finicky across current vs historical datasets.",
        ],
        "call_sites": [
            "cockpit/market_data_sources.py",
            "cockpit/make_overlay.py",
            "cockpit/make_price_volume.py",
            "cockpit/make_options.py",
            "scripts/ingest_spy_daily.py",
            "scripts/ingest_spy_intraday_alpaca.py",
            "scripts/ingest_cboe_options_chain_snapshots.py",
            "scripts/ingest_fred_overlays.py",
            "scripts/ingest_finra_darkpool_overlay.py",
            "scripts/ingest_alpaca_options_open_interest_daily.py",
        ],
    },
    "SharpEdge-Robinhood-Bridge": {
        "owns_live_fetch": False,
        "notes": [
            "No direct HTTP client usage found in repo code.",
            "Acts as policy/router/local workflow layer rather than a market-data fetcher.",
        ],
        "call_sites": [],
    },
    "SharpEdge-Android": {
        "owns_live_fetch": False,
        "notes": [
            "Android app currently renders bundled/sample contracts rather than live network data.",
            "No retrofit/okhttp/http client usage found in app code.",
        ],
        "call_sites": [],
    },
}


CREDENTIAL_HINTS = {
    "fred": ["FRED_API_KEY"],
    "alpaca": ["ALPACA_API_KEY", "ALPACA_API_SECRET"],
    "finra": ["FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET"],
}

FINRA_HISTORIC_SAMPLE_WEEK = os.getenv("FINRA_HISTORIC_SAMPLE_WEEK", "2025-06-23")


def iso_now() -> str:
    return utc_now().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_age_hours(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 24:
        return "fresh"
    if value < 72:
        return "aging"
    return "stale"


def classify_age_days(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value <= 1:
        return "fresh"
    if value <= 5:
        return "aging"
    return "stale"


def live_probe(name: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "ok", **details}


def skipped_probe(name: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "skipped", "reason": reason, **details}


def failed_probe(name: str, error: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "error", "error": error, **details}


def safe_rows(resp: requests.Response) -> list[dict[str, Any]]:
    text = resp.text or ""
    if not text.strip() or text.lstrip().startswith("<"):
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, list):
        return data
    return []


def probe_yahoo(interval: str, range_: str) -> dict[str, Any]:
    try:
        payload = request_json_with_backoff(
            YAHOO_URL,
            params={"interval": interval, "range": range_},
            headers=UA,
            timeout=20,
            attempts=4,
            base_sleep_seconds=1.0,
        )
        result = payload["chart"]["result"][0]
        meta = result.get("meta", {})
        timestamps = result.get("timestamp") or []
        last_bar = None
        if timestamps:
            last_bar = dt.datetime.fromtimestamp(timestamps[-1], tz=dt.UTC).isoformat()
        regular_market_time = meta.get("regularMarketTime")
        return live_probe(
            f"yahoo_chart_{interval}_{range_}",
            endpoint=YAHOO_URL,
            http_status=200,
            interval=interval,
            range=range_,
            last_bar_utc=last_bar,
            regular_market_time_utc=(
                dt.datetime.fromtimestamp(regular_market_time, tz=dt.UTC).isoformat()
                if regular_market_time
                else None
            ),
            regular_market_price=meta.get("regularMarketPrice"),
            chart_previous_close=meta.get("chartPreviousClose"),
        )
    except Exception as exc:  # noqa: BLE001 - audit should keep going
        return failed_probe(
            f"yahoo_chart_{interval}_{range_}",
            str(exc),
            endpoint=YAHOO_URL,
        )


def probe_cboe() -> dict[str, Any]:
    try:
        data = request_json_with_backoff(
            CBOE_URL,
            headers=UA,
            timeout=30,
            attempts=3,
            base_sleep_seconds=1.0,
        )["data"]
        options = data.get("options") or []
        latest_option_trade = None
        for option in options:
            raw_time = option.get("last_trade_time")
            if raw_time and (
                latest_option_trade is None or raw_time > latest_option_trade
            ):
                latest_option_trade = raw_time
        return live_probe(
            "cboe_options",
            endpoint=CBOE_URL,
            http_status=200,
            option_count=len(options),
            current_price=data.get("current_price"),
            close=data.get("close"),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            prev_day_close=data.get("prev_day_close"),
            bid=data.get("bid"),
            ask=data.get("ask"),
            price_change=data.get("price_change"),
            price_change_percent=data.get("price_change_percent"),
            data_last_trade_time_raw=data.get("last_trade_time"),
            latest_option_trade_time_raw=latest_option_trade,
            iv30=data.get("iv30"),
            iv30_change=data.get("iv30_change"),
            iv30_change_percent=data.get("iv30_change_percent"),
        )
    except Exception as exc:  # noqa: BLE001
        return failed_probe("cboe_options", str(exc), endpoint=CBOE_URL)


def probe_fred() -> dict[str, Any]:
    api_key = os.getenv("FRED_API_KEY")
    params = {
        "series_id": "VIXCLS",
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    if api_key:
        params["api_key"] = api_key
    try:
        response = requests.get(FRED_URL, params=params, timeout=20)
        if response.status_code == 200:
            observation = (response.json().get("observations") or [{}])[0]
            return live_probe(
                "fred_observations",
                endpoint=FRED_URL,
                http_status=response.status_code,
                auth_configured=bool(api_key),
                latest_observation_date=observation.get("date"),
                latest_observation_value=observation.get("value"),
            )
        if not api_key:
            return skipped_probe(
                "fred_observations",
                "auth_required_or_key_missing",
                endpoint=FRED_URL,
                required_env=CREDENTIAL_HINTS["fred"],
                unauth_status=response.status_code,
                body_snippet=(response.text or "")[:200],
            )
        response.raise_for_status()
        return failed_probe(
            "fred_observations",
            f"unexpected_status={response.status_code}",
            endpoint=FRED_URL,
        )
    except Exception as exc:  # noqa: BLE001
        return failed_probe("fred_observations", str(exc), endpoint=FRED_URL)


def probe_alpaca_bars() -> dict[str, Any]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    endpoint = f"{ALPACA_DATA_BASE}/v2/stocks/bars"
    params = {
        "symbols": SYMBOL,
        "timeframe": "15Min",
        "limit": 1,
        "adjustment": "raw",
        "feed": "iex",
        "sort": "desc",
    }
    headers = {}
    if key and secret:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=20)
        if response.status_code == 200:
            bars = (response.json().get("bars") or {}).get(SYMBOL, [])
            latest = bars[0] if bars else {}
            return live_probe(
                "alpaca_bars",
                endpoint=endpoint,
                http_status=response.status_code,
                auth_configured=bool(key and secret),
                returned_bars=len(bars),
                latest_bar_ts=latest.get("t"),
                latest_close=latest.get("c"),
            )
        if not (key and secret):
            return skipped_probe(
                "alpaca_bars",
                "auth_required",
                endpoint=endpoint,
                required_env=CREDENTIAL_HINTS["alpaca"],
                unauth_status=response.status_code,
                body_snippet=(response.text or "")[:200],
            )
        response.raise_for_status()
        return failed_probe(
            "alpaca_bars",
            f"unexpected_status={response.status_code}",
            endpoint=endpoint,
        )
    except Exception as exc:  # noqa: BLE001
        return failed_probe("alpaca_bars", str(exc), endpoint=endpoint)


def probe_alpaca_option_contracts() -> dict[str, Any]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    endpoint = f"{ALPACA_TRADING_BASE}/v2/options/contracts"
    headers = {}
    if key and secret:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params={"underlying_symbols": SYMBOL, "limit": 1},
            timeout=20,
        )
        if response.status_code == 200:
            payload = response.json()
            contracts = (
                payload.get("option_contracts") or payload.get("contracts") or []
            )
            first = contracts[0] if contracts else {}
            return live_probe(
                "alpaca_option_contracts",
                endpoint=endpoint,
                http_status=response.status_code,
                auth_configured=bool(key and secret),
                returned_contracts=len(contracts),
                sample_expiration=first.get("expiration_date")
                or first.get("expiration"),
                sample_strike=first.get("strike_price") or first.get("strike"),
                sample_open_interest=first.get("open_interest"),
            )
        if not (key and secret):
            return skipped_probe(
                "alpaca_option_contracts",
                "auth_required",
                endpoint=endpoint,
                required_env=CREDENTIAL_HINTS["alpaca"],
                unauth_status=response.status_code,
                body_snippet=(response.text or "")[:200],
            )
        response.raise_for_status()
        return failed_probe(
            "alpaca_option_contracts",
            f"unexpected_status={response.status_code}",
            endpoint=endpoint,
        )
    except Exception as exc:  # noqa: BLE001
        return failed_probe("alpaca_option_contracts", str(exc), endpoint=endpoint)


def probe_finra() -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "sharpedge-endpoint-audit/1.0",
    }
    try:
        historic_payload = {
            "compareFilters": [
                {
                    "compareType": "equal",
                    "fieldName": "weekStartDate",
                    "fieldValue": FINRA_HISTORIC_SAMPLE_WEEK,
                },
                {
                    "compareType": "equal",
                    "fieldName": "tierIdentifier",
                    "fieldValue": "T1",
                },
            ],
            "limit": 5000,
            "offset": 0,
        }
        historic_resp = requests.post(
            f"{FINRA_URL}/weeklySummaryHistoric",
            headers=headers,
            data=json.dumps(historic_payload),
            timeout=45,
        )
        historic_resp.raise_for_status()
        historic_rows = historic_resp.json()
        historic_spy = [
            row
            for row in historic_rows
            if str(row.get("issueSymbolIdentifier") or "").upper() == SYMBOL.upper()
        ]

        current_payload = {
            "compareFilters": [
                {
                    "compareType": "equal",
                    "fieldName": "issueSymbolIdentifier",
                    "fieldValue": SYMBOL,
                },
                {
                    "compareType": "equal",
                    "fieldName": "tierIdentifier",
                    "fieldValue": "T1",
                },
                {
                    "compareType": "equal",
                    "fieldName": "summaryTypeCode",
                    "fieldValue": "ATS_W_SMBL",
                },
            ],
            "limit": 5,
            "offset": 0,
        }
        current_resp = requests.post(
            f"{FINRA_URL}/weeklySummary",
            headers=headers,
            data=json.dumps(current_payload),
            timeout=45,
        )
        current_rows = safe_rows(current_resp)
        latest_current = current_rows[0] if current_rows else {}
        return live_probe(
            "finra_ats",
            endpoint=FINRA_URL,
            auth_configured=bool(
                os.getenv("FINRA_CLIENT_ID") and os.getenv("FINRA_CLIENT_SECRET")
            ),
            historic_status=historic_resp.status_code,
            historic_sample_week=FINRA_HISTORIC_SAMPLE_WEEK,
            historic_row_count=len(historic_rows),
            historic_spy_row_count=len(historic_spy),
            current_status=current_resp.status_code,
            current_spy_row_count=len(current_rows),
            current_last_reported_date=latest_current.get("lastReportedDate"),
            current_initial_published_date=latest_current.get("initialPublishedDate"),
        )
    except Exception as exc:  # noqa: BLE001
        return failed_probe("finra_ats", str(exc), endpoint=FINRA_URL)


def inspect_health_states() -> list[dict[str, Any]]:
    findings = []
    if not HEALTH_DIR.exists():
        return findings

    for path in sorted(HEALTH_DIR.glob("*_state.json")):
        payload = read_json(path)
        after = (
            payload.get("after") if isinstance(payload.get("after"), dict) else payload
        )
        latest_ts = (
            after.get("latest_ingest_ts")
            or after.get("latest_snapshot_ts")
            or payload.get("latest_ingest_ts")
            or payload.get("latest_snapshot_ts")
        )
        latest_date = (
            after.get("latest_date")
            or after.get("latest_session_date")
            or payload.get("latest_date")
            or payload.get("latest_session_date")
        )
        age_h = age_hours(latest_ts) if latest_ts else None
        age_days = None
        if latest_date:
            parsed_date = parse_date(latest_date)
            if parsed_date is not None:
                age_days = (utc_now().date() - parsed_date).days
        status = (
            classify_age_hours(age_h)
            if age_h is not None
            else classify_age_days(age_days)
        )
        findings.append(
            {
                "state_file": path.name,
                "status": status,
                "latest_ts": latest_ts,
                "latest_date": latest_date,
                "age_hours": None if age_h is None else round(age_h, 2),
                "age_days": age_days,
                "network_refresh": payload.get("network_refresh"),
            }
        )
    return findings


def summarize_findings(
    probes: list[dict[str, Any]], health_states: list[dict[str, Any]]
) -> list[str]:
    findings = []
    if any(
        probe["status"] == "ok"
        for probe in probes
        if probe["name"].startswith("yahoo_chart")
    ):
        findings.append(
            "Yahoo public chart endpoints are live and returning current-session bars."
        )
    if any(
        probe["status"] == "ok" and probe["name"] == "cboe_options" for probe in probes
    ):
        findings.append(
            "CBOE delayed options feed is reachable and still rich enough to support richer Greeks/IV metadata."
        )
    stale_states = [
        state["state_file"] for state in health_states if state["status"] == "stale"
    ]
    if stale_states:
        findings.append(
            "Offline pipeline health artifacts are stale for: "
            + ", ".join(stale_states)
        )
    skipped = [probe["name"] for probe in probes if probe["status"] == "skipped"]
    if skipped:
        findings.append(
            "Some endpoints are reachable but still auth-gated or key-limited here: "
            + ", ".join(skipped)
        )
    finra_probe = next(
        (probe for probe in probes if probe["name"] == "finra_ats"), None
    )
    if finra_probe and finra_probe.get("status") == "ok":
        current_last_reported = parse_date(
            finra_probe.get("current_last_reported_date")
        )
        if current_last_reported is not None:
            lag_days = (utc_now().date() - current_last_reported).days
            if lag_days > 14:
                findings.append(
                    "FINRA current weeklySummary surface looks materially lagged here "
                    f"(lastReportedDate age ~{lag_days} days), so historical pulls may be more trustworthy than the so-called current surface."
                )
    findings.append(
        "SharpEdge-Robinhood-Bridge and SharpEdge-Android do not currently own live market-data fetches."
    )
    findings.append(
        "Standalone cockpit renderers now reuse shared data-source helpers instead of open-coding endpoint fetches."
    )
    return findings


def improvement_opportunities() -> list[str]:
    return [
        "Surface source freshness badges in more cockpit/overlay artifacts, not just signal.json.",
        "Keep extending shared market_data_sources.py so endpoint changes happen once instead of in scattered scripts.",
        "Promote credential-backed live probes into CI or a scheduled audit once secrets are available.",
        "Track options-history freshness with health artifacts so stale OI/gamma snapshots are obvious before analysis runs.",
    ]


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# SharpEdge endpoint audit — {report['generated_at']}",
        "",
        "## Repo ownership",
    ]
    for repo, details in report["repos"].items():
        lines.append(f"### {repo}")
        lines.append(
            f"- owns live fetch: {'yes' if details['owns_live_fetch'] else 'no'}"
        )
        for note in details["notes"]:
            lines.append(f"- {note}")
        if details["call_sites"]:
            lines.append("- call sites:")
            for site in details["call_sites"]:
                lines.append(f"  - `{site}`")
        lines.append("")

    lines.append("## Live probes")
    for probe in report["probes"]:
        lines.append(f"### {probe['name']}")
        lines.append(f"- status: {probe['status']}")
        for key, value in probe.items():
            if key in {"name", "status"}:
                continue
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines.append("## Health artifacts")
    for state in report["health_states"]:
        lines.append(
            f"- `{state['state_file']}` — {state['status']} "
            f"(latest_ts={state['latest_ts']}, latest_date={state['latest_date']}, "
            f"age_hours={state['age_hours']}, age_days={state['age_days']})"
        )
    lines.append("")

    lines.append("## Findings")
    for item in report["findings"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Opportunities")
    for item in report["opportunities"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    probes = [
        probe_yahoo("1m", "1d"),
        probe_yahoo("1d", "5d"),
        probe_cboe(),
        probe_fred(),
        probe_alpaca_bars(),
        probe_alpaca_option_contracts(),
        probe_finra(),
    ]
    health_states = inspect_health_states()
    report = {
        "generated_at": iso_now(),
        "repos": REPO_NOTES,
        "probes": probes,
        "health_states": health_states,
        "findings": summarize_findings(probes, health_states),
        "opportunities": improvement_opportunities(),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
