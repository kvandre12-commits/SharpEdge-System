#!/usr/bin/env python3
"""Incrementally ingest FINRA ATS weekly data and rebuild darkpool overlays.

The expensive part is FINRA network access. The cheap part is rebuilding daily
overlays from the persisted weekly table. Keep those separate so routine pipeline
runs do not re-fetch years of weekly history for no reason.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from statistics import stdev
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

FINRA_CLIENT_ID = os.getenv("FINRA_CLIENT_ID", "")
FINRA_CLIENT_SECRET = os.getenv("FINRA_CLIENT_SECRET", "")

BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
DATASET_PRIMARY = "weeklySummaryHistoric"
DATASET_FALLBACK = "weeklySummary"

TIER = os.getenv("FINRA_TIER", "T1")
SUMMARY_TYPE = os.getenv("FINRA_SUMMARY_TYPE", "ATS_W_SMBL")

START = os.getenv("FINRA_START", "2024-01-01")
LIMIT = int(os.getenv("FINRA_LIMIT", "5000"))
SLEEP_S = float(os.getenv("FINRA_SLEEP_S", "0.25"))
CACHE_TTL_HOURS = float(os.getenv("FINRA_CACHE_TTL_HOURS", "144"))
REFRESH_LOOKBACK_WEEKS = int(os.getenv("FINRA_REFRESH_LOOKBACK_WEEKS", "4"))
FORCE_REFRESH = os.getenv("FINRA_FORCE_REFRESH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

OUTPUT_WEEKLY = Path("outputs/spy_finra_ats_weekly.csv")
OUTPUT_STATE = Path("outputs/health/finra_state.json")
WEEKLY_EXPORT_COLUMNS = [
    "weekStartDate",
    "week_start",
    "symbol",
    "ats_weekly_shares",
    "ats_weekly_trades",
    "ats_weekly_notional",
    "ats_venue_count",
    "top_mpid",
    "top_market_participant_name",
    "top_mpid_share",
    "venue_hhi",
    "last_reported_date",
    "last_update_date",
    "initial_published_date",
    "avg_trade_size",
    "shares_vs_13w_avg",
    "trades_vs_13w_avg",
    "shares_z_26w",
    "ingest_ts",
]


WeekRecord = dict[str, Any]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def monday_of_week(value: dt.date) -> dt.date:
    return value - dt.timedelta(days=value.weekday())


def daterange_mondays(start: dt.date, end: dt.date):
    cur = monday_of_week(start)
    endm = monday_of_week(end)
    while cur <= endm:
        yield cur
        cur += dt.timedelta(days=7)


def parse_date(value: Any) -> dt.date | None:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_datetime(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.UTC).replace(tzinfo=None)
    return parsed


def cache_is_fresh(latest_ingest_ts: Any, now: dt.datetime | None = None) -> bool:
    latest = parse_datetime(latest_ingest_ts)
    if latest is None:
        return False
    age_hours = ((now or utc_now()) - latest).total_seconds() / 3600.0
    return 0 <= age_hours < CACHE_TTL_HOURS


def build_historic_payload(week_start: str) -> dict[str, Any]:
    """Build a FINRA historical payload using only allowed query fields."""
    return {
        "compareFilters": [
            {
                "compareType": "equal",
                "fieldName": "weekStartDate",
                "fieldValue": week_start,
            },
            {
                "compareType": "equal",
                "fieldName": "tierIdentifier",
                "fieldValue": TIER,
            },
        ],
        "limit": LIMIT,
        "offset": 0,
    }


def build_current_payload(week_start: str) -> dict[str, Any]:
    return {
        "compareFilters": [
            {
                "compareType": "equal",
                "fieldName": "issueSymbolIdentifier",
                "fieldValue": SYMBOL,
            },
            {
                "compareType": "equal",
                "fieldName": "tierIdentifier",
                "fieldValue": TIER,
            },
            {
                "compareType": "equal",
                "fieldName": "summaryTypeCode",
                "fieldValue": SUMMARY_TYPE,
            },
            {
                "compareType": "equal",
                "fieldName": "weekStartDate",
                "fieldValue": week_start,
            },
        ],
        "limit": LIMIT,
        "offset": 0,
    }


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


def post_rows(
    dataset: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    auth: HTTPBasicAuth | None,
) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{dataset}"
    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
        auth=auth,
    )

    if resp.status_code in (401, 403):
        raise PermissionError(
            "FINRA auth failed (401/403). Check FINRA_CLIENT_ID/FINRA_CLIENT_SECRET."
        )
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:200].replace("\n", " ")
        raise RuntimeError(f"FINRA HTTP {resp.status_code}: {snippet}")

    return safe_rows(resp)


def filter_symbol_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = SYMBOL.upper()
    return [
        row
        for row in rows
        if str(row.get("issueSymbolIdentifier") or "").upper() == wanted
    ]


def fetch_week(
    week_start: dt.date,
    headers: dict[str, str],
    auth: HTTPBasicAuth | None,
) -> list[dict[str, Any]]:
    week_text = week_start.isoformat()
    historic_rows: list[dict[str, Any]] = []
    try:
        historic_rows = post_rows(
            DATASET_PRIMARY,
            build_historic_payload(week_text),
            headers,
            auth,
        )
    except Exception:
        historic_rows = []

    filtered_historic = filter_symbol_rows(historic_rows)
    if filtered_historic:
        return filtered_historic

    return post_rows(
        DATASET_FALLBACK,
        build_current_payload(week_text),
        headers,
        auth,
    )


def ensure_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ats_weekly (
          week_start TEXT NOT NULL,
          symbol TEXT NOT NULL,
          ats_weekly_shares REAL,
          ats_weekly_trades REAL,
          ats_weekly_notional REAL,
          ats_venue_count INTEGER,
          top_mpid TEXT,
          top_market_participant_name TEXT,
          top_mpid_share REAL,
          venue_hhi REAL,
          last_reported_date TEXT,
          last_update_date TEXT,
          initial_published_date TEXT,
          avg_trade_size REAL,
          shares_vs_13w_avg REAL,
          trades_vs_13w_avg REAL,
          shares_z_26w REAL,
          ingest_ts TEXT,
          PRIMARY KEY(symbol, week_start)
        )
        """
    )
    existing = {row[1] for row in con.execute("PRAGMA table_info(ats_weekly)")}
    for col, decl in (
        ("ats_weekly_notional", "REAL"),
        ("top_mpid", "TEXT"),
        ("top_market_participant_name", "TEXT"),
        ("top_mpid_share", "REAL"),
        ("venue_hhi", "REAL"),
        ("last_reported_date", "TEXT"),
        ("last_update_date", "TEXT"),
        ("initial_published_date", "TEXT"),
    ):
        if col not in existing:
            con.execute(f"ALTER TABLE ats_weekly ADD COLUMN {col} {decl}")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS overlays_daily (
          date TEXT NOT NULL,
          symbol TEXT NOT NULL,
          overlay_type TEXT NOT NULL,
          overlay_strength REAL NOT NULL,
          notes TEXT,
          PRIMARY KEY (symbol, date, overlay_type)
        )
        """
    )
    con.commit()


def latest_finra_state(con: sqlite3.Connection) -> dict[str, Any]:
    row = con.execute(
        """
        SELECT MAX(week_start) AS latest_week_start,
               MAX(ingest_ts) AS latest_ingest_ts,
               COUNT(*) AS rows
        FROM ats_weekly
        WHERE symbol = ?
        """,
        (SYMBOL,),
    ).fetchone()
    if not row:
        return {"latest_week_start": None, "latest_ingest_ts": None, "rows": 0}
    return {
        "latest_week_start": row[0],
        "latest_ingest_ts": row[1],
        "rows": row[2] or 0,
    }


def weeks_to_fetch(
    state: dict[str, Any],
    today: dt.date | None = None,
    force: bool = FORCE_REFRESH,
) -> list[dt.date]:
    start = dt.date.fromisoformat(START)
    end = monday_of_week(today or dt.date.today())
    latest_week = parse_date(state.get("latest_week_start"))
    freshness_now = (
        dt.datetime.combine(today, dt.time(12, 0, 0))
        if today is not None
        else utc_now()
    )

    if (
        latest_week
        and not force
        and cache_is_fresh(state.get("latest_ingest_ts"), now=freshness_now)
    ):
        return []

    if latest_week and not force:
        start = max(start, latest_week - dt.timedelta(days=7 * REFRESH_LOOKBACK_WEEKS))

    return list(daterange_mondays(start, end))


def _first_present(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in row:
            return row[key]
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _later_date(current: str | None, candidate: str | None) -> str | None:
    if not candidate:
        return current
    if not current or candidate > current:
        return candidate
    return current


def _normalize_weekly_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        week_date = parse_date(_first_present(row, ["weekStartDate", "weekstartdate"]))
        symbol = _first_present(row, ["issueSymbolIdentifier", "issuesymbolidentifier"])
        mpid = _first_present(
            row,
            [
                "marketParticipantIdentifier",
                "marketparticipantidentifier",
                "MPID",
                "mpid",
            ],
        )
        market_participant_name = _first_present(
            row,
            ["marketParticipantName", "marketparticipantname"],
        )
        shares = _to_float(
            _first_present(
                row,
                [
                    "totalWeeklyShareQuantity",
                    "totalweeklysharequantity",
                    "totalsharequantitysum",
                ],
            )
        )
        trades = _to_float(
            _first_present(
                row,
                [
                    "totalWeeklyTradeCount",
                    "totalweeklytradecount",
                    "totaltradecountsum",
                ],
            )
        )
        notional = _to_float(
            _first_present(
                row,
                ["totalNotionalSum", "totalnotionalsum"],
            )
        )
        last_reported_date = _first_present(
            row,
            ["lastReportedDate", "lastreporteddate"],
        )
        last_update_date = _first_present(
            row,
            ["lastUpdateDate", "lastupdatedate"],
        )
        initial_published_date = _first_present(
            row,
            ["initialPublishedDate", "initialpublisheddate"],
        )
        if week_date is None or not symbol:
            continue
        normalized.append(
            {
                "week_start": week_date.isoformat(),
                "symbol": str(symbol).upper(),
                "mpid": "" if mpid in (None, "") else str(mpid),
                "market_participant_name": (
                    ""
                    if market_participant_name in (None, "")
                    else str(market_participant_name)
                ),
                "shares": shares or 0.0,
                "trades": trades or 0.0,
                "notional": notional or 0.0,
                "last_reported_date": str(last_reported_date or "") or None,
                "last_update_date": str(last_update_date or "") or None,
                "initial_published_date": str(initial_published_date or "") or None,
            }
        )
    return normalized


def aggregate_raw_weekly(raw_rows: list[dict[str, Any]]) -> list[WeekRecord]:
    grouped: dict[tuple[str, str], WeekRecord] = {}
    for row in _normalize_weekly_rows(raw_rows):
        key = (row["symbol"], row["week_start"])
        bucket = grouped.setdefault(
            key,
            {
                "week_start": row["week_start"],
                "symbol": row["symbol"],
                "ats_weekly_shares": 0.0,
                "ats_weekly_trades": 0.0,
                "ats_weekly_notional": 0.0,
                "_mpids": set(),
                "_shares_by_mpid": {},
                "_names_by_mpid": {},
                "last_reported_date": None,
                "last_update_date": None,
                "initial_published_date": None,
            },
        )
        bucket["ats_weekly_shares"] += row["shares"]
        bucket["ats_weekly_trades"] += row["trades"]
        bucket["ats_weekly_notional"] += row["notional"]
        bucket["last_reported_date"] = _later_date(
            bucket.get("last_reported_date"), row.get("last_reported_date")
        )
        bucket["last_update_date"] = _later_date(
            bucket.get("last_update_date"), row.get("last_update_date")
        )
        bucket["initial_published_date"] = _later_date(
            bucket.get("initial_published_date"), row.get("initial_published_date")
        )
        if row["mpid"]:
            bucket["_mpids"].add(row["mpid"])
            bucket["_shares_by_mpid"][row["mpid"]] = (
                float(bucket["_shares_by_mpid"].get(row["mpid"], 0.0)) + row["shares"]
            )
            if row.get("market_participant_name"):
                bucket["_names_by_mpid"][row["mpid"]] = row["market_participant_name"]

    weekly: list[WeekRecord] = []
    for bucket in grouped.values():
        total_shares = float(bucket["ats_weekly_shares"] or 0.0)
        shares_by_mpid = bucket.get("_shares_by_mpid") or {}
        top_mpid = None
        top_mpid_shares = 0.0
        if shares_by_mpid:
            top_mpid = max(shares_by_mpid, key=shares_by_mpid.get)
            top_mpid_shares = float(shares_by_mpid[top_mpid])
        top_mpid_share = (
            top_mpid_shares / total_shares if total_shares > 0 and top_mpid else None
        )
        venue_hhi = None
        if total_shares > 0 and shares_by_mpid:
            venue_hhi = sum(
                (float(shares) / total_shares) ** 2
                for shares in shares_by_mpid.values()
            )
        weekly.append(
            {
                "week_start": bucket["week_start"],
                "symbol": bucket["symbol"],
                "ats_weekly_shares": total_shares,
                "ats_weekly_trades": bucket["ats_weekly_trades"],
                "ats_weekly_notional": bucket["ats_weekly_notional"],
                "ats_venue_count": len(bucket["_mpids"]),
                "top_mpid": top_mpid,
                "top_market_participant_name": (
                    (bucket.get("_names_by_mpid") or {}).get(top_mpid)
                    if top_mpid
                    else None
                ),
                "top_mpid_share": top_mpid_share,
                "venue_hhi": venue_hhi,
                "last_reported_date": bucket.get("last_reported_date"),
                "last_update_date": bucket.get("last_update_date"),
                "initial_published_date": bucket.get("initial_published_date"),
            }
        )
    weekly.sort(key=lambda row: (row["week_start"], row["symbol"]))
    return weekly


def load_existing_weekly(con: sqlite3.Connection) -> list[WeekRecord]:
    rows = con.execute(
        """
        SELECT week_start, symbol, ats_weekly_shares, ats_weekly_trades,
               ats_weekly_notional, ats_venue_count, top_mpid,
               top_market_participant_name, top_mpid_share, venue_hhi,
               last_reported_date, last_update_date, initial_published_date
        FROM ats_weekly
        WHERE symbol = ?
        ORDER BY week_start ASC
        """,
        (SYMBOL,),
    ).fetchall()
    return [
        {
            "week_start": row[0],
            "symbol": row[1],
            "ats_weekly_shares": float(row[2] or 0.0),
            "ats_weekly_trades": float(row[3] or 0.0),
            "ats_weekly_notional": float(row[4] or 0.0),
            "ats_venue_count": int(row[5] or 0),
            "top_mpid": row[6],
            "top_market_participant_name": row[7],
            "top_mpid_share": float(row[8]) if row[8] is not None else None,
            "venue_hhi": float(row[9]) if row[9] is not None else None,
            "last_reported_date": row[10],
            "last_update_date": row[11],
            "initial_published_date": row[12],
        }
        for row in rows
    ]


def _rolling_mean(
    values: list[float], end_index: int, window: int, min_periods: int
) -> float | None:
    start = max(0, end_index - window + 1)
    sample = values[start : end_index + 1]
    if len(sample) < min_periods:
        return None
    return sum(sample) / len(sample)


def _rolling_std(
    values: list[float], end_index: int, window: int, min_periods: int
) -> float | None:
    start = max(0, end_index - window + 1)
    sample = values[start : end_index + 1]
    if len(sample) < min_periods:
        return None
    if len(sample) < 2:
        return None
    return stdev(sample)


def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return numerator / denominator


def recompute_metrics(weekly: list[WeekRecord], ingest_ts: str) -> list[WeekRecord]:
    if not weekly:
        return []

    rows = sorted(weekly, key=lambda row: row["week_start"])
    shares_values = [float(row.get("ats_weekly_shares") or 0.0) for row in rows]
    trades_values = [float(row.get("ats_weekly_trades") or 0.0) for row in rows]

    for index, row in enumerate(rows):
        shares = shares_values[index]
        trades = trades_values[index]
        shares_avg_13 = _rolling_mean(shares_values, index, window=13, min_periods=4)
        trades_avg_13 = _rolling_mean(trades_values, index, window=13, min_periods=4)
        shares_mean_26 = _rolling_mean(shares_values, index, window=26, min_periods=8)
        shares_std_26 = _rolling_std(shares_values, index, window=26, min_periods=8)

        row["ats_weekly_shares"] = shares
        row["ats_weekly_trades"] = trades
        row["ats_weekly_notional"] = float(row.get("ats_weekly_notional") or 0.0)
        row["avg_trade_size"] = _safe_ratio(shares, trades)
        row["shares_vs_13w_avg"] = _safe_ratio(shares, shares_avg_13)
        row["trades_vs_13w_avg"] = _safe_ratio(trades, trades_avg_13)
        if shares_mean_26 is None or shares_std_26 in (None, 0):
            row["shares_z_26w"] = None
        else:
            row["shares_z_26w"] = (shares - shares_mean_26) / shares_std_26
        row["ingest_ts"] = ingest_ts
    return rows


def merge_weekly(
    existing: list[WeekRecord],
    fetched: list[WeekRecord],
    ingest_ts: str,
) -> list[WeekRecord]:
    combined: dict[tuple[str, str], WeekRecord] = {}
    for row in existing + fetched:
        symbol = str(row.get("symbol") or SYMBOL).upper()
        if symbol != SYMBOL.upper():
            continue
        week_start = str(row["week_start"])
        combined[(symbol, week_start)] = {
            "week_start": week_start,
            "symbol": symbol,
            "ats_weekly_shares": float(row.get("ats_weekly_shares") or 0.0),
            "ats_weekly_trades": float(row.get("ats_weekly_trades") or 0.0),
            "ats_weekly_notional": float(row.get("ats_weekly_notional") or 0.0),
            "ats_venue_count": int(row.get("ats_venue_count") or 0),
            "top_mpid": row.get("top_mpid"),
            "top_market_participant_name": row.get("top_market_participant_name"),
            "top_mpid_share": row.get("top_mpid_share"),
            "venue_hhi": row.get("venue_hhi"),
            "last_reported_date": row.get("last_reported_date"),
            "last_update_date": row.get("last_update_date"),
            "initial_published_date": row.get("initial_published_date"),
        }
    return recompute_metrics(list(combined.values()), ingest_ts)


def fetch_finra_weeks(weeks: list[dt.date]) -> tuple[list[WeekRecord], int]:
    if not weeks:
        return [], 0

    auth = None
    if FINRA_CLIENT_ID and FINRA_CLIENT_SECRET:
        auth = HTTPBasicAuth(FINRA_CLIENT_ID, FINRA_CLIENT_SECRET)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "spy-finra-darkpool/2.1",
    }
    raw_rows: list[dict[str, Any]] = []
    failures = 0
    for week in weeks:
        try:
            raw_rows.extend(fetch_week(week, headers, auth))
        except Exception:
            failures += 1
        time.sleep(SLEEP_S)
    return aggregate_raw_weekly(raw_rows), failures


def upsert_weekly(con: sqlite3.Connection, weekly: list[WeekRecord]) -> None:
    cols = [
        "week_start",
        "symbol",
        "ats_weekly_shares",
        "ats_weekly_trades",
        "ats_weekly_notional",
        "ats_venue_count",
        "top_mpid",
        "top_market_participant_name",
        "top_mpid_share",
        "venue_hhi",
        "last_reported_date",
        "last_update_date",
        "initial_published_date",
        "avg_trade_size",
        "shares_vs_13w_avg",
        "trades_vs_13w_avg",
        "shares_z_26w",
        "ingest_ts",
    ]
    upsert = f"""
    INSERT INTO ats_weekly ({",".join(cols)})
    VALUES ({",".join(["?"] * len(cols))})
    ON CONFLICT(symbol, week_start) DO UPDATE SET
      ats_weekly_shares=excluded.ats_weekly_shares,
      ats_weekly_trades=excluded.ats_weekly_trades,
      ats_weekly_notional=excluded.ats_weekly_notional,
      ats_venue_count=excluded.ats_venue_count,
      top_mpid=excluded.top_mpid,
      top_market_participant_name=excluded.top_market_participant_name,
      top_mpid_share=excluded.top_mpid_share,
      venue_hhi=excluded.venue_hhi,
      last_reported_date=excluded.last_reported_date,
      last_update_date=excluded.last_update_date,
      initial_published_date=excluded.initial_published_date,
      avg_trade_size=excluded.avg_trade_size,
      shares_vs_13w_avg=excluded.shares_vs_13w_avg,
      trades_vs_13w_avg=excluded.trades_vs_13w_avg,
      shares_z_26w=excluded.shares_z_26w,
      ingest_ts=excluded.ingest_ts
    """
    rows = [tuple(row.get(col) for col in cols) for row in weekly]
    con.executemany(upsert, rows)
    con.commit()


def z_to_strength(z_value: float | None) -> float:
    if z_value is None or math.isnan(z_value):
        return 0.0
    return float(max(0.0, min(1.0, (z_value - 1.0) / 1.5)))


def rebuild_daily_overlays(con: sqlite3.Connection, weekly: list[WeekRecord]) -> int:
    if not weekly:
        return 0

    days = con.execute(
        "SELECT date FROM bars_daily WHERE symbol=? ORDER BY date ASC",
        (SYMBOL,),
    ).fetchall()
    if not days:
        return 0

    weekly_by_week = {
        str(row["week_start"]): row for row in weekly if row.get("week_start")
    }
    rows = []
    for (day_text,) in days:
        day = parse_date(day_text)
        if day is None:
            continue
        week_start = monday_of_week(day).isoformat()
        weekly_row = weekly_by_week.get(week_start) or {}
        z_value = weekly_row.get("shares_z_26w")
        note = (
            f"finra_ats_shares_z_26w={z_value:.2f}"
            if z_value is not None and not math.isnan(z_value)
            else "finra_ats_missing"
        )
        top_mpid = weekly_row.get("top_mpid")
        top_share = weekly_row.get("top_mpid_share")
        venue_count = weekly_row.get("ats_venue_count")
        if top_mpid:
            top_share_text = (
                f"{top_share:.2f}" if isinstance(top_share, (int, float)) else "na"
            )
            note += f" | top_mpid={top_mpid} share={top_share_text}"
        if isinstance(venue_count, int) and venue_count > 0:
            note += f" | venues={venue_count}"
        rows.append((day.isoformat(), SYMBOL, z_to_strength(z_value), note))

    write_overlay = """
    INSERT INTO overlays_daily (date, symbol, overlay_type, overlay_strength, notes)
    VALUES (?, ?, 'darkpool', ?, ?)
    ON CONFLICT(symbol, date, overlay_type) DO UPDATE SET
      overlay_strength=excluded.overlay_strength,
      notes=excluded.notes
    """
    con.executemany(write_overlay, rows)
    con.commit()
    return len(rows)


def export_weekly_rows(weekly: list[WeekRecord]) -> list[dict[str, Any]]:
    rows = []
    for row in weekly:
        exported = {"weekStartDate": row.get("week_start")}
        for column in WEEKLY_EXPORT_COLUMNS[1:]:
            exported[column] = row.get(column)
        rows.append(exported)
    return rows


def write_outputs(weekly: list[WeekRecord], state: dict[str, Any]) -> None:
    OUTPUT_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATE.parent.mkdir(parents=True, exist_ok=True)
    exported_rows = export_weekly_rows(weekly)
    with OUTPUT_WEEKLY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WEEKLY_EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(exported_rows)
    OUTPUT_STATE.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    ingest_ts = utc_now().isoformat()
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(con)
        before_state = latest_finra_state(con)
        requested_weeks = weeks_to_fetch(before_state)
        fetched, failures = fetch_finra_weeks(requested_weeks)
        existing = load_existing_weekly(con)
        weekly = merge_weekly(
            existing,
            fetched,
            ingest_ts if fetched else before_state.get("latest_ingest_ts") or ingest_ts,
        )

        if not weekly:
            raise RuntimeError(
                "No FINRA rows available from API or persisted ats_weekly state."
            )

        if fetched:
            upsert_weekly(con, weekly)

        overlay_rows = rebuild_daily_overlays(con, weekly)
        after_state = latest_finra_state(con)
    finally:
        con.close()

    output_state = {
        "symbol": SYMBOL,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "force_refresh": FORCE_REFRESH,
        "network_refresh": bool(requested_weeks),
        "requested_weeks": [week.isoformat() for week in requested_weeks],
        "requested_week_count": len(requested_weeks),
        "fetched_week_count": len(fetched),
        "failures": failures,
        "overlay_rows": overlay_rows,
        "before": before_state,
        "after": after_state,
    }
    write_outputs(weekly, output_state)
    mode = "network_refresh" if requested_weeks else "cache_rebuild_only"
    print(
        f"OK: {OUTPUT_WEEKLY} | mode={mode} | weeks={len(weekly)} | "
        f"requested={len(requested_weeks)} | fetched={len(fetched)} | "
        f"overlay_rows={overlay_rows} | failures={failures}"
    )


if __name__ == "__main__":
    main()
