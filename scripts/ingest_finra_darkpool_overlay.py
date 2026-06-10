#!/usr/bin/env python3
"""Incrementally ingest FINRA ATS weekly data and rebuild darkpool overlays.

The expensive part is FINRA network access. The cheap part is rebuilding daily
overlays from the persisted weekly table. Keep those separate so routine pipeline
runs do not re-fetch years of weekly history for no reason.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
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
FORCE_REFRESH = os.getenv("FINRA_FORCE_REFRESH", "0").strip().lower() in {"1", "true", "yes", "y"}

OUTPUT_WEEKLY = Path("outputs/spy_finra_ats_weekly.csv")
OUTPUT_STATE = Path("outputs/health/finra_state.json")


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


def build_payload(week_start: str) -> dict[str, Any]:
    return {
        "compareFilters": [
            {"compareType": "equal", "fieldName": "issueSymbolIdentifier", "fieldValue": SYMBOL},
            {"compareType": "equal", "fieldName": "tierIdentifier", "fieldValue": TIER},
            {"compareType": "equal", "fieldName": "summaryTypeCode", "fieldValue": SUMMARY_TYPE},
            {"compareType": "equal", "fieldName": "weekStartDate", "fieldValue": week_start},
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


def post_rows(dataset: str, payload: dict[str, Any], headers: dict[str, str], auth: HTTPBasicAuth) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{dataset}"
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60, auth=auth)

    if resp.status_code in (401, 403):
        raise PermissionError("FINRA auth failed (401/403). Check FINRA_CLIENT_ID/FINRA_CLIENT_SECRET.")
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:200].replace("\n", " ")
        raise RuntimeError(f"FINRA HTTP {resp.status_code}: {snippet}")

    return safe_rows(resp)


def fetch_week(week_start: dt.date, headers: dict[str, str], auth: HTTPBasicAuth) -> pd.DataFrame:
    payload = build_payload(week_start.isoformat())
    try:
        rows = post_rows(DATASET_PRIMARY, payload, headers, auth)
    except Exception:
        rows = post_rows(DATASET_FALLBACK, payload, headers, auth)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def ensure_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ats_weekly (
          week_start TEXT NOT NULL,
          symbol TEXT NOT NULL,
          ats_weekly_shares REAL,
          ats_weekly_trades REAL,
          ats_venue_count INTEGER,
          avg_trade_size REAL,
          shares_vs_13w_avg REAL,
          trades_vs_13w_avg REAL,
          shares_z_26w REAL,
          ingest_ts TEXT,
          PRIMARY KEY(symbol, week_start)
        )
        """
    )
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
    return {"latest_week_start": row[0], "latest_ingest_ts": row[1], "rows": row[2] or 0}


def weeks_to_fetch(state: dict[str, Any], today: dt.date | None = None, force: bool = FORCE_REFRESH) -> list[dt.date]:
    start = dt.date.fromisoformat(START)
    end = monday_of_week(today or dt.date.today())
    latest_week = parse_date(state.get("latest_week_start"))

    if latest_week and not force and cache_is_fresh(state.get("latest_ingest_ts")):
        return []

    if latest_week and not force:
        start = max(start, latest_week - dt.timedelta(days=7 * REFRESH_LOOKBACK_WEEKS))

    return list(daterange_mondays(start, end))


def pick_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"None of these columns found: {candidates}")


def aggregate_raw_weekly(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    col_week = pick_column(raw, ["weekStartDate", "weekstartdate"])
    col_symbol = pick_column(raw, ["issueSymbolIdentifier", "issuesymbolidentifier"])
    col_mpid = pick_column(raw, ["marketParticipantIdentifier", "marketparticipantidentifier", "MPID", "mpid"])
    col_shares = pick_column(raw, ["totalWeeklyShareQuantity", "totalweeklysharequantity", "totalsharequantitysum"])
    col_trades = pick_column(raw, ["totalWeeklyTradeCount", "totalweeklytradecount", "totaltradecountsum"])

    raw = raw.copy()
    raw[col_week] = pd.to_datetime(raw[col_week], errors="coerce")
    raw[col_shares] = pd.to_numeric(raw[col_shares], errors="coerce")
    raw[col_trades] = pd.to_numeric(raw[col_trades], errors="coerce")

    weekly = (
        raw.groupby(raw[col_week], as_index=False)
        .agg(
            week_start=(col_week, "first"),
            symbol=(col_symbol, "first"),
            ats_weekly_shares=(col_shares, "sum"),
            ats_weekly_trades=(col_trades, "sum"),
            ats_venue_count=(col_mpid, pd.Series.nunique),
        )
        .dropna(subset=["week_start"])
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"]).dt.strftime("%Y-%m-%d")
    return weekly


def load_existing_weekly(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT week_start, symbol, ats_weekly_shares, ats_weekly_trades, ats_venue_count
        FROM ats_weekly
        WHERE symbol = ?
        ORDER BY week_start ASC
        """,
        con,
        params=(SYMBOL,),
    )


def recompute_metrics(weekly: pd.DataFrame, ingest_ts: str) -> pd.DataFrame:
    if weekly.empty:
        return weekly
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
    weekly["ats_weekly_shares"] = pd.to_numeric(weekly["ats_weekly_shares"], errors="coerce")
    weekly["ats_weekly_trades"] = pd.to_numeric(weekly["ats_weekly_trades"], errors="coerce")
    weekly = weekly.dropna(subset=["week_start"]).sort_values("week_start").reset_index(drop=True)

    weekly["avg_trade_size"] = weekly["ats_weekly_shares"] / weekly["ats_weekly_trades"].replace(0, pd.NA)
    weekly["shares_vs_13w_avg"] = weekly["ats_weekly_shares"] / weekly["ats_weekly_shares"].rolling(13, min_periods=4).mean()
    weekly["trades_vs_13w_avg"] = weekly["ats_weekly_trades"] / weekly["ats_weekly_trades"].rolling(13, min_periods=4).mean()
    rolling_mean = weekly["ats_weekly_shares"].rolling(26, min_periods=8).mean()
    rolling_std = weekly["ats_weekly_shares"].rolling(26, min_periods=8).std()
    weekly["shares_z_26w"] = (weekly["ats_weekly_shares"] - rolling_mean) / rolling_std
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["ingest_ts"] = ingest_ts
    return weekly


def merge_weekly(existing: pd.DataFrame, fetched: pd.DataFrame, ingest_ts: str) -> pd.DataFrame:
    frames = [frame for frame in [existing, fetched] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["symbol"] = combined["symbol"].fillna(SYMBOL).astype(str).str.upper()
    combined = combined[combined["symbol"] == SYMBOL.upper()]
    combined = combined.drop_duplicates(subset=["symbol", "week_start"], keep="last")
    return recompute_metrics(combined, ingest_ts)


def fetch_finra_weeks(weeks: list[dt.date]) -> tuple[pd.DataFrame, int]:
    if not weeks:
        return pd.DataFrame(), 0
    if not FINRA_CLIENT_ID or not FINRA_CLIENT_SECRET:
        raise RuntimeError("Missing FINRA_CLIENT_ID or FINRA_CLIENT_SECRET env vars for FINRA refresh.")

    auth = HTTPBasicAuth(FINRA_CLIENT_ID, FINRA_CLIENT_SECRET)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "spy-finra-darkpool/2.1",
    }
    frames: list[pd.DataFrame] = []
    failures = 0
    for week in weeks:
        try:
            frame = fetch_week(week, headers, auth)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            failures += 1
        time.sleep(SLEEP_S)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return aggregate_raw_weekly(raw), failures


def upsert_weekly(con: sqlite3.Connection, weekly: pd.DataFrame) -> None:
    cols = [
        "week_start",
        "symbol",
        "ats_weekly_shares",
        "ats_weekly_trades",
        "ats_venue_count",
        "avg_trade_size",
        "shares_vs_13w_avg",
        "trades_vs_13w_avg",
        "shares_z_26w",
        "ingest_ts",
    ]
    upsert = f"""
    INSERT INTO ats_weekly ({','.join(cols)})
    VALUES ({','.join(['?'] * len(cols))})
    ON CONFLICT(symbol, week_start) DO UPDATE SET
      ats_weekly_shares=excluded.ats_weekly_shares,
      ats_weekly_trades=excluded.ats_weekly_trades,
      ats_venue_count=excluded.ats_venue_count,
      avg_trade_size=excluded.avg_trade_size,
      shares_vs_13w_avg=excluded.shares_vs_13w_avg,
      trades_vs_13w_avg=excluded.trades_vs_13w_avg,
      shares_z_26w=excluded.shares_z_26w,
      ingest_ts=excluded.ingest_ts
    """
    con.executemany(upsert, weekly[cols].values.tolist())
    con.commit()


def z_to_strength(z_value: float) -> float:
    if z_value is None or pd.isna(z_value):
        return 0.0
    return float(max(0.0, min(1.0, (z_value - 1.0) / 1.5)))


def rebuild_daily_overlays(con: sqlite3.Connection, weekly: pd.DataFrame) -> int:
    days = pd.read_sql_query(
        "SELECT date FROM bars_daily WHERE symbol=? ORDER BY date ASC",
        con,
        params=(SYMBOL,),
    )
    if days.empty or weekly.empty:
        return 0

    days["date"] = pd.to_datetime(days["date"])
    days["week_start"] = days["date"].map(lambda value: monday_of_week(value.date()))
    days["week_start"] = pd.to_datetime(days["week_start"])

    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    joined = days.merge(weekly[["week_start", "shares_z_26w"]], on="week_start", how="left")
    joined["overlay_strength"] = joined["shares_z_26w"].apply(z_to_strength)
    joined["notes"] = joined["shares_z_26w"].apply(
        lambda z_value: f"finra_ats_shares_z_26w={z_value:.2f}" if pd.notna(z_value) else "finra_ats_missing"
    )

    write_overlay = """
    INSERT INTO overlays_daily (date, symbol, overlay_type, overlay_strength, notes)
    VALUES (?, ?, 'darkpool', ?, ?)
    ON CONFLICT(symbol, date, overlay_type) DO UPDATE SET
      overlay_strength=excluded.overlay_strength,
      notes=excluded.notes
    """
    rows = [
        (day.strftime("%Y-%m-%d"), SYMBOL, float(strength), str(note))
        for day, strength, note in zip(joined["date"], joined["overlay_strength"], joined["notes"])
    ]
    con.executemany(write_overlay, rows)
    con.commit()
    return len(rows)


def export_weekly_frame(weekly: pd.DataFrame) -> pd.DataFrame:
    exported = weekly.copy()
    if "weekStartDate" not in exported.columns and "week_start" in exported.columns:
        exported.insert(0, "weekStartDate", exported["week_start"])
    return exported


def write_outputs(weekly: pd.DataFrame, state: dict[str, Any]) -> None:
    OUTPUT_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATE.parent.mkdir(parents=True, exist_ok=True)
    export_weekly_frame(weekly).to_csv(OUTPUT_WEEKLY, index=False)
    OUTPUT_STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ingest_ts = utc_now().isoformat()
    con = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(con)
        before_state = latest_finra_state(con)
        requested_weeks = weeks_to_fetch(before_state)
        fetched, failures = fetch_finra_weeks(requested_weeks)
        existing = load_existing_weekly(con)
        weekly = merge_weekly(existing, fetched, ingest_ts if not fetched.empty else before_state.get("latest_ingest_ts") or ingest_ts)

        if weekly.empty:
            raise RuntimeError("No FINRA rows available from API or persisted ats_weekly state.")

        if not fetched.empty:
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
        "fetched_week_count": 0 if fetched.empty else len(fetched),
        "failures": failures,
        "overlay_rows": overlay_rows,
        "before": before_state,
        "after": after_state,
    }
    write_outputs(weekly, output_state)
    mode = "network_refresh" if requested_weeks else "cache_rebuild_only"
    print(
        f"OK: {OUTPUT_WEEKLY} | mode={mode} | weeks={len(weekly)} | "
        f"requested={len(requested_weeks)} | fetched={output_state['fetched_week_count']} | "
        f"overlay_rows={overlay_rows} | failures={failures}"
    )


if __name__ == "__main__":
    main()
