#!/usr/bin/env python3
import os, json, time, sqlite3
import datetime as dt
import pandas as pd
import requests

DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
SYMBOL = os.getenv("SYMBOL", "SPY")

FINRA_API_KEY = os.getenv("FINRA_API_KEY", "")
BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
DATASET_PRIMARY = "weeklySummaryHistoric"
DATASET_FALLBACK = "weeklySummary"

TIER = os.getenv("FINRA_TIER", "T1")
SUMMARY_TYPE = os.getenv("FINRA_SUMMARY_TYPE", "ATS_W_SMBL")

START = os.getenv("FINRA_START", "2024-01-01")
LIMIT = int(os.getenv("FINRA_LIMIT", "5000"))

def monday_of_week(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())

def daterange_mondays(start: dt.date, end: dt.date):
    s = monday_of_week(start)
    e = monday_of_week(end)
    cur = s
    while cur <= e:
        yield cur
        cur += dt.timedelta(days=7)

def build_payload(week_start: str):
    return {
        "compareFilters": [
            {"compareType": "equal", "fieldName": "issueSymbolIdentifier", "fieldValue": SYMBOL},
            {"compareType": "equal", "fieldName": "tierIdentifier", "fieldValue": TIER},
            {"compareType": "equal", "fieldName": "summaryTypeCode", "fieldValue": SUMMARY_TYPE},
            {"compareType": "equal", "fieldName": "weekStartDate", "fieldValue": week_start},
        ],
        "limit": LIMIT,
        "offset": 0
    }

def post_rows(dataset: str, payload: dict, headers: dict):
    url = f"{BASE_URL}/{dataset}"
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if r.status_code in (401, 403):
        raise PermissionError(f"{r.status_code} from FINRA. Set FINRA_API_KEY secret.")
    r.raise_for_status()
    data = r.json()
    return data["data"] if isinstance(data, dict) and "data" in data else data

def fetch_week(week_start: dt.date, headers: dict) -> pd.DataFrame:
    ws = week_start.isoformat()
    payload = build_payload(ws)
    try:
        rows = post_rows(DATASET_PRIMARY, payload, headers)
    except Exception:
        rows = post_rows(DATASET_FALLBACK, payload, headers)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def ensure_tables(con: sqlite3.Connection):
    con.execute("""
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
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS overlays_daily (
      date TEXT NOT NULL,
      symbol TEXT NOT NULL,
      overlay_type TEXT NOT NULL,
      overlay_strength REAL NOT NULL,
      notes TEXT,
      PRIMARY KEY (symbol, date, overlay_type)
    )
    """)
    con.commit()

def upsert_weekly(con: sqlite3.Connection, weekly: pd.DataFrame):
    cols = [
        "week_start","symbol","ats_weekly_shares","ats_weekly_trades","ats_venue_count",
        "avg_trade_size","shares_vs_13w_avg","trades_vs_13w_avg","shares_z_26w","ingest_ts"
    ]
    sql = f"""
    INSERT INTO ats_weekly ({",".join(cols)})
    VALUES ({",".join(["?"]*len(cols))})
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
    con.executemany(sql, weekly[cols].values.tolist())
    con.commit()

def z_to_strength(z: float) -> float:
    # 0 until z>=1.0, ramps to 1 by z>=2.5
    if z is None or pd.isna(z):
        return 0.0
    return float(max(0.0, min(1.0, (z - 1.0) / 1.5)))

def write_daily_overlay(con: sqlite3.Connection, weekly: pd.DataFrame):
    days = pd.read_sql_query(
        "SELECT date FROM bars_daily WHERE symbol=? ORDER BY date ASC",
        con, params=(SYMBOL,)
    )
    if days.empty:
        return
    days["date"] = pd.to_datetime(days["date"])
    w = weekly.copy()
    w["week_start"] = pd.to_datetime(w["week_start"])
    w = w.sort_values("week_start")

    # FINRA week is Monday; map each day to its Monday
    days["week_start"] = days["date"].map(lambda d: monday_of_week(d.date()))
    j = days.merge(w[["week_start","shares_z_26w"]], on="week_start", how="left")

    j["overlay_strength"] = j["shares_z_26w"].apply(z_to_strength)
    j["notes"] = j["shares_z_26w"].apply(lambda z: f"finra_ats_shares_z_26w={z:.2f}" if pd.notna(z) else "finra_ats_missing")

    sql = """
    INSERT INTO overlays_daily (date, symbol, overlay_type, overlay_strength, notes)
    VALUES (?, ?, 'darkpool', ?, ?)
    ON CONFLICT(symbol, date, overlay_type) DO UPDATE SET
      overlay_strength=excluded.overlay_strength,
      notes=excluded.notes
    """
    con.executemany(sql, [
        (d.strftime("%Y-%m-%d"), SYMBOL, float(s), str(n))
        for d, s, n in zip(j["date"], j["overlay_strength"], j["notes"])
    ])
    con.commit()

def main():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "spy-finra-darkpool/1.0",
    }
    if FINRA_API_KEY:
        headers["X-API-KEY"] = FINRA_API_KEY

    start = dt.date.fromisoformat(START)
    end = dt.date.today()

    frames = []
    for m in daterange_mondays(start, end):
        dfw = fetch_week(m, headers=headers)
        if not dfw.empty:
            frames.append(dfw)
        time.sleep(0.15)

    if not frames:
        raise RuntimeError("No FINRA rows returned. Check FINRA_API_KEY and filters.")

    raw = pd.concat(frames, ignore_index=True)

    def pick(cands):
        for c in cands:
            if c in raw.columns:
                return c
        raise KeyError(f"None of these columns found: {cands}")

    col_week = pick(["weekStartDate","weekstartdate"])
    col_symbol = pick(["issueSymbolIdentifier","issuesymbolidentifier"])
    col_mpid = pick(["marketParticipantIdentifier","marketparticipantidentifier","MPID","mpid"])
    col_shares = pick(["totalWeeklyShareQuantity","totalweeklysharequantity","totalsharequantitysum"])
    col_trades = pick(["totalWeeklyTradeCount","totalweeklytradecount","totaltradecountsum"])

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
           .sort_values("week_start")
           .reset_index(drop=True)
    )

    weekly["avg_trade_size"] = weekly["ats_weekly_shares"] / weekly["ats_weekly_trades"].replace(0, pd.NA)
    weekly["shares_vs_13w_avg"] = weekly["ats_weekly_shares"] / weekly["ats_weekly_shares"].rolling(13, min_periods=4).mean()
    weekly["trades_vs_13w_avg"] = weekly["ats_weekly_trades"] / weekly["ats_weekly_trades"].rolling(13, min_periods=4).mean()
    weekly["shares_z_26w"] = (weekly["ats_weekly_shares"] - weekly["ats_weekly_shares"].rolling(26, min_periods=8).mean()) / weekly["ats_weekly_shares"].rolling(26, min_periods=8).std()
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["ingest_ts"] = dt.datetime.utcnow().isoformat()

    con = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(con)
        upsert_weekly(con, weekly)
        write_daily_overlay(con, weekly)
    finally:
        con.close()

    os.makedirs("outputs", exist_ok=True)
    weekly.to_csv("outputs/spy_finra_ats_weekly.csv", index=False)
    print("Wrote outputs/spy_finra_ats_weekly.csv and updated overlays_daily(darkpool).")

if __name__ == "__main__":
    main()
