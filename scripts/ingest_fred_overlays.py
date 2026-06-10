#!/usr/bin/env python3
"""Ingest/cache FRED macro overlays for the daily regime layer."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pandas as pd
import requests

try:
    from scripts.utils.pipeline_state import parse_date, utc_now, write_state
except ModuleNotFoundError:  # pragma: no cover - path execution fallback
    from utils.pipeline_state import parse_date, utc_now, write_state

FRED_API_KEY = os.getenv("FRED_API_KEY")
SYMBOL = os.getenv("SYMBOL", "SPY")
OUTPUT_PATH = "outputs/spy_macro_overlays_daily.csv"
DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")
FRED_START = os.getenv("FRED_START", "2000-01-01")
MAX_LAG_DAYS = int(os.getenv("FRED_MAX_LAG_DAYS", "5"))
FORCE_REFRESH = os.getenv("FRED_FORCE_REFRESH", "0").strip().lower() in {"1", "true", "yes", "y"}

SERIES = {
    "VIXCLS": "vix",
    "VXVCLS": "vix3m",
    "DGS10": "rates10y",
}
OVERLAY_TYPES = ("vix", "vix3m", "vix_term", "rates10y")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def ensure_overlay_table(con: sqlite3.Connection) -> None:
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


def latest_macro_state(con: sqlite3.Connection) -> dict[str, Any]:
    ensure_overlay_table(con)
    placeholders = ",".join("?" for _ in OVERLAY_TYPES)
    row = con.execute(
        f"""
        SELECT COUNT(*) AS rows,
               MIN(date) AS earliest_date,
               MAX(date) AS latest_date
        FROM overlays_daily
        WHERE symbol = ? AND overlay_type IN ({placeholders})
        """,
        (SYMBOL, *OVERLAY_TYPES),
    ).fetchone()
    return {"rows": row[0] or 0, "earliest_date": row[1], "latest_date": row[2]}


def macro_is_fresh(state: dict[str, Any]) -> bool:
    if FORCE_REFRESH or not state.get("rows"):
        return False
    latest = parse_date(state.get("latest_date"))
    if latest is None:
        return False
    lag_days = (utc_now().date() - latest).days
    return 0 <= lag_days <= MAX_LAG_DAYS


def fetch_series(series_id: str, start: str = FRED_START) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    response = requests.get(FRED_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"FRED error {response.status_code} for series_id={series_id}. "
            f"Response (truncated): {response.text[:200]}"
        )

    rows = []
    for observation in response.json().get("observations", []):
        value = observation.get("value")
        if value is None or value == ".":
            continue
        rows.append({"date": pd.to_datetime(observation["date"]), series_id: float(value)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No observations returned for series_id={series_id}")
    return frame


def zscore(series: pd.Series, win: int = 252) -> pd.Series:
    mean = series.rolling(win, min_periods=40).mean()
    std = series.rolling(win, min_periods=40).std()
    return (series - mean) / std


def z_to_strength(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, (value - 1.0) / 1.5)))


def build_macro_frame() -> pd.DataFrame:
    if not FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY env var (set it in GitHub Secrets).")

    frames = [fetch_series(series_id) for series_id in SERIES]
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")

    macro = macro.sort_values("date").ffill()
    macro["vix_term"] = macro["VIXCLS"] / macro["VXVCLS"]
    macro["vix_z"] = zscore(macro["VIXCLS"])
    macro["vix3m_z"] = zscore(macro["VXVCLS"])
    macro["vix_term_z"] = zscore(macro["vix_term"])
    macro["rates_z"] = zscore(macro["DGS10"].diff(5))
    return macro


def build_overlay_frame(macro: pd.DataFrame) -> pd.DataFrame:
    overlays: list[dict[str, Any]] = []
    for _, row in macro.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        overlays.extend(
            [
                {
                    "date": date,
                    "symbol": SYMBOL,
                    "overlay_type": "vix",
                    "overlay_strength": z_to_strength(row["vix_z"]),
                    "raw_value": row["VIXCLS"],
                    "z_score": row["vix_z"],
                    "notes": "VIX spot level (VIXCLS)",
                },
                {
                    "date": date,
                    "symbol": SYMBOL,
                    "overlay_type": "vix3m",
                    "overlay_strength": z_to_strength(row["vix3m_z"]),
                    "raw_value": row["VXVCLS"],
                    "z_score": row["vix3m_z"],
                    "notes": "3M VIX level (VXVCLS)",
                },
                {
                    "date": date,
                    "symbol": SYMBOL,
                    "overlay_type": "vix_term",
                    "overlay_strength": z_to_strength(row["vix_term_z"]),
                    "raw_value": row["vix_term"],
                    "z_score": row["vix_term_z"],
                    "notes": "Term structure proxy (VIXCLS / VXVCLS)",
                },
                {
                    "date": date,
                    "symbol": SYMBOL,
                    "overlay_type": "rates10y",
                    "overlay_strength": z_to_strength(row["rates_z"]),
                    "raw_value": row["DGS10"],
                    "z_score": row["rates_z"],
                    "notes": "10Y rate impulse (DGS10 5D chg z-score)",
                },
            ]
        )
    return pd.DataFrame(overlays)


def upsert_overlays(con: sqlite3.Connection, overlays: pd.DataFrame) -> int:
    ensure_overlay_table(con)
    rows = [
        (
            str(row["date"]),
            str(row["symbol"]),
            str(row["overlay_type"]),
            float(row["overlay_strength"]) if pd.notna(row["overlay_strength"]) else 0.0,
            f'{row.get("notes", "")}|raw={row.get("raw_value", "")}|z={row.get("z_score", "")}',
        )
        for _, row in overlays.iterrows()
    ]
    con.executemany(
        """
        INSERT INTO overlays_daily (date, symbol, overlay_type, overlay_strength, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date, overlay_type) DO UPDATE SET
          overlay_strength=excluded.overlay_strength,
          notes=excluded.notes
        """,
        rows,
    )
    con.commit()
    return len(rows)


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        before = latest_macro_state(con)
        network_refresh = not macro_is_fresh(before)
        rows = 0
        output_rows = 0

        if network_refresh:
            overlays = build_overlay_frame(build_macro_frame())
            rows = upsert_overlays(con, overlays)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            overlays.to_csv(OUTPUT_PATH, index=False)
            output_rows = len(overlays)
        elif os.path.exists(OUTPUT_PATH):
            output_rows = sum(1 for _ in open(OUTPUT_PATH, encoding="utf-8")) - 1

        after = latest_macro_state(con)
    finally:
        con.close()

    state = {
        "symbol": SYMBOL,
        "max_lag_days": MAX_LAG_DAYS,
        "force_refresh": FORCE_REFRESH,
        "network_refresh": network_refresh,
        "upserted_rows": rows,
        "output_rows": output_rows,
        "before": before,
        "after": after,
    }
    write_state("fred", state)
    mode = "network_refresh" if network_refresh else "cache_reuse"
    print(
        f"OK: FRED macro overlays | mode={mode} | upserted={rows} | "
        f"output_rows={output_rows} | latest={after.get('latest_date')}"
    )


if __name__ == "__main__":
    main()
