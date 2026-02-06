import os
import requests
import pandas as pd
import sqlite3

FRED_API_KEY = os.getenv("FRED_API_KEY")
SYMBOL = "SPY"

OUTPUT_PATH = "outputs/spy_macro_overlays_daily.csv"
DB_PATH = os.getenv("SPY_DB_PATH", "data/spy_truth.db")

# FRED-valid series:
# - VIXCLS: CBOE VIX close
# - VXVCLS: CBOE 3-Month VIX close (FRED proxy for short-term structure context)
# - DGS10: 10Y Treasury yield
SERIES = {
    "VIXCLS": "vix",
    "VXVCLS": "vix3m",
    "DGS10": "rates10y",
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch one FRED series into a 2-col frame: [date, <series_id>]."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(FRED_URL, params=params, timeout=30)

    # Helpful error if FRED rejects the request
    if r.status_code != 200:
        raise RuntimeError(
            f"FRED error {r.status_code} for series_id={series_id}. "
            f"Response (truncated): {r.text[:200]}"
        )

    data = r.json()
    obs = data.get("observations", [])

    rows = []
    for o in obs:
        v = o.get("value")
        if v is None or v == ".":
            continue
        rows.append({"date": pd.to_datetime(o["date"]), series_id: float(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No observations returned for series_id={series_id}")
    return df


def zscore(s: pd.Series, win: int = 252) -> pd.Series:
    """Rolling z-score with a reasonable min_periods so early history doesn't NaN forever."""
    mu = s.rolling(win, min_periods=40).mean()
    sd = s.rolling(win, min_periods=40).std()
    return (s - mu) / sd


def z_to_strength(z) -> float:
    """
    Map z-score to [0,1]:
    - 0 until z>=1.0
    - ramps to 1 by z>=2.5
    """
    if pd.isna(z):
        return 0.0
    return float(max(0.0, min(1.0, (z - 1.0) / 1.5)))


def main():
    if not FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY env var (set it in GitHub Secrets).")

    # 1) Fetch all series and outer-join on date
    frames = []
    for sid in SERIES.keys():
        frames.append(fetch_series(sid))

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="date", how="outer")

    macro = macro.sort_values("date").ffill()

    # 2) Derived series
    # Term structure proxy (stress when >1 i.e., backwardation-like)
    macro["vix_term"] = macro["VIXCLS"] / macro["VXVCLS"]

    # 3) Z-scores
    macro["vix_z"] = zscore(macro["VIXCLS"])
    macro["vix3m_z"] = zscore(macro["VXVCLS"])
    macro["vix_term_z"] = zscore(macro["vix_term"])

    # Rates impulse: 5-day change, then z-score
    macro["rates_z"] = zscore(macro["DGS10"].diff(5))

    # 4) Build long-form overlays output
    overlays = []
    for _, r in macro.iterrows():
        date = r["date"].strftime("%Y-%m-%d")

        overlays.extend([
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "vix",
                "overlay_strength": z_to_strength(r["vix_z"]),
                "raw_value": r["VIXCLS"],
                "z_score": r["vix_z"],
                "notes": "VIX spot level (VIXCLS)",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "vix3m",
                "overlay_strength": z_to_strength(r["vix3m_z"]),
                "raw_value": r["VXVCLS"],
                "z_score": r["vix3m_z"],
                "notes": "3M VIX level (VXVCLS)",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "vix_term",
                "overlay_strength": z_to_strength(r["vix_term_z"]),
                "raw_value": r["vix_term"],
                "z_score": r["vix_term_z"],
                "notes": "Term structure proxy (VIXCLS / VXVCLS)",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "rates10y",
                "overlay_strength": z_to_strength(r["rates_z"]),
                "raw_value": r["DGS10"],
                "z_score": r["rates_z"],
                "notes": "10Y rate impulse (DGS10 5D chg z-score)",
            },
        ])

    out = pd.DataFrame(overlays)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"OK: wrote {OUTPUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
