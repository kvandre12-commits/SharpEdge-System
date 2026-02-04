import os
import requests
import pandas as pd
from datetime import datetime

FRED_API_KEY = os.getenv("FRED_API_KEY")
SYMBOL = "SPY"

OUTPUT_PATH = "outputs/spy_macro_overlays_daily.csv"

SERIES = {
    "VIXCLS": "vix",
    "VIX9D": "vix3m",
    "DGS10": "rates10y",
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id, start="2000-01-01"):
    params = {
    "series_id": series_id,
    "api_key": FRED_API_KEY,
    "file_type": "json",
    "observation_start": start,
}
    r = requests.get(FRED_URL, params=params, timeout=30)
    r.raise_for_status()

    rows = []
    for o in r.json()["observations"]:
        if o["value"] == ".":
            continue
        rows.append({
            "date": pd.to_datetime(o["date"]),
            series_id: float(o["value"]),
        })

    return pd.DataFrame(rows)


def zscore(s, win=252):
    mu = s.rolling(win, min_periods=40).mean()
    sd = s.rolling(win, min_periods=40).std()
    return (s - mu) / sd


def z_to_strength(z):
    if pd.isna(z):
        return 0.0
    return max(0.0, min(1.0, (z - 1.0) / 1.5))


def main():
    if not FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY env var")

    dfs = []
    for sid in SERIES:
        df = fetch_series(sid)
        dfs.append(df)

    macro = dfs[0]
    for d in dfs[1:]:
        macro = macro.merge(d, on="date", how="outer")

    macro = macro.sort_values("date").ffill()

    overlays = []

    # VIX level
    macro["vix_z"] = zscore(macro["VIXCLS"])
    # VIX9D level
    macro["vixCLS_z"] = zscore(macro["VIXCLS"])
    # Term structure
    macro["vix_term"] = macro["VIX9D"] / macro["VIXCLS"]
    macro["vix_term_z"] = zscore(macro["vix_term"])
    # Rates impulse
    macro["rates_z"] = zscore(macro["DGS10"].diff(5))

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
                "notes": "VIX level",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "vixCLS",
                "overlay_strength": z_to_strength(r["vixCLS_z"]),
                "raw_value": r["VIXCLS"],
                "z_score": r["vixCLS_z"],
                "notes": "3mon VIX level",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "vix_term",
                "overlay_strength": z_to_strength(r["vix_term_z"]),
                "raw_value": r["vix_term"],
                "z_score": r["vix_term_z"],
                "notes": "VIX3MON / VIX",
            },
            {
                "date": date,
                "symbol": SYMBOL,
                "overlay_type": "rates10y",
                "overlay_strength": z_to_strength(r["rates_z"]),
                "raw_value": r["DGS10"],
                "z_score": r["rates_z"],
                "notes": "10Y rate impulse",
            },
        ])

    out = pd.DataFrame(overlays)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"OK: wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
