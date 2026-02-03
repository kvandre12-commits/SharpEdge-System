import pandas as pd

FEATURES_PATH = "outputs/spy_features_daily.csv"
MACRO_PATH    = "outputs/spy_macro_overlays_daily.csv"
OUT_PATH      = "outputs/spy_features_daily_with_macro.csv"

# 1) load
feat = pd.read_csv(FEATURES_PATH)
macro = pd.read_csv(MACRO_PATH)

# 2) normalize date types/format
feat["date"] = pd.to_datetime(feat["date"]).dt.strftime("%Y-%m-%d")
macro["date"] = pd.to_datetime(macro["date"]).dt.strftime("%Y-%m-%d")

# 3) pivot overlays -> wide columns
macro_wide = (
    macro.pivot_table(
        index=["date", "symbol"],
        columns="overlay_type",
        values="overlay_strength",
        aggfunc="last",
    )
    .reset_index()
)

# (optional) keep only SPY
macro_wide = macro_wide[macro_wide["symbol"] == "SPY"].drop(columns=["symbol"])

# 4) left join into features
out = feat.merge(macro_wide, on="date", how="left")

# 5) fill missing overlays with 0
overlay_cols = ["vix", "vix9d", "vix_term", "rates10y"]
for c in overlay_cols:
    if c not in out.columns:
        out[c] = 0.0
out[overlay_cols] = out[overlay_cols].fillna(0.0)

# 6) save
out.to_csv(OUT_PATH, index=False)
print(f"OK: wrote {OUT_PATH}")
