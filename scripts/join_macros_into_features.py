#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime

FEATURES_PATH = "outputs/spy_features_daily.csv"
MACRO_PATH = "outputs/spy_macro_overlays_daily.csv"
OUT_PATH = "outputs/spy_features_daily_with_macro.csv"
OVERLAY_COLS = ["vix", "vix9d", "vix_term", "rates10y"]


def normalize_date(value: str) -> str:
    return datetime.fromisoformat(str(value)[:10]).date().isoformat()


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_macro_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        date_text = normalize_date(row["date"])
        symbol = row.get("symbol", "")
        overlay_type = row.get("overlay_type", "")
        if not date_text or not symbol or not overlay_type:
            continue
        try:
            strength = float(row.get("overlay_strength", "0") or 0.0)
        except ValueError:
            strength = 0.0
        by_key[(date_text, symbol)][overlay_type] = strength
    return {
        date_text: values
        for (date_text, symbol), values in by_key.items()
        if symbol == "SPY"
    }


def main() -> None:
    features = read_csv_rows(FEATURES_PATH)
    macro = read_csv_rows(MACRO_PATH)
    macro_lookup = build_macro_lookup(macro)

    if not features:
        raise RuntimeError(f"No feature rows found in {FEATURES_PATH}")

    fieldnames = list(features[0].keys())
    for column in OVERLAY_COLS:
        if column not in fieldnames:
            fieldnames.append(column)

    out_rows = []
    for row in features:
        date_text = normalize_date(row["date"])
        merged = dict(row)
        merged["date"] = date_text
        overlay_values = macro_lookup.get(date_text, {})
        for column in OVERLAY_COLS:
            merged[column] = overlay_values.get(column, 0.0)
        out_rows.append(merged)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"OK: wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
