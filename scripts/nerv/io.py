"""Artifact writers for NERV snapshots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import NERVSnapshot
from .scorer import build_liquidity_board, enrich_quote_record


CSV_FIELDS = [
    "underlying",
    "contract_symbol",
    "option_type",
    "expiration",
    "strike",
    "underlying_price",
    "bid",
    "ask",
    "midpoint",
    "bid_ask_width",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "in_the_money",
    "moneyness",
    "quote_quality_score",
    "liquidity_score",
    "nerv_score",
    "width_pct",
    "manual_validation_priority",
    "rejection_flags",
    "fresh_quote_required",
    "source",
    "data_mode",
    "quote_timestamp",
    "fetch_timestamp",
    "quote_age_seconds",
    "research_only_warning",
]


def write_snapshot_json(
    snapshot: NERVSnapshot,
    output_dir: str | Path,
    *,
    include_raw: bool = False,
    name: str = "nerv_options_snapshot.json",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    payload = _snapshot_payload(snapshot, include_raw=include_raw)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_snapshot_csv(
    snapshot: NERVSnapshot,
    output_dir: str | Path,
    *,
    name: str = "nerv_options_snapshot.csv",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for quote in snapshot.quotes:
            writer.writerow(enrich_quote_record(quote.to_record(include_raw=False)))
    return path


def write_liquidity_board_json(
    snapshot: NERVSnapshot,
    output_dir: str | Path,
    *,
    limit: int = 50,
    name: str = "nerv_liquidity_board.json",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    records = [quote.to_record(include_raw=False) for quote in snapshot.quotes]
    board = build_liquidity_board(records, limit=limit)
    payload = {
        "schema": "sharpedge.nerv_liquidity_board.v1",
        "summary": {**snapshot.summary(), "board_limit": limit, "board_count": len(board)},
        "contracts": board,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_liquidity_board_csv(
    snapshot: NERVSnapshot,
    output_dir: str | Path,
    *,
    limit: int = 50,
    name: str = "nerv_liquidity_board.csv",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    records = [quote.to_record(include_raw=False) for quote in snapshot.quotes]
    board = build_liquidity_board(records, limit=limit)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(board)
    return path


def _snapshot_payload(snapshot: NERVSnapshot, *, include_raw: bool = False) -> dict[str, Any]:
    payload = snapshot.to_payload(include_raw=include_raw)
    payload["quotes"] = [
        enrich_quote_record(quote.to_record(include_raw=include_raw))
        for quote in snapshot.quotes
    ]
    return payload


def write_provider_status_json(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "nerv_provider_status.json",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return path
