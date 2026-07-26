#!/usr/bin/env python3
"""Run NERV-1 free-data option-chain discovery.

Default path uses yfinance because it requires no credentials. Alpaca/Tradier are
reported as credential seams for the next official-API implementation step.

Examples:
  python3 scripts/nerv_free_data_adapter.py --symbols STNG,FRO,DHT --max-expirations 2
  python3 scripts/nerv_free_data_adapter.py --symbols SPY --expirations 2026-01-16 --include-raw
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.io import (  # noqa: E402
    write_liquidity_board_csv,
    write_liquidity_board_json,
    write_provider_status_json,
    write_snapshot_csv,
    write_snapshot_json,
)
from nerv.provider_status import provider_statuses  # noqa: E402
from nerv.runtime_retention import DEFAULT_RETENTION_HOURS, prune_stale_files  # noqa: E402
from nerv.yfinance_adapter import YFinanceOptionsAdapter  # noqa: E402

DEFAULT_SYMBOLS = [
    "STNG",
    "FRO",
    "DHT",
    "TNK",
    "INSW",
    "LPG",
    "LNG",
    "BKR",
    "SLB",
    "HAL",
    "VLO",
    "OKE",
]


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NERV-1 free options-data adapter: yfinance discovery/export.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated underlyings. Defaults to the NERV manual-validation queue.",
    )
    parser.add_argument(
        "--expirations",
        default="",
        help="Optional comma-separated expirations YYYY-MM-DD. If omitted, first N expirations are used.",
    )
    parser.add_argument(
        "--max-expirations",
        type=int,
        default=int(os.getenv("NERV_MAX_EXPIRATIONS", "2")),
        help="Number of expirations per symbol when --expirations is omitted. Use 0 for all.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("NERV_OUTPUT_DIR", "outputs/nerv"),
        help="Directory for JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--json-name",
        default="nerv_options_snapshot.json",
        help="Snapshot JSON filename.",
    )
    parser.add_argument(
        "--csv-name",
        default="nerv_options_snapshot.csv",
        help="Snapshot CSV filename.",
    )
    parser.add_argument(
        "--board-limit",
        type=int,
        default=int(os.getenv("NERV_BOARD_LIMIT", "50")),
        help="Number of top-scored contracts to write to the liquidity board. Use 0 for all.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw yfinance rows in JSON output. No effect on CSV.",
    )
    parser.add_argument(
        "--retention-hours",
        type=float,
        default=float(
            os.getenv("NERV_OUTPUT_RETENTION_HOURS", DEFAULT_RETENTION_HOURS)
        ),
        help="Opportunistically delete stale files in the output dir before writing. Use 0 to disable.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Write provider status and exit without fetching chains.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    deleted = prune_stale_files(output_dir, max_age_hours=args.retention_hours)
    if deleted:
        print(f"[nerv] pruned stale artifacts: {len(deleted)}")
    statuses = [status.to_record() for status in provider_statuses()]
    status_path = write_provider_status_json(statuses, output_dir)

    print(f"[nerv] provider status: {status_path}")
    for status in statuses:
        configured = "configured" if status["configured"] else "missing credentials"
        print(f"[nerv] {status['name']}: {configured} ({status['data_mode']})")

    if args.status_only:
        return 0

    symbols = parse_csv_arg(args.symbols)
    expirations = parse_csv_arg(args.expirations)
    max_expirations = None if args.max_expirations <= 0 else args.max_expirations

    adapter = YFinanceOptionsAdapter()
    snapshot = adapter.fetch(
        symbols,
        max_expirations=max_expirations,
        expirations=expirations or None,
    )
    json_path = write_snapshot_json(
        snapshot,
        output_dir,
        include_raw=args.include_raw,
        name=args.json_name,
    )
    csv_path = write_snapshot_csv(snapshot, output_dir, name=args.csv_name)
    board_json_path = write_liquidity_board_json(
        snapshot,
        output_dir,
        limit=args.board_limit,
    )
    board_csv_path = write_liquidity_board_csv(
        snapshot,
        output_dir,
        limit=args.board_limit,
    )
    summary = snapshot.summary()

    print(f"[nerv] snapshot json: {json_path}")
    print(f"[nerv] snapshot csv: {csv_path}")
    print(f"[nerv] liquidity board json: {board_json_path}")
    print(f"[nerv] liquidity board csv: {board_csv_path}")
    print(
        "[nerv] quotes={quote_count} symbols={symbols_with_quotes} errors={error_count}".format(
            **summary
        )
    )
    if summary["error_count"]:
        for error in summary["errors"]:
            print(f"[nerv] WARNING {error['symbol']}: {error['error']}")
    _print_top_board(board_json_path)
    print("[nerv] research-only: confirm final executable prices at broker.")
    return 0


def _print_top_board(board_json_path: Path, *, limit: int = 8) -> None:
    import json

    payload = json.loads(board_json_path.read_text(encoding="utf-8"))
    contracts = payload.get("contracts", [])[:limit]
    if not contracts:
        print("[nerv] liquidity board: no scored contracts")
        return
    print("[nerv] top liquidity candidates:")
    for row in contracts:
        print(
            "[nerv]   {underlying:5} {expiration} {cp} {strike:8.2f} "
            "score={score:5.1f} priority={priority:6} vol={vol} oi={oi} "
            "width_pct={width_pct} flags={flags}".format(
                underlying=row["underlying"],
                expiration=row["expiration"],
                cp=row["option_type"][0].upper(),
                strike=float(row["strike"]),
                score=float(row.get("nerv_score") or 0),
                priority=row.get("manual_validation_priority"),
                vol=row.get("volume"),
                oi=row.get("open_interest"),
                width_pct=row.get("width_pct"),
                flags=row.get("rejection_flags") or "none",
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
