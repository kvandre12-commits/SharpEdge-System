#!/usr/bin/env python3
"""Backfill decision-receipt outcomes from daily truth and optional trade logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from decision_receipts import update_receipt_outcomes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt-path",
        default="outputs/permission_receipts_spy.jsonl",
        help="JSONL receipt ledger to enrich",
    )
    parser.add_argument(
        "--truth-path",
        default="outputs/spy_truth_daily.csv",
        help="Daily OHLC truth CSV",
    )
    parser.add_argument(
        "--db-path",
        default="data/spy_truth.db",
        help="Optional sqlite db containing trades or trade_execution_log",
    )
    parser.add_argument(
        "--signal-path",
        default="outputs/signal.json",
        help="Optional live signal artifact to refresh if latest receipt matches",
    )
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args()

    summary = update_receipt_outcomes(
        receipt_path=Path(args.receipt_path),
        truth_path=Path(args.truth_path),
        db_path=Path(args.db_path) if args.db_path else None,
        signal_path=Path(args.signal_path) if args.signal_path else None,
        symbol=args.symbol,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
