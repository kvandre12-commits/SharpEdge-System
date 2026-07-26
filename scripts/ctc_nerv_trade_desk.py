#!/usr/bin/env python3
"""Build the CTC/NERV research trade-desk board.

Example:
  python3 scripts/ctc_nerv_trade_desk.py \
    --ctc-workbook '/sdcard/Download/CTC_C001_Full_34_Name_Disposition_v0_5 (1).xlsx'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.ctc_trade_desk import (  # noqa: E402
    DEFAULT_CTC_WORKBOOK,
    DEFAULT_NERV_BOARD,
    DEFAULT_OUTPUT_DIR,
    build_trade_desk_payload,
    write_trade_desk_artifacts,
)
from nerv.runtime_retention import DEFAULT_RETENTION_HOURS, prune_stale_files  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join CTC workbook disposition rows with NERV liquidity board rows.",
    )
    parser.add_argument(
        "--ctc-workbook",
        default=str(DEFAULT_CTC_WORKBOOK),
        help="Path to CTC workbook .xlsx.",
    )
    parser.add_argument(
        "--nerv-board",
        default=str(DEFAULT_NERV_BOARD),
        help="Path to NERV liquidity board JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for JSON/CSV/Markdown board artifacts.",
    )
    parser.add_argument(
        "--retention-hours",
        type=float,
        default=DEFAULT_RETENTION_HOURS,
        help="Opportunistically delete stale files in the output dir before writing. Use 0 to disable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deleted = prune_stale_files(args.output_dir, max_age_hours=args.retention_hours)
    if deleted:
        print(f"[ctc/nerv] pruned stale artifacts: {len(deleted)}")
    payload = build_trade_desk_payload(
        ctc_workbook=args.ctc_workbook,
        nerv_board_path=args.nerv_board,
    )
    paths = write_trade_desk_artifacts(payload, args.output_dir)
    print(f"[ctc/nerv] rows={payload['summary']['row_count']}")
    print(f"[ctc/nerv] states={payload['summary']['states']}")
    suggested = payload["summary"].get("suggested_nerv_symbols") or []
    if suggested:
        print("[ctc/nerv] suggested NERV symbols: " + ",".join(suggested))
        print(
            "[ctc/nerv] next: python3 scripts/nerv_free_data_adapter.py "
            f"--symbols {','.join(suggested)} --max-expirations 2"
        )
    for kind, path in paths.items():
        print(f"[ctc/nerv] {kind}: {path}")
    print("[ctc/nerv] research-only: broker fresh quote + operator approval required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
