#!/usr/bin/env python3
"""Nominate fresh, liquid symbols for separately frozen Paper Boy lanes."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.paper_boy_scout import (  # noqa: E402
    build_scout_payload,
    load_json,
    write_scout_artifacts,
)

DEFAULT_BOARD = ROOT / "outputs" / "nerv_paper_boy_scan" / "nerv_liquidity_board.json"
DEFAULT_UNIVERSE = ROOT / "outputs" / "catalyst_paper_agents" / "latest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "nerv_paper_boy_scout"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--catalyst-universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-snapshot-age-minutes", type=float, default=45.0)
    parser.add_argument("--max-contract-age-seconds", type=int, default=20 * 60)
    parser.add_argument("--min-nerv-score", type=float, default=65.0)
    parser.add_argument("--min-usable-contracts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    catalyst = (
        load_json(args.catalyst_universe) if args.catalyst_universe.exists() else None
    )
    payload = build_scout_payload(
        load_json(args.board),
        catalyst_universe=catalyst,
        as_of=datetime.now(UTC),
        max_snapshot_age_minutes=args.max_snapshot_age_minutes,
        max_contract_age_seconds=args.max_contract_age_seconds,
        min_nerv_score=args.min_nerv_score,
        min_usable_contracts=args.min_usable_contracts,
        limit=args.limit,
    )
    paths = write_scout_artifacts(payload, args.output_dir)
    symbols = payload["summary"]["nominated_symbols"]
    print(f"[nerv/paper-boy] nominations: {','.join(symbols) or 'none'}")
    print(f"[nerv/paper-boy] source fresh: {payload['source']['snapshot_fresh']}")
    for kind, path in paths.items():
        print(f"[nerv/paper-boy] {kind}: {path}")
    print("[nerv/paper-boy] nomination only; a new frozen manifest is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
