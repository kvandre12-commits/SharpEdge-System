"""CLI and cadence runner for the Paper Boy surface audit."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
import time
from typing import Any


def parse_args(
    *, input_path: Path, output_root: Path, html_path: Path
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic WHY/HOW supervisor for the Paper Boy surface comparison."
    )
    parser.add_argument("--input", type=Path, default=input_path)
    parser.add_argument("--output-json", type=Path, default=output_root / "latest.json")
    parser.add_argument(
        "--output-markdown", type=Path, default=output_root / "latest.md"
    )
    parser.add_argument("--output-html", type=Path, default=html_path)
    parser.add_argument(
        "--ledger", type=Path, default=output_root / "recommendation_ledger.jsonl"
    )
    parser.add_argument("--interval-seconds", type=float, default=0)
    return parser.parse_args()


def run_loop(
    args: argparse.Namespace,
    run_once: Callable[[argparse.Namespace], dict[str, Any]],
) -> int:
    if args.interval_seconds < 0:
        raise SystemExit("--interval-seconds must be non-negative")
    while True:
        started = time.monotonic()
        try:
            report = run_once(args)
            print(
                f"surface audit: {report['why']['status']} "
                f"recommendations={len(report['how']['recommendations'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"surface audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.interval_seconds:
                return 1
        if not args.interval_seconds:
            return 0
        elapsed = time.monotonic() - started
        time.sleep(max(0.01, args.interval_seconds - elapsed))
