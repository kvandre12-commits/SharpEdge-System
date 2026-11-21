#!/usr/bin/env python3
"""Prove compounding evidence and review publication survive a process crash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.compounding_agent import build_candidate, summarize_candidates
from scripts.agents.deterministic_cycle_demo import write_atomic
from scripts.agents.recoverable_cycle_demo import RecoveryLedger


def frozen_candidates() -> list[dict[str, Any]]:
    """Return three distinct settled synthetic observations."""
    return [
        build_candidate(
            cycle_receipt_id=f"settled-cycle-{number}",
            adjudication={"receipt_id": f"adjudication-{number}"},
            outcome={
                "receipt_id": f"outcome-{number}",
                "observed_delta_pct": delta,
            },
        )
        for number, delta in enumerate((15.0, 12.0, 9.0), start=1)
    ]


def run_recoverable_compounding(
    *,
    state_dir: Path,
    output: Path,
    crash_during_publication: bool = False,
) -> dict[str, Any]:
    """Stage evidence, publish one review, and rebuild its projection."""
    candidates = frozen_candidates()
    summary = summarize_candidates(candidates)
    inserted: dict[str, Any] = {}

    with RecoveryLedger(state_dir / "cycle-ledger.sqlite3") as ledger:
        inserted["candidates"] = [
            ledger.record_compounding_candidate(candidate) for candidate in candidates
        ]
        inserted["review"] = ledger.publish_compounding_review(
            summary,
            crash_before_commit=crash_during_publication,
        )
        counts = ledger.compounding_counts()

    write_atomic(output, summary)
    return {
        "receipt_id": summary["receipt_id"],
        "inserted": inserted,
        "counts": counts,
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crash-before-review", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_recoverable_compounding(
        state_dir=args.state_dir,
        output=args.output,
        crash_during_publication=args.crash_before_review,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
