#!/usr/bin/env python3
"""Run the full canonical auction-expectancy pipeline, in the correct order.

Stage 1 (build_auction_expectancy_events.py) does DROP TABLE + CREATE TABLE AS,
which WIPES the enrichment columns that stages 2-3 add in place. So the four
stages must ALWAYS run together, in order. Run THIS orchestrator -- never
stage 1 alone -- so the canonical dataset and its edge matrix stay coherent.

Order + why:
    1. build_auction_expectancy_events.py      base rows (gap + fill + context)
    2. measure_gap_excursions.py               MAE/MFE/reward-risk (needs bars)
    3. classify_fill_paths.py                  fill_path_type + path labels
    4. build_conditional_expectancy_matrix.py  the conditional edge matrix

Usage:
    cd ~/SharpEdge-System
    python scripts/build_auction_expectancy_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

STAGES = [
    "build_auction_expectancy_events.py",
    "measure_gap_excursions.py",
    "classify_fill_paths.py",
    "build_conditional_expectancy_matrix.py",
]


def main() -> int:
    for stage in STAGES:
        path = os.path.join(HERE, stage)
        if not os.path.exists(path):
            print(f"FAILED: missing stage script {stage}")
            return 2
        print(f"\n=== running {stage} ===", flush=True)
        result = subprocess.run([sys.executable, path], cwd=REPO_ROOT)
        if result.returncode != 0:
            print(
                f"\nFAILED at {stage} (exit {result.returncode}). Stopping so the "
                "dataset is not left half-built.",
                file=sys.stderr,
            )
            return result.returncode
    print("\nOK: canonical auction-expectancy pipeline complete (4/4 stages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
