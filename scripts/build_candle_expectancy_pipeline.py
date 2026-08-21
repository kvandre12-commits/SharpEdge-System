#!/usr/bin/env python3
"""Run the candle-conditioned expectancy pipeline in order.

Order:
    1. build_candle_expectancy_events.py
    2. build_candle_conditioned_expectancy_matrix.py
    3. build_candle_confidence_weights.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

STAGES = [
    "build_candle_expectancy_events.py",
    "build_candle_conditioned_expectancy_matrix.py",
    "build_candle_confidence_weights.py",
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
                f"\nFAILED at {stage} (exit {result.returncode}). Stopping before stale matrix nonsense.",
                file=sys.stderr,
            )
            return result.returncode
    print("\nOK: candle-conditioned expectancy pipeline complete (3/3 stages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
