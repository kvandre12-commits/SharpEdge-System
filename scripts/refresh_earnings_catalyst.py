#!/usr/bin/env python3
"""Print the advisory earnings-catalyst verification report on demand.

Automates event_calendar.py's "VERIFY each quarter" toil: checks each mega-cap
earnings headliner against the network (issuer IR page + SEC EDGAR) and prints
what is stale / mismatched / suggested, plus SEC filing-newness re-underwrite
flags. It NEVER edits the canonical schedule — the operator reads the report and
updates cockpit/event_calendar.py::EARNINGS_DATES by hand.

State (previous SEC accessions, for filing-newness across runs) lives OUTSIDE the
repo at $SHARPEDGE_CATALYST_STATE (default ~/.sharpedge/catalyst_state.json).

Usage:
  python3 scripts/refresh_earnings_catalyst.py            # summary lines
  python3 scripts/refresh_earnings_catalyst.py --json     # full report JSON
  python3 scripts/refresh_earnings_catalyst.py --no-state # ignore/skip state
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_COCKPIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cockpit")
if _COCKPIT_DIR not in sys.path:
    sys.path.insert(0, _COCKPIT_DIR)
from earnings_catalyst import build_earnings_catalyst_report, summarize

DEFAULT_STATE_PATH = os.path.expanduser(
    os.getenv("SHARPEDGE_CATALYST_STATE", "~/.sharpedge/catalyst_state.json")
)


def load_prev_accessions(path: str) -> dict[str, str]:
    """Return the last run's accession map, or empty on any read problem."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        accessions = data.get("accessions", {})
        return {str(k): str(v) for k, v in accessions.items()}
    except (OSError, ValueError):
        return {}


def save_accessions(path: str, accessions: dict[str, str]) -> None:
    """Persist the current accession map for the next run's newness diff."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"accessions": accessions}, handle, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--no-state", action="store_true", help="skip state load/save")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="state file path")
    args = parser.parse_args(argv)

    prev = {} if args.no_state else load_prev_accessions(args.state)
    report = build_earnings_catalyst_report(previous_accessions=prev)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("\n".join(summarize(report)))

    if not args.no_state:
        save_accessions(args.state, report.get("current_accessions", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
