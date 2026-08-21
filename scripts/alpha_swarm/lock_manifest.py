#!/usr/bin/env python3
"""Freeze a paper-only alpha-swarm evaluator manifest before predictions exist."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from scripts.alpha_swarm.contracts import (
    EXPLICIT_OUTCOME_FIELD,
    FORBIDDEN_OUTCOME_FIELDS,
    RUN_MANIFEST_SCHEMA,
    canonical_json,
    manifest_sha256,
    source_bundle_sha256,
    validate_manifest,
)

NY = ZoneInfo("America/New_York")
DEFAULT_UNIVERSE = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN")


def evaluator_source_paths() -> list[Path]:
    package = Path(__file__).resolve().parent
    return [
        package / "contracts.py",
        package / "evaluator.py",
        package / "lock_manifest.py",
    ]


def _local_iso(session: date, value: time) -> str:
    return datetime.combine(session, value, tzinfo=NY).isoformat()


def build_manifest(
    *,
    run_id: str,
    sessions: list[date],
    universe: list[str],
    locked_at: str,
    evaluator_source_sha256: str,
) -> dict[str, Any]:
    symbols = [symbol.strip().upper() for symbol in universe]
    slots = []
    for session in sessions:
        for symbol in symbols:
            slots.append(
                {
                    "slot_id": f"{session.isoformat()}-1045-{symbol}",
                    "session_date": session.isoformat(),
                    "symbol": symbol,
                    "eligible": True,
                    "unavailable_reason": None,
                    "eligibility_declared_at": _local_iso(session, time(10, 30)),
                    "prediction_ts": _local_iso(session, time(10, 45)),
                    "entry_ts": _local_iso(session, time(10, 50)),
                    "exit_ts": _local_iso(session, time(15, 45)),
                    "label_available_ts": _local_iso(session, time(16, 15)),
                }
            )
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "locked_at": locked_at,
        "evaluator_version": "1.0.0",
        "evaluator_source_sha256": evaluator_source_sha256,
        "evaluator_source_files": [path.name for path in evaluator_source_paths()],
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "universe": symbols,
        "slots": slots,
        "label_contract": {
            "outcome_field": EXPLICIT_OUTCOME_FIELD,
            "forbidden_outcome_fields": sorted(FORBIDDEN_OUTCOME_FIELDS),
            "definition": "vehicle value from locked prediction-to-exit timestamps after costs",
            "feature_time_field": "feature_available_ts",
            "prediction_time_field": "prediction_ts",
            "entry_time_field": "entry_ts",
            "exit_time_field": "exit_ts",
            "label_available_time_field": "label_available_ts",
        },
        "fill_rules": {
            "equity_entry": "next_complete_bar_plus_adverse_slippage",
            "equity_exit": "first_complete_bar_at_or_after_exit_minus_adverse_slippage",
            "debit_spread_entry": "buy_ask_sell_bid",
            "debit_spread_exit": "sell_bid_buy_ask",
            "midpoint_fills_allowed": False,
        },
        "cost_model": {
            "equity_per_side_bps": 5.0,
            "option_per_leg_per_side_dollars": 0.05,
        },
        "metric": {
            "name": "lower_confidence_net_utility_per_eligible_slot",
            "utility_floor": -1.0,
            "utility_cap": 1.0,
            "stand_down_utility": 0.0,
            "missing_or_rejected_utility": 0.0,
            "lower_quantile": 0.10,
            "bootstrap_method": "session_block_resample",
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260810,
        },
        "governance": {
            "aggregate_score_hidden_during_pilot": True,
            "evaluator_changes_require_new_manifest": True,
            "all_variants_counted": True,
            "broker_access_allowed": False,
            "order_actions_allowed": False,
            "approval_policy": "not_applicable_paper_only",
        },
    }
    validate_manifest(manifest)
    return manifest


def _parse_sessions(value: str) -> list[date]:
    sessions = [
        date.fromisoformat(item.strip()) for item in value.split(",") if item.strip()
    ]
    if not sessions:
        raise argparse.ArgumentTypeError("at least one session date is required")
    if any(session.weekday() >= 5 for session in sessions):
        raise argparse.ArgumentTypeError("weekend dates cannot be locked as sessions")
    return sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sessions", required=True, type=_parse_sessions)
    parser.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_hash = source_bundle_sha256(evaluator_source_paths())
    manifest = build_manifest(
        run_id=args.run_id,
        sessions=args.sessions,
        universe=[item for item in args.universe.split(",") if item.strip()],
        locked_at=datetime.now(UTC).isoformat(),
        evaluator_source_sha256=source_hash,
    )
    args.output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "manifest_sha256": manifest_sha256(manifest),
                "evaluator_source_sha256": source_hash,
                "slot_count": len(manifest["slots"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
