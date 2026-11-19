#!/usr/bin/env python3
"""Crash-and-resume wrapper for the deterministic protective cycle demo."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.deterministic_cycle_demo import run_cycle, write_atomic
from scripts.alpha_swarm.contracts import canonical_json, payload_sha256

CRASH_EXIT_CODE = 86


class ContradictoryHistoryError(RuntimeError):
    """Raised when a stable history identity is presented with different content."""


class RecoveryLedger:
    """Small SQLite ledger with idempotent, immutable cycle records."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS component_records (
                cycle_id TEXT NOT NULL,
                component TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (cycle_id, component)
            );
            CREATE TABLE IF NOT EXISTS protective_intents (
                cycle_id TEXT NOT NULL,
                action_key TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (cycle_id, action_key)
            );
            CREATE TABLE IF NOT EXISTS cycle_commits (
                cycle_id TEXT PRIMARY KEY,
                receipt_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RecoveryLedger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _record_once(
        self,
        *,
        table: str,
        cycle_id: str,
        identity_column: str,
        identity: str,
        payload: dict[str, Any],
    ) -> bool:
        if table not in {"component_records", "protective_intents"}:
            raise ValueError("unsupported immutable record table")
        if identity_column not in {"component", "action_key"}:
            raise ValueError("unsupported immutable identity column")
        receipt_id = str(payload.get("receipt_id") or "")
        if not receipt_id:
            raise ValueError("immutable record requires receipt_id")
        payload_json = canonical_json(payload)
        with self.connection:
            row = self.connection.execute(
                f"SELECT receipt_id, payload_json FROM {table} "
                f"WHERE cycle_id = ? AND {identity_column} = ?",
                (cycle_id, identity),
            ).fetchone()
            if row is not None:
                if row != (receipt_id, payload_json):
                    raise ContradictoryHistoryError(
                        f"conflicting {table} history for {cycle_id}/{identity}"
                    )
                return False
            self.connection.execute(
                f"INSERT INTO {table} "
                f"(cycle_id, {identity_column}, receipt_id, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (cycle_id, identity, receipt_id, payload_json),
            )
        return True

    def record_component(self, cycle_id: str, payload: dict[str, Any]) -> bool:
        component = str(payload.get("component") or "")
        if not component:
            raise ValueError("component record requires component")
        return self._record_once(
            table="component_records",
            cycle_id=cycle_id,
            identity_column="component",
            identity=component,
            payload=payload,
        )

    def record_protective_intent(
        self, cycle_id: str, payload: dict[str, Any]
    ) -> bool:
        action_key = str(payload.get("action_key") or "")
        if not action_key:
            raise ValueError("protective intent requires action_key")
        return self._record_once(
            table="protective_intents",
            cycle_id=cycle_id,
            identity_column="action_key",
            identity=action_key,
            payload=payload,
        )

    def publish(
        self,
        cycle: dict[str, Any],
        *,
        crash_before_commit: bool = False,
    ) -> bool:
        """Atomically publish once after verifying staged component history."""
        cycle_id = str(cycle["cycle_id"])
        expected = {
            name: (payload["receipt_id"], canonical_json(payload))
            for name, payload in cycle["components"].items()
        }
        payload_json = canonical_json(cycle)
        receipt_id = str(cycle["receipt_id"])

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self.connection.execute(
                "SELECT component, receipt_id, payload_json "
                "FROM component_records WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchall()
            actual = {name: (stored_receipt, stored_json) for name, stored_receipt, stored_json in rows}
            if actual != expected:
                raise ContradictoryHistoryError(
                    f"staged components do not match canonical cycle {cycle_id}"
                )
            if crash_before_commit:
                print(
                    "SIMULATED_CRASH=during_cycle_receipt_publication",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(CRASH_EXIT_CODE)

            row = self.connection.execute(
                "SELECT receipt_id, payload_json FROM cycle_commits WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
            if row is not None:
                if row != (receipt_id, payload_json):
                    raise ContradictoryHistoryError(
                        f"conflicting canonical commit for {cycle_id}"
                    )
                self.connection.commit()
                return False
            self.connection.execute(
                "INSERT INTO cycle_commits (cycle_id, receipt_id, payload_json) "
                "VALUES (?, ?, ?)",
                (cycle_id, receipt_id, payload_json),
            )
            self.connection.commit()
            return True
        except BaseException:
            self.connection.rollback()
            raise

    def counts(self, cycle_id: str) -> dict[str, int]:
        return {
            name: int(
                self.connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE cycle_id = ?", (cycle_id,)
                ).fetchone()[0]
            )
            for name, table in {
                "components": "component_records",
                "protective_intents": "protective_intents",
                "commits": "cycle_commits",
            }.items()
        }


def _protective_intent(cycle: dict[str, Any]) -> dict[str, Any]:
    steward = cycle["components"]["position_steward"]
    payload = {
        "schema": "sharpedge.protective_intent.v1",
        "cycle_id": cycle["cycle_id"],
        "action_key": f"position_steward:exit_review:{steward['position_ref']}",
        "action": "owner_exit_review",
        "effect": "review_only",
        "source_receipt_id": steward["receipt_id"],
        "broker_action_allowed": False,
    }
    payload["receipt_id"] = payload_sha256(payload)
    return payload


def run_recoverable(
    *,
    state_dir: Path,
    output: Path,
    crash_during_publication: bool = False,
) -> dict[str, Any]:
    """Stage immutable records, atomically publish, and rebuild the projection."""
    cycle = run_cycle()
    cycle_id = str(cycle["cycle_id"])
    inserted: dict[str, bool] = {}
    with RecoveryLedger(state_dir / "cycle-ledger.sqlite3") as ledger:
        for name in ("observer", "helios", "position_steward"):
            inserted[name] = ledger.record_component(cycle_id, cycle["components"][name])
        inserted["protective_intent"] = ledger.record_protective_intent(
            cycle_id, _protective_intent(cycle)
        )
        inserted["commit"] = ledger.publish(
            cycle, crash_before_commit=crash_during_publication
        )
        counts = ledger.counts(cycle_id)

    write_atomic(output, cycle)
    return {
        "cycle_id": cycle_id,
        "receipt_id": cycle["receipt_id"],
        "inserted": inserted,
        "counts": counts,
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crash-after-steward", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_recoverable(
        state_dir=args.state_dir,
        output=args.output,
        crash_during_publication=args.crash_after_steward,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
