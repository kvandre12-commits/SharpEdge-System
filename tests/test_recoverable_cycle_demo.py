from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agents.deterministic_cycle_demo import run_cycle
from scripts.agents.recoverable_cycle_demo import (
    CRASH_EXIT_CODE,
    ContradictoryHistoryError,
    RecoveryLedger,
)
from scripts.alpha_swarm.contracts import payload_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agents" / "recoverable_cycle_demo.py"


def _command(state_dir: Path, output: Path, *, crash: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--state-dir",
        str(state_dir),
        "--output",
        str(output),
    ]
    if crash:
        command.append("--crash-after-steward")
    return command


def test_process_crash_resumes_without_duplicate_or_contradictory_history(tmp_path):
    state_dir = tmp_path / "state"
    output = tmp_path / "canonical-cycle.json"
    cycle = run_cycle()
    cycle_id = cycle["cycle_id"]

    crashed = subprocess.run(
        _command(state_dir, output, crash=True),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert crashed.returncode == CRASH_EXIT_CODE
    assert "SIMULATED_CRASH=during_cycle_receipt_publication" in crashed.stderr
    assert not output.exists()
    with RecoveryLedger(state_dir / "cycle-ledger.sqlite3") as ledger:
        assert ledger.counts(cycle_id) == {
            "components": 3,
            "protective_intents": 1,
            "commits": 0,
        }

    resumed = subprocess.run(
        _command(state_dir, output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(resumed.stdout)
    assert summary["inserted"] == {
        "observer": False,
        "helios": False,
        "position_steward": False,
        "protective_intent": False,
        "commit": True,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == cycle
    first_bytes = output.read_bytes()

    repeated = subprocess.run(
        _command(state_dir, output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert repeated.returncode == 0, repeated.stderr
    repeated_summary = json.loads(repeated.stdout)
    assert all(inserted is False for inserted in repeated_summary["inserted"].values())
    assert output.read_bytes() == first_bytes
    with RecoveryLedger(state_dir / "cycle-ledger.sqlite3") as ledger:
        assert ledger.counts(cycle_id) == {
            "components": 3,
            "protective_intents": 1,
            "commits": 1,
        }


def test_conflicting_component_history_fails_closed(tmp_path):
    cycle = run_cycle()
    cycle_id = cycle["cycle_id"]
    original = cycle["components"]["position_steward"]
    conflicting = {key: value for key, value in original.items() if key != "receipt_id"}
    conflicting["decision"] = "hold"
    conflicting["receipt_id"] = payload_sha256(conflicting)

    with RecoveryLedger(tmp_path / "ledger.sqlite3") as ledger:
        assert ledger.record_component(cycle_id, original) is True
        with pytest.raises(ContradictoryHistoryError):
            ledger.record_component(cycle_id, conflicting)
        assert ledger.counts(cycle_id)["components"] == 1
