from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agents.compounding_agent import summarize_candidates
from scripts.agents.recoverable_compounding_demo import frozen_candidates
from scripts.agents.recoverable_cycle_demo import (
    CRASH_EXIT_CODE,
    ContradictoryHistoryError,
    RecoveryLedger,
)
from scripts.alpha_swarm.contracts import payload_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agents" / "recoverable_compounding_demo.py"


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
        command.append("--crash-before-review")
    return command


def test_crash_reuses_evidence_and_publishes_exactly_one_review(tmp_path):
    state_dir = tmp_path / "state"
    output = tmp_path / "compounding-review.json"

    crashed = subprocess.run(
        _command(state_dir, output, crash=True),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert crashed.returncode == CRASH_EXIT_CODE
    assert "SIMULATED_CRASH=during_compounding_review_publication" in crashed.stderr
    assert not output.exists()
    with RecoveryLedger(state_dir / "cycle-ledger.sqlite3") as ledger:
        assert ledger.compounding_counts() == {"candidates": 3, "reviews": 0}

    resumed = subprocess.run(
        _command(state_dir, output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(resumed.stdout)
    assert result["inserted"] == {
        "candidates": [False, False, False],
        "review": True,
    }
    assert result["counts"] == {"candidates": 3, "reviews": 1}
    projection = output.read_bytes()

    replayed = subprocess.run(
        _command(state_dir, output),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert replayed.returncode == 0, replayed.stderr
    replay = json.loads(replayed.stdout)
    assert replay["inserted"] == {
        "candidates": [False, False, False],
        "review": False,
    }
    assert replay["counts"] == {"candidates": 3, "reviews": 1}
    assert output.read_bytes() == projection


def test_review_must_be_reproducible_from_stored_evidence(tmp_path):
    candidates = frozen_candidates()
    tampered = summarize_candidates(candidates)
    tampered["mean_observed_delta_pct"] = 999.0
    tampered.pop("receipt_id")
    tampered["receipt_id"] = payload_sha256(tampered)

    with RecoveryLedger(tmp_path / "ledger.sqlite3") as ledger:
        for candidate in candidates:
            assert ledger.record_compounding_candidate(candidate) is True
        with pytest.raises(ContradictoryHistoryError, match="not reproducible"):
            ledger.publish_compounding_review(tampered)
        assert ledger.compounding_counts() == {"candidates": 3, "reviews": 0}


def test_contradictory_candidate_history_fails_closed(tmp_path):
    original = frozen_candidates()[0]
    conflicting = dict(original)
    conflicting["observed_delta_pct"] = 999.0
    conflicting.pop("receipt_id")
    conflicting["receipt_id"] = payload_sha256(conflicting)

    with RecoveryLedger(tmp_path / "ledger.sqlite3") as ledger:
        assert ledger.record_compounding_candidate(original) is True
        with pytest.raises(ContradictoryHistoryError):
            ledger.record_compounding_candidate(conflicting)
        assert ledger.compounding_counts() == {"candidates": 1, "reviews": 0}
