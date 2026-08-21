from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from runtime_artifact_hygiene import (
    _candidate_files,
    _delete_candidates,
)


def test_candidate_files_only_allowlisted_runtime_paths(tmp_path: Path) -> None:
    old_nerv = tmp_path / "outputs" / "nerv" / "old.json"
    old_nerv.parent.mkdir(parents=True)
    old_nerv.write_text("old", encoding="utf-8")
    not_allowlisted = tmp_path / "outputs" / "keep.json"
    not_allowlisted.write_text("keep", encoding="utf-8")
    now = 1_000_000
    os.utime(old_nerv, (now - 100_000, now - 100_000))
    os.utime(not_allowlisted, (now - 100_000, now - 100_000))

    candidates = _candidate_files(tmp_path, set(), max_age_hours=1)

    assert [candidate.path for candidate in candidates] == ["outputs/nerv/old.json"]


def test_delete_candidates_skips_tracked_files_by_default(tmp_path: Path) -> None:
    runtime_file = tmp_path / "outputs" / "nerv" / "runtime.json"
    tracked_file = tmp_path / "outputs" / "nerv" / "tracked.json"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("runtime", encoding="utf-8")
    tracked_file.write_text("tracked", encoding="utf-8")

    candidates = _candidate_files(
        tmp_path,
        {"outputs/nerv/tracked.json"},
        max_age_hours=0,
    )
    deleted = _delete_candidates(tmp_path, candidates, include_tracked=False)

    assert [candidate.path for candidate in deleted] == ["outputs/nerv/runtime.json"]
    assert not runtime_file.exists()
    assert tracked_file.exists()
