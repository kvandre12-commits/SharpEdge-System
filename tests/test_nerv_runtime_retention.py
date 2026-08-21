from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.runtime_retention import prune_stale_files


def test_prune_stale_files_deletes_old_files_only(tmp_path: Path) -> None:
    old_file = tmp_path / "old.json"
    fresh_file = tmp_path / "fresh.json"
    old_file.write_text("old", encoding="utf-8")
    fresh_file.write_text("fresh", encoding="utf-8")
    now = 1_000_000.0
    os.utime(old_file, (now - 100_000, now - 100_000))
    os.utime(fresh_file, (now, now))

    deleted = prune_stale_files(tmp_path, max_age_hours=24, now=now)

    assert deleted == [old_file]
    assert not old_file.exists()
    assert fresh_file.exists()


def test_prune_stale_files_can_be_disabled(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("keep", encoding="utf-8")

    deleted = prune_stale_files(tmp_path, max_age_hours=0, now=1_000_000.0)

    assert deleted == []
    assert artifact.exists()
