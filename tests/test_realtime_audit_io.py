from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import realtime_audit_io as io


def test_parse_timestamp_handles_z_and_naive():
    assert io.parse_timestamp("2026-08-22T15:00:00Z").tzinfo == timezone.utc
    assert io.parse_timestamp("2026-08-22T15:00:00").tzinfo == timezone.utc  # naive -> utc
    assert io.parse_timestamp("garbage") is None
    assert io.parse_timestamp(None) is None


def test_ledger_round_trip_and_skips_corrupt(tmp_path):
    path = tmp_path / "led.jsonl"
    io.write_ledger(path, [{"ts": "b", "v": 2}, {"ts": "a", "v": 1}])
    # append a corrupt line
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n\n")
    rows = io.read_ledger(path)
    assert [r["ts"] for r in rows] == ["b", "a"]  # corrupt/blank skipped


def test_cache_snapshot_paths_finds_dated_dirs_excludes_latest(tmp_path):
    for name in ("20260822T150000Z", "20260822T150500Z", "latest"):
        d = tmp_path / name / "outputs"
        d.mkdir(parents=True)
        (d / "signal.json").write_text("{}")
    (tmp_path / "not_a_dir.txt").write_text("x")
    paths = io.cache_snapshot_paths(tmp_path)
    names = [p.parent.parent.name for p in paths]
    assert names == ["20260822T150000Z", "20260822T150500Z"]  # sorted, no 'latest'


def test_read_json_tolerates_missing(tmp_path):
    assert io.read_json(tmp_path / "nope.json") is None
