"""Shared IO primitives for the realtime self-audit loops.

Extracted so the confluence-zone auditor reuses the same snapshot/ledger
plumbing the spine auditor established, without duplicating it. Pure filesystem +
parsing helpers; no scoring, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 stamp (``Z`` tolerated); naive stamps assumed UTC."""
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_json(path: Path | str) -> Any | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def cache_snapshot_paths(cache_dir: Path) -> list[Path]:
    """Timestamped ``<TS>/outputs/signal.json`` snapshots, name-sorted (oldest-first)."""
    if not cache_dir.exists():
        return []
    paths: list[Path] = []
    for child in sorted(cache_dir.iterdir()):
        if not child.is_dir() or child.name == "latest":
            continue
        signal_path = child / "outputs" / "signal.json"
        if signal_path.exists():
            paths.append(signal_path)
    return paths


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL ledger, skipping blank/corrupt lines."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL (deterministic key order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "cache_snapshot_paths",
    "parse_timestamp",
    "read_json",
    "read_ledger",
    "utc_now_iso",
    "write_ledger",
]
