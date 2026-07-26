"""Small TTL broom for disposable NERV runtime artifacts."""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_RETENTION_HOURS = 24.0


def prune_stale_files(
    directory: str | Path,
    *,
    max_age_hours: float = DEFAULT_RETENTION_HOURS,
    now: float | None = None,
) -> list[Path]:
    """Delete stale files under ``directory`` and return deleted paths.

    This is intentionally boring: it only deletes files, only under the provided
    directory, and treats non-positive max age as disabled. No recursive directory
    demolition goblinry.
    """

    root = Path(directory)
    if max_age_hours <= 0 or not root.exists():
        return []
    cutoff = (now if now is not None else time.time()) - (max_age_hours * 3600)
    deleted: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
            path.unlink()
        except OSError:
            continue
        deleted.append(path)
    _remove_empty_dirs(root)
    return deleted


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
