#!/usr/bin/env python3
"""Audit and prune allowlisted SharpEdge runtime artifacts.

Dry-run by default. This is intentionally conservative: it only considers known
runtime scratch paths and skips git-tracked files unless explicitly told not to.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_AGE_HOURS = 24.0
ALLOWLIST_DIRS = (
    "outputs/nerv",
    "outputs/nerv_trade_desk",
    "outputs/regime_cartridges",
    "outputs/runtime_tmp",
)
ALLOWLIST_GLOBS = (
    "outputs/nerv_*",
    "outputs/cockpit_regime_split_loop.*",
    "outputs/*.log",
    "outputs/*.pid",
)


@dataclass(frozen=True)
class ArtifactCandidate:
    path: str
    size_bytes: int
    age_hours: float
    tracked: bool
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repo root.")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help="Only prune candidates at least this old. Use 0 to match all candidates.",
    )
    parser.add_argument(
        "--largest", type=int, default=15, help="Largest files to report."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually delete candidates."
    )
    parser.add_argument(
        "--include-tracked",
        action="store_true",
        help="Allow deleting tracked files. Usually a terrible idea. Cute, but no.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine JSON only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    tracked = _tracked_files(root)
    candidates = _candidate_files(root, tracked, max_age_hours=args.max_age_hours)
    largest = _largest_outputs(root, tracked, limit=args.largest)
    deleted = (
        _delete_candidates(root, candidates, args.include_tracked) if args.apply else []
    )
    report = {
        "schema": "sharpedge.runtime_artifact_hygiene.v1",
        "root": str(root),
        "apply": args.apply,
        "max_age_hours": args.max_age_hours,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item.size_bytes for item in candidates),
        "deleted_count": len(deleted),
        "deleted_bytes": sum(item.size_bytes for item in deleted),
        "candidates": [asdict(item) for item in candidates],
        "largest_outputs": [asdict(item) for item in largest],
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def _candidate_files(
    root: Path,
    tracked: set[str],
    *,
    max_age_hours: float,
) -> list[ArtifactCandidate]:
    now = time.time()
    candidates: list[ArtifactCandidate] = []
    for path, reason in _allowlisted_paths(root):
        if path.is_dir():
            paths = [item for item in path.rglob("*") if item.is_file()]
        elif path.is_file():
            paths = [path]
        else:
            continue
        for file_path in paths:
            rel = _rel(root, file_path)
            stat = file_path.stat()
            age_hours = max((now - stat.st_mtime) / 3600, 0)
            if age_hours < max_age_hours:
                continue
            candidates.append(
                ArtifactCandidate(
                    path=rel,
                    size_bytes=stat.st_size,
                    age_hours=round(age_hours, 2),
                    tracked=rel in tracked,
                    reason=reason,
                )
            )
    return sorted(candidates, key=lambda item: item.size_bytes, reverse=True)


def _allowlisted_paths(root: Path) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    for rel in ALLOWLIST_DIRS:
        paths.append((root / rel, "allowlisted_runtime_dir"))
    for pattern in ALLOWLIST_GLOBS:
        for path in root.glob(pattern):
            paths.append((path, "allowlisted_runtime_glob"))
    deduped: dict[Path, str] = {}
    for path, reason in paths:
        deduped[path] = reason
    return sorted(deduped.items(), key=lambda item: str(item[0]))


def _delete_candidates(
    root: Path,
    candidates: list[ArtifactCandidate],
    include_tracked: bool,
) -> list[ArtifactCandidate]:
    deleted: list[ArtifactCandidate] = []
    for candidate in candidates:
        if candidate.tracked and not include_tracked:
            continue
        path = root / candidate.path
        try:
            path.unlink()
        except OSError:
            continue
        deleted.append(candidate)
    _remove_empty_allowlisted_dirs(root)
    return deleted


def _remove_empty_allowlisted_dirs(root: Path) -> None:
    for rel in ALLOWLIST_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for path in sorted((p for p in base.rglob("*") if p.is_dir()), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass


def _largest_outputs(
    root: Path, tracked: set[str], *, limit: int
) -> list[ArtifactCandidate]:
    outputs = root / "outputs"
    if not outputs.exists():
        return []
    now = time.time()
    items = []
    for path in outputs.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        items.append(
            ArtifactCandidate(
                path=_rel(root, path),
                size_bytes=stat.st_size,
                age_hours=round(max((now - stat.st_mtime) / 3600, 0), 2),
                tracked=_rel(root, path) in tracked,
                reason="largest_outputs_report_only",
            )
        )
    return sorted(items, key=lambda item: item.size_bytes, reverse=True)[:limit]


def _tracked_files(root: Path) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {item.decode() for item in proc.stdout.split(b"\0") if item}


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _print_report(report: dict[str, object]) -> None:
    print("SharpEdge runtime artifact hygiene")
    print(f"root: {report['root']}")
    print(f"apply: {report['apply']} max_age_hours: {report['max_age_hours']}")
    print(
        f"candidates: {report['candidate_count']} "
        f"({int(report['candidate_bytes']) / 1024 / 1024:.2f} MiB)"
    )
    print(
        f"deleted: {report['deleted_count']} "
        f"({int(report['deleted_bytes']) / 1024 / 1024:.2f} MiB)"
    )
    _print_items("\nCandidates", report["candidates"])
    _print_items("\nLargest outputs (report only)", report["largest_outputs"])


def _print_items(title: str, raw_items: object) -> None:
    items = list(raw_items or [])
    print(title)
    if not items:
        print("  none")
        return
    for item in items[:25]:
        print(
            "  {size:>9} bytes  age={age:>7}h  tracked={tracked:<5}  {path}".format(
                size=item["size_bytes"],
                age=item["age_hours"],
                tracked=str(item["tracked"]).lower(),
                path=item["path"],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
