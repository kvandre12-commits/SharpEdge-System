#!/usr/bin/env python3
"""Small stdlib-only Python quality gate for Termux and CI.

Ruff is better when available, but it can require a Rust build on Android/Termux.
This script keeps the project from having zero local quality gate when Ruff is not
practical. It intentionally checks boring high-signal things only.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_PATHS = ["scripts"]
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
]
MERGE_MARKERS = ("<<<<<<< ", "=======", ">>>>>>> ")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    code: str
    message: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"{location}: {self.code} {self.message}"


class ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.import_lines: dict[str, int] = {}
        self.used_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API name
        for alias in node.names:
            name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            self.import_lines.setdefault(name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API name
        if node.module == "__future__":
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            self.import_lines.setdefault(name, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API name
        self.used_names.add(node.id)


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_python_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".py" and not is_excluded(path):
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*.py")
                if child.is_file() and not is_excluded(child)
            )
    return sorted(set(files))


def check_text(
    path: Path,
    max_lines: int,
    max_line_length: int,
    strict_style: bool,
) -> list[Issue]:
    issues: list[Issue] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if len(lines) > max_lines:
        issues.append(
            Issue(
                path,
                0,
                "Q001",
                f"file has {len(lines)} lines; split before it becomes soup (max {max_lines})",
            )
        )

    if text and not text.endswith("\n"):
        issues.append(Issue(path, len(lines), "Q002", "missing trailing newline"))

    for line_no, line in enumerate(lines, start=1):
        if strict_style and line.rstrip(" \t") != line:
            issues.append(Issue(path, line_no, "Q003", "trailing whitespace"))
        if "\t" in line[: len(line) - len(line.lstrip())]:
            issues.append(Issue(path, line_no, "Q004", "tab indentation"))
        if strict_style and len(line) > max_line_length:
            issues.append(
                Issue(
                    path,
                    line_no,
                    "Q005",
                    f"line too long ({len(line)} > {max_line_length})",
                )
            )
        if line.startswith(MERGE_MARKERS):
            issues.append(Issue(path, line_no, "Q006", "merge conflict marker"))
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            issues.append(Issue(path, line_no, "Q007", "possible secret literal"))

    return issues


def check_ast(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Issue(path, exc.lineno or 0, "Q100", exc.msg)]

    visitor = ImportVisitor()
    visitor.visit(tree)
    for name, line_no in sorted(visitor.import_lines.items(), key=lambda item: item[1]):
        if name.startswith("_"):
            continue
        if name not in visitor.used_names:
            issues.append(Issue(path, line_no, "Q101", f"possibly unused import '{name}'"))

    return issues


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stdlib-only Python quality checks.")
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--max-lines", type=int, default=600)
    parser.add_argument("--max-line-length", type=int, default=160)
    parser.add_argument(
        "--strict-style",
        action="store_true",
        help="Also fail on trailing whitespace, long lines, and likely unused imports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    files = iter_python_files(args.paths)
    if not files:
        print("python_quality: no Python files found")
        return 0

    issues: list[Issue] = []
    for path in files:
        file_issues = check_text(
            path,
            max_lines=args.max_lines,
            max_line_length=args.max_line_length,
            strict_style=args.strict_style,
        )
        if args.strict_style:
            file_issues.extend(check_ast(path))
        else:
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            except SyntaxError as exc:
                file_issues.append(Issue(path, exc.lineno or 0, "Q100", exc.msg))
        issues.extend(file_issues)

    if issues:
        print(f"python_quality: found {len(issues)} issue(s) across {len(files)} file(s)")
        for issue in issues:
            print(issue.render())
        return 1

    print(f"python_quality: OK ({len(files)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
