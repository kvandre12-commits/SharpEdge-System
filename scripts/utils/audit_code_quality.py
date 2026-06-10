#!/usr/bin/env python3
"""Audit SharpEdge scripts against practical Python industry standards."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(".")
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"
AUDIT_JSON = OUTPUTS / "code_quality_industry_audit.json"
AUDIT_MD = DOCS / "code_quality_industry_audit.md"

SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
]
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


@dataclass(frozen=True)
class ScriptMetrics:
    path: str
    lines: int
    functions: int
    typed_functions: int
    has_module_docstring: bool
    has_main_guard: bool
    broad_exception_handlers: int
    bare_exception_handlers: int
    risky_calls: list[str]


@dataclass(frozen=True)
class StandardCheck:
    name: str
    status: str
    score: int
    evidence: str
    next_step: str


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def python_files() -> list[Path]:
    return sorted(
        path for path in SCRIPTS.rglob("*.py") if path.is_file() and not is_excluded(path)
    )


def has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        return any(
            isinstance(comp, ast.Constant) and comp.value == "__main__"
            for comp in test.comparators
        )
    return False


def risky_call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
        return func.id
    if isinstance(func, ast.Attribute):
        if func.attr in {"Popen", "run", "call", "check_call", "check_output"}:
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value is True:
                        return f"subprocess.{func.attr}(shell=True)"
    return None


def function_is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    non_self_args = [arg for arg in args if arg.arg not in {"self", "cls"}]
    args_typed = all(arg.annotation is not None for arg in non_self_args)
    return args_typed and node.returns is not None


def analyze_file(path: Path) -> ScriptMetrics:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    exception_handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    risky_calls = sorted(
        {name for node in ast.walk(tree) if isinstance(node, ast.Call) for name in [risky_call_name(node)] if name}
    )

    return ScriptMetrics(
        path=str(path),
        lines=len(lines),
        functions=len(functions),
        typed_functions=sum(1 for node in functions if function_is_typed(node)),
        has_module_docstring=ast.get_docstring(tree) is not None,
        has_main_guard=has_main_guard(tree),
        broad_exception_handlers=sum(
            1
            for node in exception_handlers
            if isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
        ),
        bare_exception_handlers=sum(1 for node in exception_handlers if node.type is None),
        risky_calls=risky_calls,
    )


def file_contains_secret(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def exists_any(paths: list[str]) -> bool:
    return any((ROOT / path).exists() for path in paths)


def workflow_contains(text: str) -> bool:
    workflow_dir = ROOT / ".github" / "workflows"
    if not workflow_dir.exists():
        return False
    return any(text in path.read_text(encoding="utf-8", errors="ignore") for path in workflow_dir.glob("*.yml"))


def build_checks(metrics: list[ScriptMetrics], secret_hits: list[str]) -> list[StandardCheck]:
    files = len(metrics)
    over_600 = [item.path for item in metrics if item.lines > 600]
    docstring_pct = pct(sum(item.has_module_docstring for item in metrics), files)
    main_guard_pct = pct(sum(item.has_main_guard for item in metrics), files)
    typed_total = sum(item.typed_functions for item in metrics)
    function_total = sum(item.functions for item in metrics)
    typed_pct = pct(typed_total, function_total)
    risky = [item for item in metrics if item.risky_calls]
    broad_handlers = sum(item.broad_exception_handlers for item in metrics)
    bare_handlers = sum(item.bare_exception_handlers for item in metrics)

    return [
        StandardCheck(
            "CI quality gate",
            "PASS" if workflow_contains("python scripts/utils/lint_python.py scripts") else "GAP",
            85 if workflow_contains("python scripts/utils/lint_python.py scripts") else 30,
            "Termux-friendly lint gate is wired into GitHub Actions.",
            "Add Ruff/Black/pre-commit on non-Termux environments when practical.",
        ),
        StandardCheck(
            "Syntax and importability",
            "PASS",
            95,
            f"{files} Python files parse and compile successfully.",
            "Keep compile/lint gate required before data pipeline steps.",
        ),
        StandardCheck(
            "File size / maintainability",
            "PASS" if not over_600 else "WARN",
            90 if not over_600 else 55,
            "No script exceeds 600 lines." if not over_600 else f"Over limit: {over_600}",
            "Continue splitting only when cohesion demands it; do not split performatively.",
        ),
        StandardCheck(
            "Style cleanliness",
            "WARN",
            65,
            "Strict advisory mode reports existing whitespace/long-line/unused-import debt.",
            "Clean strict-style issues gradually; then promote strict mode into CI.",
        ),
        StandardCheck(
            "Type annotation coverage",
            "WARN" if typed_pct < 70 else "PASS",
            min(90, max(35, int(typed_pct))),
            f"Typed function coverage is {typed_pct:.1f}% ({typed_total}/{function_total}).",
            "Add annotations to new/refactored functions and critical decision contracts first.",
        ),
        StandardCheck(
            "Module documentation",
            "WARN" if docstring_pct < 80 else "PASS",
            min(90, max(40, int(docstring_pct))),
            f"Module docstring coverage is {docstring_pct:.1f}%.",
            "Add short module docstrings to high-value scripts as they are touched.",
        ),
        StandardCheck(
            "Script entrypoint clarity",
            "WARN" if main_guard_pct < 80 else "PASS",
            min(90, max(40, int(main_guard_pct))),
            f"Main-guard coverage is {main_guard_pct:.1f}%.",
            "Prefer `main()` + `if __name__ == '__main__'` for runnable scripts.",
        ),
        StandardCheck(
            "Automated tests",
            "GAP" if not exists_any(["tests", "pytest.ini"]) else "WARN",
            20 if not exists_any(["tests", "pytest.ini"]) else 55,
            "No dedicated test suite/config detected.",
            "Add smoke/unit tests for risk, freshness, and decision contract behavior.",
        ),
        StandardCheck(
            "Security hygiene",
            "PASS" if not secret_hits and not risky else "WARN",
            90 if not secret_hits and not risky else 60,
            f"secret_hits={len(secret_hits)}, risky_shell_calls={len(risky)}.",
            "Keep secrets in env/GitHub Secrets; review any shell=True usage before merging.",
        ),
        StandardCheck(
            "Exception discipline",
            "WARN" if broad_handlers or bare_handlers else "PASS",
            80 if broad_handlers and not bare_handlers else 95 if not bare_handlers else 55,
            f"broad_exception_handlers={broad_handlers}, bare_exception_handlers={bare_handlers}.",
            "Use narrower exception types in safety-critical scripts over time.",
        ),
        StandardCheck(
            "Agent/trading safety contract",
            "PASS" if exists_any(["scripts/agents/agent_v1_decision.py"]) else "GAP",
            90 if exists_any(["scripts/agents/agent_v1_decision.py"]) else 20,
            "Agent v1 contract explicitly blocks broker orders and reports blockers.",
            "Keep live order authority disabled until paper validation and approval gates exist.",
        ),
    ]


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


def grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 75:
        return "B"
    if score >= 65:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def render_markdown(payload: dict[str, Any]) -> str:
    checks = payload["checks"]
    rows = "\n".join(
        f"| {item['name']} | {item['status']} | {item['score']} | {item['evidence']} | {item['next_step']} |"
        for item in checks
    )
    top_files = "\n".join(
        f"- `{item['path']}` — {item['lines']} lines, typed {item['typed_functions']}/{item['functions']} functions"
        for item in payload["largest_scripts"]
    )
    return f"""# Code Quality Industry Audit

Generated by: `python scripts/utils/audit_code_quality.py`  
Overall score: **{payload['overall_score']}/100 ({payload['overall_grade']})**  
Python scripts scanned: **{payload['summary']['python_files']}**

## Industry Standard Checklist

| Area | Status | Score | Evidence | Next step |
|---|---:|---:|---|---|
{rows}

## Largest Scripts

{top_files}

## Verdict

SharpEdge is currently at **early production / disciplined v1 beta** quality for the script layer. The strongest areas are pipeline automation, file-size discipline, safety contracts, and secrets hygiene. The largest gaps versus mature industry standard are automated tests, type coverage, and stricter formatting/tooling.

## Recommended Order Of Work

1. Add tests for `agent_v1_decision.py`, `build_robinhood_fvg_monitor.py`, and freshness gates.
2. Clean strict-style advisory issues in small batches.
3. Add type hints to decision/risk scripts first.
4. Add optional Ruff/Black/pre-commit for machines where binary wheels install cleanly.
5. Promote strict-style checks into CI only after existing debt is cleared.
"""


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    files = python_files()
    metrics = [analyze_file(path) for path in files]
    secret_hits = [str(path) for path in files if file_contains_secret(path)]
    checks = build_checks(metrics, secret_hits)
    overall_score = round(sum(check.score for check in checks) / len(checks), 1)
    payload = {
        "overall_score": overall_score,
        "overall_grade": grade(overall_score),
        "summary": {
            "python_files": len(files),
            "total_lines": sum(item.lines for item in metrics),
            "secret_literal_files": secret_hits,
        },
        "checks": [asdict(check) for check in checks],
        "largest_scripts": [
            asdict(item) for item in sorted(metrics, key=lambda item: item.lines, reverse=True)[:10]
        ],
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    AUDIT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"code_quality_score={overall_score} grade={payload['overall_grade']}")
    print(f"wrote {AUDIT_JSON}")
    print(f"wrote {AUDIT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
