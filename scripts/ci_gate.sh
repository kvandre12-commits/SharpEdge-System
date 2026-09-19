#!/usr/bin/env sh
# Single source of truth for "can this safely land on main?"
#
# Called by BOTH the local pre-push hook (.githooks/pre-push) and CI
# (.github/workflows/ci.yml) so "green locally" provably means "green in CI" --
# one gate, no drift. Fail closed: any step non-zero aborts the whole gate.
#
# Kept deliberately honest: no step is allowed to swallow its own failure with
# `|| true`. If you need to loosen a check, change it here, in the open.
set -eu

echo ">> byte-compile source (excludes outputs/ rescue scratch)"
python -m compileall -q -x '(^|/)(outputs|\.git|venv|__pycache__|node_modules)(/|$)' .

echo ">> lint: crash-class rules only (syntax, undefined names, bad f-strings/format)"
# The full default ruff rule set currently reports ~249 style findings in this
# repo; gating on all of them today would just get routed around. Gate on the
# bug-class rules now (these pass clean) and tighten later via `ruff check --fix`
# once the style backlog is burned down. outputs/ is rescued scratch, not source.
ruff check . --exclude outputs --select E9,F63,F7,F82

echo ">> tests: real suite (outputs/ has rescued files that break collection)"
python -m pytest tests/ -q -p no:cacheprovider --ignore=outputs

echo ">> gate PASSED"
