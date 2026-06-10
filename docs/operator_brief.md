# Operator Brief MVP

`python scripts/agents/operator_brief.py`

This MVP emits a single operator-facing summary from the existing SharpEdge
artifacts. It does not create new trade logic. It compresses the current state
into one brief you can read fast.

## Outputs

- `outputs/operator_brief.json`
- `outputs/operator_brief.txt`
- `outputs/operator_watchlist.json`
- `outputs/operator_journal_append.jsonl`

## Inputs

- `outputs/agent_controller_decision.json`
- `outputs/robinhood_fvg_monitor.json`
- `outputs/agent_v1_decision.json`
- `outputs/health/warnings.log` when present

## What it answers

- Should I stand down, monitor, or review a trade plan?
- What gap direction / fill level matters right now?
- Which option side is the current thesis watching?
- What blockers or stale inputs are stopping action?
- Is broker integration live, or is this manual-review only?

## Safety posture

The brief is operator-facing only.

- It never enables auto-trading.
- It never overrides the dry-run safety contract.
- It preserves the existing order block / operator-confirmation posture.

This is a compression layer, not a permission layer.

## Extra MVP artifacts

### Watchlist

`operator_watchlist.json` holds the current setup snapshot with:

- status: `ready_for_review` / `monitor_only` / `blocked`
- priority
- key levels and thesis
- blocker and risk-flag context

### Journal append

`operator_journal_append.jsonl` appends one structured line per distinct setup state.
Repeated runs with the same source-state fingerprint do not duplicate the latest
entry, so the pipeline stays append-only without mindless spam.

## Follow-on artifacts

### Session review

`python scripts/agents/operator_session_review.py`

Writes:

- `outputs/operator_session_review.json`
- `outputs/operator_session_review.txt`

### Morning open dashboard

`python scripts/agents/morning_open_dashboard.py`

Writes:

- `outputs/morning_open_dashboard.json`
- `outputs/morning_open_dashboard.txt`
