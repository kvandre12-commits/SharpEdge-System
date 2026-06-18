# Operator Review + Morning Dashboard

## Session review

```bash
python scripts/agents/operator_session_review.py
```

Outputs:

- `outputs/operator_session_review.json`
- `outputs/operator_session_review.txt`

This summarizes recent `operator_journal_append.jsonl` activity into:

- latest setup state
- action/status distributions
- top recurring blockers
- top recurring risk flags
- current watchlist activity count

## Morning open dashboard

```bash
python scripts/agents/morning_open_dashboard.py
```

Outputs:

- `outputs/morning_open_dashboard.json`
- `outputs/morning_open_dashboard.txt`

This is a compact open-check artifact built from the brief, watchlist, contract,
and session review. It highlights:

- readiness: `review` / `monitor` / `blocked`
- key focus levels
- permissions and blockers
- opening checklist items
- recent review context
- optional historical hints from `trade_journal_hints.json`
- canonical agent-language mapping via `workflow_state`, `execution_plan`, `approval_decision`, and `journal`
