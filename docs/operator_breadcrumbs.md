# Operator Breadcrumbs

SharpEdge should leave a clear reconstruction trail for operator-facing decisions.
If the system says `stand_down`, `monitor_only`, or `review_trade_plan`, an
operator should be able to tell **why** without reverse-engineering the codebase.

## Rules

- Prefer explicit artifacts over implied state.
- Preserve blockers, stale-input signals, and risk flags in downstream outputs.
- Keep append-only history where it helps review (`operator_journal_append.jsonl`).
- Do not turn breadcrumb artifacts into permission escalation.
- Favor short, local, deterministic outputs over hidden agent memory.

## Current breadcrumb chain

1. `outputs/health/*_state.json` and `outputs/health/warnings.log`
2. `outputs/robinhood_fvg_monitor.json`
3. `outputs/agent_v1_decision.json`
4. `outputs/operator_brief.json`
5. `outputs/operator_watchlist.json`
6. `outputs/operator_journal_append.jsonl`
7. `outputs/operator_session_review.json`
8. `outputs/morning_open_dashboard.json`
9. `outputs/robinhood_beta_execution.json`

## What good looks like

A later review should be able to answer:

- what setup was active
- what blockers existed
- whether broker integration was live
- whether the decision was review / monitor / blocked
- whether beta execution was artifact-only, monitoring-only, or approval-queue ready
- what changed across runs

Breadcrumbs are for traceability, not for auto-trading authority.
