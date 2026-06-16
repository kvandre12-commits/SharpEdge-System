---
name: sharpedge-operator-agents
description: SharpEdge's operator-facing agent layer (scripts/agents/) — the safety contract, deterministic publication controller, the four canonical agent-language objects (workflow_state / execution_plan / approval_decision / journal), operator brief, morning dashboard, session review, beta execution handoff, and journal hints. Use when working on operator artifacts, the approval/authority model, or the publication pipeline.
---

# SharpEdge Operator Agents Layer

Validated against `SharpEdge-System/scripts/agents/` (+ tests in `tests/`).
**None of these execute trades.** They normalize pipeline state into honest,
operator-facing artifacts with one explicit authority object.

## The authority rule (memorize this)
From `agent_language_objects.py`, four canonical objects each own one role:

| Object | Role | File |
|--------|------|------|
| `workflow_state` | what is true **right now** | `workflow_state.json` |
| `execution_plan` | what should happen **next if conditions allow** | `execution_plan.json` |
| `approval_decision` | what is **actually authorized** | `approval_decision.json` |
| `journal` | what **happened + was learned** | `journal.json` |

> **`approval_decision` is the ONLY authoritative permission object.** Everything
> else is description, intent, or history. Never treat workflow_state or
> execution_plan as permission.

## The safety contract — `agent_v1_decision.py` → `agent_v1_decision.json`
"Not a trade executor. A safety-oriented contract builder." Normalizes final
pipeline state into explicit **permissions, blockers, freshness/staleness checks,
artifact references**, and a `trade_edge_confidence`. Reads `spy_truth.db`
(`SPY_DB_PATH`), the monitor, and health warnings. `trade_journal_hints` must
**never override** this contract.

## Deterministic publication — `controller_agent.py` → `agent_controller_decision.json`
Local-only (replaced an old Gemini call). Reads DB + `latest_signal_strength.csv`
+ `robinhood_fvg_monitor.json` + `health/warnings.log`, applies conservative rules.
`MIN_CONTROLLER_CONFIDENCE = 0.55` (env `CONTROLLER_MIN_CONFIDENCE`).

## Operator artifacts (producers → outputs)
- `operator_brief.py` → `operator_brief.json/.txt`, `operator_watchlist.json`,
  `operator_journal_append.jsonl` (append-only journal line).
- `operator_session_review.py` → `operator_session_review.json/.txt` — Counter
  summary of the append-only journal.
- `morning_open_dashboard.py` → `morning_open_dashboard.json/.txt` — a checklist
  over brief/watchlist/contract/session_review/workflow/execution_plan/approval/
  journal/beta, via `agent_language_views` resolvers.
- `robinhood_beta_execution.py` → `robinhood_beta_execution.json/.txt` — an
  approval-gated **shadow-order** broker handoff: read/monitor perms, order-draft
  intent, explicit operator approval gates, artifact-only fallback. **No live
  order authority.**
- `trade_journal_hints.py` → `trade_journal_hints.json` — non-authoritative
  historical hints from the tiny manual journal (context only).

## Consumer helpers — `agent_language_views.py`
`resolve_workflow_state / resolve_approval_decision / resolve_execution_plan /
resolve_journal` read canonical objects **with legacy fallback**;
`derive_readiness(operator_action)` maps an action string to a readiness label.
Always consume via these resolvers so old artifacts still work.

## Publication order (rough DAG)
pipeline (DB + signal_strength + `robinhood_fvg_monitor.json`)
→ `controller_agent` → `agent_v1_decision` (safety contract)
→ `agent_language_objects` (the 4 objects)
→ `operator_brief` → `operator_session_review` → `robinhood_beta_execution`
→ `morning_open_dashboard`; `trade_journal_hints` feeds context alongside.

## Validate
`cd ~/SharpEdge-System && python3 -m pytest tests/test_operator_brief.py
tests/test_morning_open_dashboard.py tests/test_operator_session_review.py
tests/test_robinhood_beta_execution.py tests/test_agent_v1_decision.py -q`

## Guardrails
- Only `approval_decision` authorizes. Hints/briefs/plans never grant permission.
- These artifacts describe state; the live order still routes through the Bridge's
  `operator_confirm_required` gate (see `sharpedge-execution-governance`).
- Respect freshness: honor the staleness flags in `agent_v1_decision.json`.
