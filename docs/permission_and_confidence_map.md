# Permission and Confidence Map

This repo has several layers that talk about **readiness**, **confidence**, or
**permission**. They are not interchangeable. If the app mixes them up, it will
sound confident while still being correctly blocked. Cute for comedy. Bad for
logic.

## Source-of-truth stack

| Layer | File | What it decides | What it must **not** decide |
|---|---|---|---|
| Controller | `scripts/agents/controller_agent.py` | Whether the local artifact stack looks complete enough to `post` vs `hold` | Live trade permission |
| Safety contract | `scripts/agents/agent_v1_decision.py` | `trade_allowed`, blockers, risk flags, required human action, permission booleans | Autonomous live order placement |
| Operator brief | `scripts/agents/operator_brief.py` | Human-facing compression of current state | Escalating permissions |
| Dashboard | `scripts/agents/morning_open_dashboard.py` | Open-check visibility and checklist status | Recomputing authority |
| Beta execution handoff | `scripts/agents/robinhood_beta_execution.py` | Approval-gated draft/execution packaging | Bypassing contract gates |
| Historical hints | `scripts/agents/trade_journal_hints.py` | Low-authority historical context from manual logs | Overriding blockers or permissions |

## Confidence meanings

| Field | File | Meaning | Safe interpretation |
|---|---|---|---|
| `controller.confidence` | `controller_agent.py` | Artifact completeness / internal consistency | "Pipeline looks assembled" |
| `confidence_evidence_quality` | `agent_v1_decision.py` | How solid the evidence inputs appear | "Inputs are coherent" |
| `confidence_trade_edge` | `agent_v1_decision.py` | Edge confidence derived from deployment confidence when watch conditions qualify | "Trade edge may exist" |
| `deployment_confidence` | risk layer / DB | Maturity of the researched edge | Upstream trade-context confidence |
| `confidence_matrix` scores | `build_confidence_weights.py` outputs | Research analytics and state ranking | Research-side weighting only |

## Permission meanings

Only one file is allowed to speak authoritatively about trade permission:

- **Authoritative:** `scripts/agents/agent_v1_decision.py`
- **Derived display only:** brief, dashboard, beta execution, historical hints

Key contract fields:

- `trade_allowed`
- `broker_order_allowed`
- `required_human_action`
- `permissions.*`
- `blocking_reasons`
- `risk_flags`

### Important rule

If any downstream layer disagrees with the contract, the contract wins.
Always.

## App-logic consumption order

Canonical agent-language order:

1. Read `outputs/approval_decision.json`
2. If blocked, show blockers and stop escalation
3. Read `outputs/workflow_state.json` for current truth snapshot
4. Read `outputs/execution_plan.json` for suggested next steps
5. Read `outputs/journal.json` for historical context

Legacy artifact order still maps underneath that protocol:

1. Read `outputs/agent_v1_decision.json`
2. If blocked, show blockers and stop escalation
3. Read `outputs/operator_brief.json` / `outputs/morning_open_dashboard.json` for UX
4. Read `outputs/trade_journal_hints.json` only as secondary context

## Historical hints policy

`trade_journal_hints.json` is deliberately **non-authoritative**.

Use it for:

- watch-pattern reminders
- journaling discipline prompts
- metric collection backlog
- operator review context

Do **not** use it for:

- flipping `trade_allowed`
- suppressing blockers
- increasing risk budget
- granting broker order rights

## Recommended naming cleanup

If you keep evolving the app, prefer names that reduce semantic collisions:

- `controller.confidence` -> `artifact_readiness_confidence`
- `confidence_evidence_quality` -> keep
- `confidence_trade_edge` -> keep
- `trade_journal_hints` -> keep clearly non-authoritative
