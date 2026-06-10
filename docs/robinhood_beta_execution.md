# Robinhood Beta Execution Handoff

```bash
python scripts/agents/robinhood_beta_execution.py
```

This beta-phase layer converts existing SharpEdge artifacts into a broker-facing
handoff that is still approval-gated.

## Outputs

- `outputs/robinhood_beta_execution.json`
- `outputs/robinhood_beta_execution.txt`

## Inputs

- `outputs/robinhood_fvg_monitor.json`
- `outputs/agent_v1_decision.json`
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`

## Beta posture

This is **not** autonomous live trading.

It supports:

- broker read / monitor context
- order-draft previews
- approval-queue readiness
- artifact-only fallback when bridge access is unavailable

It still blocks:

- submitting orders without operator approval
- replacing orders without operator approval
- canceling orders without operator approval

## Why this exists

Humans do not scale review and compounding cleanly by memory and manual clicks
alone. The beta layer gives the AI a structured Robinhood-facing handoff while
keeping live authority behind explicit approval gates.
