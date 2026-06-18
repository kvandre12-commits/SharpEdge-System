# Agent Language Protocol

The repo now has four canonical objects intended to be the shared language for
future agents:

1. `workflow_state`
2. `execution_plan`
3. `approval_decision`
4. `journal`

Build them with:

```bash
python scripts/agents/agent_language_objects.py
```

Outputs:

- `outputs/workflow_state.json`
- `outputs/workflow_state.txt`
- `outputs/execution_plan.json`
- `outputs/execution_plan.txt`
- `outputs/approval_decision.json`
- `outputs/approval_decision.txt`
- `outputs/journal.json`
- `outputs/journal.txt`

## Object roles

### `workflow_state`

What is true right now.

Use for:

- lifecycle stage
- market context
- system readiness snapshot
- blockers and freshness visibility

Do not use it to grant permission.

### `execution_plan`

What should happen next if conditions allow it.

Use for:

- intended action
- ordered steps
- prerequisites
- required approvals
- beta execution preview context

Do not use it to escalate permissions.

### `approval_decision`

What is actually authorized.

This is the only authoritative object for:

- `trade_allowed`
- `broker_order_allowed`
- `required_human_action`
- permission booleans
- blocking reasons
- risk flags

If any downstream object disagrees with `approval_decision`,
`approval_decision` wins.

### `journal`

What happened and what was learned.

Use for:

- latest operator entry
- recurring blockers / flags
- historical pattern reminders
- research backlog and lessons

This object is historical and non-authoritative.

## Consumption order

Future agents should read objects in this order:

1. `approval_decision.json`
2. `workflow_state.json`
3. `execution_plan.json`
4. `journal.json`

That order keeps authority separate from commentary.

## Common fields

Each object includes:

- `schema_version`
- `object_name`
- `created_ts`
- `run_id`
- `agent_id`
- `symbol`
- `authority`
- `source_artifacts`

The shared `run_id` is the stitching key across the four objects.
