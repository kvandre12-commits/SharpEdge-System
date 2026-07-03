# Argus MCP Wrapper Spec

This document defines the first-pass MCP wrapper contract that lets Argus talk to SharpEdge without pretending to own strategy, permission, or broker authority.

## Golden rule

Argus can request.
SharpEdge decides.
Bridge translates.
Operator approves.

## Scope

This is the **v0 wrapper surface** for Argus as an MCP client.
It is intentionally small.

Canonical v0 MCP tools:

1. `sharpedge.discover_surface`
2. `sharpedge.get_latest_state`
3. `sharpedge.get_execution_card`
4. `sharpedge.explain_permission`
5. `sharpedge.prepare_broker_handoff`
6. `sharpedge.validate_handoff`

## Why this is 6-tool v0

Validation is its own hard gate.
It does not get merged into handoff preparation.

That separation keeps the contract honest:
- preparation assembles or writes the handoff
- validation confirms route, approval posture, and downstream readiness

## Explicit non-goals for v0

This wrapper spec does **not** define:
- permission calculation in Argus
- execution-card construction in Argus
- broker payload invention in Argus
- direct broker execution from Argus
- a `sharpedge.execute_validated_handoff` tool in v0

Execution remains downstream of this wrapper surface and must stay approval-gated.

## Resource note

Positions are still important, but in v0 they remain a **resource-first** surface:
- `sharpedge://positions/latest`

Argus may read positions through the resource layer without needing a separate v0 tool.

## Naming rule

For the actual wrapper layer, canonical tool names use dotted MCP-style names:
- `sharpedge.discover_surface`
- not `sharpedge_discover_surface`

This spec is the canonical naming source for the first-pass wrapper implementation.
Any non-dotted naming should be treated as stale unless explicitly documented later as deprecated.

## Shared response conventions

Every tool response should include:
- `status` — `ok`, `blocked`, `invalid_input`, `not_found`, or `error`
- `tool_name` — canonical dotted tool name
- `authority` — system responsible for the authoritative result
- `mutability` — `read_only`, `write_artifact`, or `validate_only`
- `generated_at` — UTC timestamp
- `source_refs` — list of files, artifacts, or modules used

Error responses should also include:
- `error_code`
- `message`
- `retryable`

## Tool 1: `sharpedge.discover_surface`

### Purpose
Return the currently supported Argus-facing MCP surface, including tools, resources, authority boundaries, and mutability.

### Authority
Argus wrapper inventory over SharpEdge + Bridge backing surfaces.

### Mutability
`read_only`

### Input schema

```json
{
  "include_resources": true,
  "include_tools": true,
  "include_authority_map": true,
  "include_legacy_aliases": false
}
```

All fields optional.

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.discover_surface",
  "authority": "Argus-MCP-Wrapper",
  "mutability": "read_only",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "broker_app_1.0/bridge/real_surface_inventory.json",
    "broker_app_1.0/tools/argus_tool_aliases.json",
    "broker_app_1.0/docs/authority_map.md"
  ],
  "surface": {
    "tools": [],
    "resources": [],
    "authority_boundary": {}
  }
}
```

### Failure behavior
- `invalid_input` if a flag is not boolean
- `error` if the wrapper inventory cannot be loaded
- never invent missing tools or resources; omit or mark unsupported instead

## Tool 2: `sharpedge.get_latest_state`

### Purpose
Return the latest authoritative SharpEdge state packet for operator review.

### Authority
SharpEdge

### Mutability
`read_only`

### Input schema

```json
{
  "source": "latest",
  "include_artifact_path": true,
  "include_raw_signal": true
}
```

All fields optional.

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.get_latest_state",
  "authority": "SharpEdge",
  "mutability": "read_only",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "~/SharpEdge-System/outputs/signal.json"
  ],
  "state": {
    "schema": "sharpedge.signal.v1",
    "symbol": "SPY",
    "spot": 620.15,
    "gamma_regime": "negative",
    "setup_tag": "failed_break_reclaim",
    "trade_permission": {}
  },
  "artifact_path": "~/SharpEdge-System/outputs/signal.json"
}
```

### Failure behavior
- `not_found` if `outputs/signal.json` does not exist
- `error` if the JSON is unreadable or invalid
- `error` if the artifact is present but not a non-empty JSON object

## Tool 3: `sharpedge.get_execution_card`

### Purpose
Return the latest authoritative execution card without recalculating it in Argus.

### Authority
SharpEdge

### Mutability
`read_only`

### Input schema

```json
{
  "source": "latest",
  "include_reasons": true,
  "include_execution_flow": true,
  "include_execution_hierarchy": true
}
```

All fields optional.

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.get_execution_card",
  "authority": "SharpEdge",
  "mutability": "read_only",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "~/SharpEdge-System/outputs/signal.json",
    "cockpit/execution_card_builder.py"
  ],
  "execution_card": {
    "schema": "sharpedge.trade_permission.v1",
    "trade_permission_score": 73,
    "trade_gate": "pass",
    "bias": "BULLISH",
    "setup_conviction": {},
    "execution_flow": {},
    "execution_hierarchy": {},
    "supporting_reasons": [],
    "warning_reasons": []
  }
}
```

### Failure behavior
- `not_found` if latest state is unavailable
- `error` if the latest state lacks a usable permission payload
- `error` if the card cannot be derived from authoritative SharpEdge output
- Argus must not backfill or fabricate missing card fields

## Tool 4: `sharpedge.explain_permission`

### Purpose
Explain the latest authoritative permission card in plain language without changing the score, gate, or underlying decision.

### Authority
SharpEdge output, explained by Argus wrapper presentation logic

### Mutability
`read_only`

### Input schema

```json
{
  "source": "latest",
  "detail_level": "standard",
  "audience": "operator",
  "include_risk": true,
  "include_invalidation": true,
  "include_reasons": true
}
```

Allowed `detail_level` values:
- `brief`
- `standard`
- `deep`

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.explain_permission",
  "authority": "SharpEdge",
  "mutability": "read_only",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "~/SharpEdge-System/outputs/signal.json",
    "cockpit/execution_card_builder.py"
  ],
  "explanation": {
    "score": 73,
    "gate": "pass",
    "plain_language_summary": "Permission is 73 because the current setup and execution context pass the trade gate, but warnings still matter.",
    "supporting_reasons": [],
    "warning_reasons": [],
    "risk_notes": [],
    "invalidation_notes": []
  }
}
```

### Failure behavior
- `invalid_input` if `detail_level` is unsupported
- `not_found` if no latest execution card is available
- `error` if explanation inputs are missing or malformed
- explanation must preserve authoritative values; it may summarize them, not mutate them

## Tool 5: `sharpedge.prepare_broker_handoff`

### Purpose
Prepare the latest broker handoff using authoritative SharpEdge state and Robinhood Bridge policy.

### Authority
SharpEdge-Robinhood-Bridge

### Mutability
`write_artifact`

### Input schema

```json
{
  "source": "latest",
  "command": "order_submit",
  "operator_approved": true,
  "test": false,
  "write_latest_artifact": true
}
```

Required rule:
- `operator_approved` must be `true`

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.prepare_broker_handoff",
  "authority": "SharpEdge-Robinhood-Bridge",
  "mutability": "write_artifact",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "~/SharpEdge-System/outputs/signal.json",
    "src/sharpedge_robinhood_bridge/cockpit_adapter.py",
    "src/sharpedge_robinhood_bridge/trade_intent.py"
  ],
  "handoff": {
    "schema": "sharpedge.robinhood_execution_handoff.v1",
    "decision": {},
    "command_plan": {},
    "delegation": {}
  },
  "artifact_path": "~/SharpEdge-System/outputs/robinhood_execution_handoff.json"
}
```

### Failure behavior
- `blocked` if `operator_approved` is not `true`
- `not_found` if latest SharpEdge state is unavailable
- `blocked` if SharpEdge stands down and no broker handoff should be prepared
- `error` if handoff planning fails or artifact writing fails
- this tool must not silently downgrade a blocked handoff into an executable one

## Tool 6: `sharpedge.validate_handoff`

### Purpose
Validate a prepared handoff for route, approval posture, payload completeness, and downstream delegation readiness.

### Authority
SharpEdge-Robinhood-Bridge

### Mutability
`validate_only`

### Input schema

```json
{
  "handoff_path": "~/SharpEdge-System/outputs/robinhood_execution_handoff.json",
  "use_latest_if_missing": true,
  "check_route": true,
  "check_approval_policy": true,
  "check_payload_contracts": true
}
```

At least one of these must be true:
- `handoff_path` provided
- `use_latest_if_missing` true

### Output schema

```json
{
  "status": "ok",
  "tool_name": "sharpedge.validate_handoff",
  "authority": "SharpEdge-Robinhood-Bridge",
  "mutability": "validate_only",
  "generated_at": "2026-07-03T00:00:00Z",
  "source_refs": [
    "~/SharpEdge-System/outputs/robinhood_execution_handoff.json",
    "src/sharpedge_robinhood_bridge/catalog.py",
    "src/sharpedge_robinhood_bridge/payload_contracts.py"
  ],
  "validation": {
    "valid": true,
    "route": "chatgpt_delegate",
    "approval_policy": "operator_confirm_required",
    "ready_for_delegation": true,
    "issues": [],
    "warnings": []
  }
}
```

### Failure behavior
- `not_found` if no handoff can be loaded
- `invalid_input` if the handoff path and flags are unusable
- `blocked` if required approval posture or payload requirements are not satisfied
- `error` if the handoff shape cannot be parsed or validated
- validation must not execute, submit, or route the handoff by itself

## Golden flow across the 6-tool surface

1. `sharpedge.discover_surface`
2. `sharpedge.get_latest_state`
3. `sharpedge.get_execution_card`
4. `sharpedge.explain_permission`
5. operator decides whether to approve preparation
6. `sharpedge.prepare_broker_handoff`
7. `sharpedge.validate_handoff`
8. only then may downstream bridge/connector execution proceed

## Out of scope but likely next

Possible future additions after v0:
- `sharpedge.get_positions` as a convenience tool if resource-only access proves awkward
- `sharpedge.get_latest_handoff`
- `sharpedge.execute_validated_handoff` only if it remains strictly approval-gated and does not move broker authority into Argus

## Implementation doctrine

The wrapper layer should be thin.

It should:
- name the surfaces clearly
- enforce input/output contracts
- preserve authority boundaries
- expose real backing artifacts and functions

It should not:
- duplicate SharpEdge logic
- duplicate Robinhood Bridge routing policy
- create “assistant guesses” where authoritative artifacts are missing
