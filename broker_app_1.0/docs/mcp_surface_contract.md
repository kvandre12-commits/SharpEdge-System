# Argus MCP Surface Contract

Argus is the operator-facing MCP client surface for SharpEdge.

The first build goal is not broker execution. The first build goal is a clean MCP surface-to-surface connection between Argus and SharpEdge.

## First-iteration posture

Argus should begin as a disciplined MCP client whose first responsibilities are:
- discover available SharpEdge surfaces
- read current state and cards
- explain permission, risk, and invalidation in plain language
- delegate validated handoffs to the proper downstream execution path

Argus should not begin as a giant broker-tool bucket.

## Boundary

Argus does:
- receive operator intent
- discover available SharpEdge surfaces
- request SharpEdge state
- request execution cards
- display permission, risk, and explanations
- prepare broker-facing commands only after explicit operator approval

Argus does not:
- invent trades
- override SharpEdge permission
- call a broker directly without a validated handoff
- mutate engine state without an explicit tool contract

SharpEdge does:
- own market state
- own execution permission
- own scoring and regime logic
- own trade-card construction
- emit broker-ready handoff packets

Robinhood Bridge does:
- translate validated handoff packets into broker-specific command plans
- keep broker rules separate from SharpEdge scoring
- own broker execution routing after validation and approval

## Minimum MCP Resources

Argus should be able to read these SharpEdge resources:

- `sharpedge://state/latest`
- `sharpedge://execution/card/latest`
- `sharpedge://permission/latest`
- `sharpedge://positions/latest`
- `sharpedge://handoff/latest`

## Minimum MCP Tools

Argus should be able to call these SharpEdge tools:

- `sharpedge.discover_surface`
- `sharpedge.get_latest_state`
- `sharpedge.get_execution_card`
- `sharpedge.explain_permission`
- `sharpedge.prepare_broker_handoff`
- `sharpedge.validate_handoff`

## Real backing surfaces today

These named MCP resources/tools are mostly a **product shell** today. The real backing surfaces already exist across the live SharpEdge stack:

### `SharpEdge-System`
Owns:
- `outputs/signal.json`
- `trade_permission` / execution-card truth
- market-state, scoring, regime, and invalidation logic

Key evidence:
- `OWNERSHIP.md`
- `cockpit/execution_card_builder.py`

### `SharpEdge-Robinhood-Bridge`
Owns:
- signal-to-handoff planning
- command classification and routes
- risk gates
- handoff artifact writing
- live position feedback loading

Key evidence:
- `src/sharpedge_robinhood_bridge/cockpit_adapter.py`
- `src/sharpedge_robinhood_bridge/trade_intent.py`
- `src/sharpedge_robinhood_bridge/catalog.py`
- `src/sharpedge_robinhood_bridge/payload_contracts.py`
- `src/sharpedge_robinhood_bridge/position_feedback.py`

### `code_puppy`
Owns:
- ChatGPT Robinhood delegation artifact packaging
- connector-facing prompt and JSON handoff packaging

Key evidence:
- `code_puppy/plugins/chatgpt_robinhood_delegate/`

## Resource-to-implementation map

### `sharpedge://state/latest`
Current semantic backing:
- `~/SharpEdge-System/outputs/signal.json`

### `sharpedge://execution/card/latest`
Current semantic backing:
- `signal.json["trade_permission"]`
- card construction in `cockpit/execution_card_builder.py`

### `sharpedge://permission/latest`
Current semantic backing:
- the `sharpedge.trade_permission.v1` payload emitted by SharpEdge

### `sharpedge://positions/latest`
Current semantic backing:
- `~/SharpEdge-System/outputs/robinhood_live_positions.json`
- loaded through bridge `position_feedback.py`

### `sharpedge://handoff/latest`
Current semantic backing:
- `~/SharpEdge-System/outputs/robinhood_execution_handoff.json`
- written by bridge `cockpit_adapter.py`

## Tool-to-implementation map

### `sharpedge.discover_surface`
Current semantic backing:
- wrapper inventory over `bridge/real_surface_inventory.json`
- authority map in `docs/authority_map.md`
- canonical names in `tools/argus_tool_aliases.json`

### `sharpedge.get_latest_state`
Current semantic backing:
- read latest SharpEdge state from `outputs/signal.json`

### `sharpedge.get_execution_card`
Current semantic backing:
- read/derive the latest permission card from SharpEdge trade-permission output

### `sharpedge.explain_permission`
Current semantic backing:
- permission reasons and execution-card explanation fields from SharpEdge

### `sharpedge.prepare_broker_handoff`
Current semantic backing:
- bridge `plan_signal_handoff(...)`
- bridge `write_handoff_artifact(...)`

### `sharpedge.validate_handoff`
Current semantic backing:
- bridge risk checks
- command route planning
- approval policy enforcement
- connector payload contract generation

## Golden Flow

1. Operator asks Argus for a market or trade review.
2. Argus reads latest SharpEdge state.
3. Argus requests an execution card.
4. SharpEdge returns permission, setup, risk, invalidation, and broker eligibility.
5. Argus explains the card in plain language.
6. Operator explicitly approves preparation.
7. Argus calls SharpEdge to prepare a broker handoff.
8. Argus validates the handoff.
9. Only then can the Robinhood Bridge receive the handoff.

## Current gap

The named Argus MCP wrapper surface is not fully implemented yet as a first-class MCP server with these exact resource and tool names.

What already exists is the **real contract logic** behind the surface.
That means Broker App 1.0 should evolve by wrapping and naming the real backing surfaces — not by duplicating or re-inventing them.

The architecture stays cleaner when Argus remains the conversational/MCP client layer over well-defined services instead of pretending to own strategy or broker truth.

## Core Rule

Permission before broker action.

If SharpEdge cannot produce a valid handoff, Argus must stop.
