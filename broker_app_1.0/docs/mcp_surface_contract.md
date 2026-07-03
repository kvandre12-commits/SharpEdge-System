# Argus MCP Surface Contract

Argus is the operator-facing broker app surface for SharpEdge.

The first build goal is not broker execution. The first build goal is a clean MCP surface-to-surface connection between Argus and SharpEdge.

## Boundary

Argus does:
- receive operator intent
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

## Minimum MCP Resources

Argus should be able to read these SharpEdge resources:

- `sharpedge://state/latest`
- `sharpedge://execution/card/latest`
- `sharpedge://permission/latest`
- `sharpedge://positions/latest`
- `sharpedge://handoff/latest`

## Minimum MCP Tools

Argus should be able to call these SharpEdge tools:

- `sharpedge_get_latest_state`
- `sharpedge_get_execution_card`
- `sharpedge_prepare_broker_handoff`
- `sharpedge_validate_handoff`
- `sharpedge_explain_permission`

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

## Core Rule

Permission before broker action.

If SharpEdge cannot produce a valid handoff, Argus must stop.
