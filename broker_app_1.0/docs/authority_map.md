# Argus Authority Map

This page answers the architectural question that causes the most accidental mess:

**Who is allowed to decide what?**

## Topology

```text
Operator
  |
  v
Argus
  |- discovers
  |- reads
  |- explains
  |- delegates
  v
SharpEdge
  |- permission
  |- scoring
  |- execution cards
  |- handoffs
  v
Robinhood Bridge
  |- translation
  |- broker execution routing
```

## Decision boundaries

### Operator
Owns:
- final approval for live-order style actions
- acceptance or rejection of a proposed handoff
- conversational intent given to Argus

Does not own:
- SharpEdge scoring logic implementation
- broker routing policy implementation

### Argus
Owns:
- operator-facing conversation layer
- MCP client behavior
- discovery of available SharpEdge surfaces
- reading current state, cards, positions, and handoffs
- plain-language explanation of authoritative SharpEdge output
- delegation of validated handoffs to the proper downstream path

Does not own:
- permission calculation
- scoring or regime logic
- execution-card construction
- broker payload invention
- direct broker authority independent of SharpEdge and Bridge

### SharpEdge
Owns:
- market-state authority
- execution-permission authority
- scoring and regime logic
- execution-card construction
- handoff preparation truth

Does not own:
- conversational client UX
- broker-specific translation details

### Robinhood Bridge
Owns:
- broker-specific translation
- command classification and routing policy
- risk and approval gating for broker-side actions
- broker execution routing after validation and approval

Does not own:
- strategy truth
- trade-permission scoring
- operator-facing reasoning as a source of authority

## Core law

Argus may **present** and **explain** authority.
Argus may not **invent** authority.

That means:
- if SharpEdge cannot produce permission, Argus cannot fake it
- if SharpEdge cannot produce a valid handoff, Argus must stop
- if Bridge requires explicit approval, Argus must preserve that gate

## First-iteration Argus posture

Argus should begin as a disciplined MCP client with these jobs:

1. Discover
2. Read
3. Explain
4. Delegate

That posture is intentionally smaller than “full broker app.”
It keeps the seams honest and the architecture DRY.
