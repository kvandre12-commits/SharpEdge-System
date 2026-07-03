# First Real-World Argus Demo

This is the first demo that should work in the real world.
Not a mock. Not a transport-only proof. A real operator loop.

## Goal

Prove that Argus can coordinate **two authoritative systems** without pretending to be either one:

- **SharpEdge** for execution authority
- **Robinhood** for brokerage authority

## Demo question

Use a question shaped like:

> Can I take the next SPY setup right now, and if so, what should I review before any broker action?

That question is good because it forces Argus to:
- read SharpEdge first
- read Robinhood second
- fuse both truths
- stop if permission is not present

## Required phases

### Phase 1 — Read SharpEdge

Argus should gather:
- latest state
- execution card
- permission explanation

Minimum SharpEdge calls:
- `sharpedge.get_latest_state`
- `sharpedge.get_execution_card`
- `sharpedge.explain_permission`

Important SharpEdge fields to respect when present:
- `trade_gate`
- `trade_permission_score`
- `execution_flow.execution_permission`
- `execution_flow.live_trigger_check`
- supporting and warning reasons
- invalidation / execution hierarchy context

SharpEdge is the authority on whether a setup is actionable.

### Phase 2 — Read Robinhood

Argus should gather:
- available accounts
- account buying power / portfolio state
- current positions
- existing open orders
- tradability if an instrument-specific follow-up is needed

Minimum Robinhood reads:
- `get_accounts`
- `get_portfolio`
- `get_option_positions` or `get_equity_positions`
- `get_option_orders` or `get_equity_orders`

Optional follow-up reads:
- `get_equity_tradability`
- `get_option_chains`
- `get_option_instruments`
- `get_option_quotes`
- `get_equity_quotes`

Robinhood is the authority on whether the account is able to support the requested broker action.

### Phase 3 — Fuse

Argus should answer in three clearly separated sections:

1. **SharpEdge says**
2. **Robinhood says**
3. **Argus decision**

Example stop case:

- SharpEdge says: `trade_gate = CAUTION`, live trigger not present
- Robinhood says: buying power is available
- Argus decision: do **not** prepare a broker handoff yet

That is a successful demo.
The point is not to force a trade. The point is to prove the orchestration contract.

## Success criteria

The demo is successful if all of these are true:

1. Argus uses real SharpEdge state, not invented summaries.
2. Argus uses real Robinhood account data, not assumed account state.
3. Argus keeps authorities separate in its explanation.
4. Argus does not treat buying power as trading permission.
5. Argus does not prepare or place a broker action when SharpEdge has not permitted it.
6. If SharpEdge does permit action, Argus moves to review-first behavior, not immediate placement.

## Explicit non-goals

This first real-world demo does **not** need to:
- place a live order
- cancel a live order
- support every Robinhood action
- expose every SharpEdge function through MCP
- solve multi-leg options orchestration

YAGNI, my beloved.

## Recommended follow-on path

Once this demo works reliably, the next honest step is:

1. prepare a SharpEdge handoff when permission exists
2. map the intended instrument on Robinhood
3. run the relevant Robinhood review tool
4. show the operator the review result
5. wait for explicit confirmation before any live write

That becomes the bridge from "real read orchestration" to "real approval-gated execution orchestration."
