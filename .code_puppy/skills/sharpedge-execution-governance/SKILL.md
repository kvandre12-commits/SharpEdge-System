---
name: sharpedge-execution-governance
description: How SharpEdge's Robinhood Bridge classifies and routes every command (reads vs. order-writes vs. local custom logic), the approval tiers, and why no live order is ever submitted autonomously. Use whenever planning a Robinhood action, deciding if a capability is real, or wiring a new bridge command.
---

# SharpEdge Execution Governance & Command Routing

Validated against `SharpEdge-Robinhood-Bridge/src/sharpedge_robinhood_bridge/`
(`catalog.py`, `router.py`, `models.py`, `executor.py`, `trade_intent.py`).
**Core law: never claim unverified capability; route all order-writes through a
human gate.**

## The registry: `CommandSpec` (catalog.py)

Every command is one frozen `CommandSpec`:
`name, category, support_tier, route, summary, aliases, approval_policy
(default "not_applicable"), handler_name, notes`.
`find_command_spec()` matches **exact lowercased name or alias**. No fuzzy guessing.

## The 4 routes (the whole governance model)

| route | support_tier | approval_policy | What it means |
|-------|--------------|-----------------|---------------|
| `public_mcp_read` | `verified_public_source` | `not_applicable` | Read-only Robinhood data (accounts, portfolio, positions, quote, historicals, news, ratings, dividends, options_positions, order_history, search). **No live-order authority.** |
| `chatgpt_delegate` | `local_beta_delegate` | **`operator_confirm_required`** | The order_* family (`create_order_draft`, `order_draft`, `order_submit`, `order_cancel`, `order_replace`). category `active_trading_write`. Routes through the approval-gated ChatGPT Robinhood connector — **never direct autonomous submit.** |
| `custom_logic_local` | `implemented_custom_logic` | n/a | Implemented HERE with a `handler_name` (e.g. `create_watchlist`, `get_watchlists`). These are SharpEdge **workflow-state** watchlists — explicitly NOT verified Robinhood writes. |
| `custom_logic_required` | `custom_logic_candidate` | n/a | Modeled but unimplemented (`update_watchlist`, `get_option_chains`, `get_option_instruments`, `get_equity_tradability`, ...). A to-do, not a capability. |

Unknown command → `plan_command` returns `matched=False`, `route="unknown"`,
`approval_policy="unknown"` + note *"Do not pretend unknown commands are
source-verified."*

## Two execution surfaces (don't confuse them)

1. **`executor.run_command()`** — only executes `custom_logic_local` handlers
   (`_HANDLER_MAP = {create_watchlist, get_watchlists}`). Returns
   `CommandExecutionResult` with status:
   - `blocked` (unmatched command)
   - `not_implemented` (spec exists, no handler)
   - `invalid_payload` (handler raised ValueError)
   - handler status (`ok`) when executed.
   **It can NOT submit orders.** order_* have no handler here by design.

2. **`trade_intent.prepare_trade()`** — the order path. Runs
   `risk_check → plan_command(Tier C) → delegation_payload`, ending in status:
   - `blocked_by_risk` (guardrail tripped)
   - `blocked_unmodeled_command`
   - `awaiting_operator_confirm` (the human gate — normal terminal state)
   - `ready`
   The live submit happens **only after the operator taps confirm**, performed by
   the ChatGPT connector — not this code.

## Risk guardrails (trade_intent.py)
SPY-only allow-list · `MAX_EQUITY_QTY=1` · `MAX_OPTION_CONTRACTS=1` ·
`MAX_NOTIONAL_USD=1500` · kill-switch file `~/.sharpedge_kill` halts all trading.

## How to wire a NEW command (the right way)
1. Add a `CommandSpec` to `COMMAND_SPECS` with honest `route`/`support_tier`.
2. Read-only? → `public_mcp_read`. Order-write? → `chatgpt_delegate` +
   `operator_confirm_required`. Local feature? → `custom_logic_local` + a
   `handler_name`, then add the handler to `_HANDLER_MAP`.
3. Never mark something `verified_public_source` unless it truly is.
4. Add/extend tests (the repo has test coverage for these paths).

## Guardrails for the agent
- If a Robinhood capability isn't in the catalog, say "not modeled yet" — don't fake it.
- Treat `awaiting_operator_confirm` as success, not a failure to push through.
- This skill pairs with `sharpedge-market-microstructure` (which decides WHETHER to
  trade); this one governs HOW any resulting action is routed and gated.
