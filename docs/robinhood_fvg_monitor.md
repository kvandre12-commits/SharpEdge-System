# Robinhood FVG Monitor Handoff

SharpEdge now builds a local-only Robinhood MCP handoff for fair-value-gap fill monitoring.

## What it does

`scripts/build_robinhood_fvg_monitor.py` reads existing repository artifacts:

- `outputs/gap_excursion_metrics.csv`
- `outputs/top_gap_fill_edges.csv`
- `data/spy_truth.db`
  - `options_positioning_metrics`
  - `risk_decision_layer` when available

It writes:

- `outputs/robinhood_fvg_monitor.json`
- `outputs/robinhood_fvg_monitor.txt`

The JSON includes:

- latest gap/fill state
- matched historical gap-fill edge
- options positioning context
- directional options watch hypothesis
- Robinhood MCP handoff metadata
- bridge status for optional Code Puppy MCP integration

If the local Robinhood MCP bridge is disabled, unbound, or unconfigured, the
script still emits the monitor artifact. In that case the handoff marks:

- `bridge_status.available = false`
- `fallback_mode = artifact_only_manual_review`
- `manual_review_required = true`

## Safety posture

The handoff is intentionally dry-run only.

Allowed MCP actions:

- read account status
- read positions
- read option chains
- monitor underlying quotes
- monitor option quotes

Blocked MCP actions:

- place order
- replace order
- cancel order without operator confirmation

Any order path must require explicit operator confirmation.

When the bridge is unavailable, downstream safety contracts downgrade broker
read/monitor permissions while preserving the artifact for manual review.

## SPY and NVDA overnight-gap watch

`scripts/build_robinhood_gap_watch.py` adds a smaller daily-bar monitor for SPY
and NVDA. It reads `bars_daily` without modifying it and writes:

- `outputs/robinhood_gap_watch.json`
- `outputs/robinhood_gap_watch.txt`

The artifact separates two layers that must not be conflated:

1. **Session gap:** the latest open versus the prior close, with a default 0.25%
   threshold and same-session fill check.
2. **Historical refill structure:** unresolved down-gap/dip targets accumulated
   across the full loaded timeline. A new event while older targets remain open
   forms a double-, triple-, or multi-dip stack.

A filled latest-session gap can therefore remain `no_trade` at the session layer
while the symbol is `watch` at the structural layer because older refill targets
remain unresolved. The output preserves stack depth, event dates, target levels,
age, and matching symbol-specific historical stack statistics. SPY and NVDA use
separate five-year refill surfaces; statistics are never borrowed across symbols.
The artifact permits read-only quote/historical monitoring and blocks all order
actions.

Refresh and build manually:

```bash
SYMBOL=SPY python scripts/ingest_spy_daily.py
SYMBOL=NVDA python scripts/ingest_spy_daily.py
python scripts/build_robinhood_gap_watch.py
```

Useful controls:

- `GAP_WATCH_SYMBOLS=SPY,NVDA`
- `GAP_WATCH_MIN_GAP_PCT=0.0025`
- `GAP_WATCH_MAX_STALE_DAYS=4`
- `SPY_DB_PATH=data/spy_truth.db`
- `REFILL_SURFACE_DIR=outputs`

Canonical research artifacts:

- SPY: `historical_refill_surface.json` and
  `historical_refill_stack_surface.json`
- NVDA: `nvda_historical_refill_surface.json` and
  `nvda_historical_refill_stack_surface.json`

## Why Gemini was removed

The prior controller depended on Gemini API access and leaked a failed request URL into an output artifact. The controller is now deterministic and local-only:

- `scripts/agents/controller_agent.py`

It keeps the same `outputs/agent_controller_decision.json` contract while avoiding external AI API dependencies.
