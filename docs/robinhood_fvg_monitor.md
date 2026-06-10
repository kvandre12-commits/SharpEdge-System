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

## Why Gemini was removed

The prior controller depended on Gemini API access and leaked a failed request URL into an output artifact. The controller is now deterministic and local-only:

- `scripts/agents/controller_agent.py`

It keeps the same `outputs/agent_controller_decision.json` contract while avoiding external AI API dependencies.
