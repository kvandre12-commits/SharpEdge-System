# Robinhood Capability Probe

This workflow classifies a Robinhood capability name into one of three buckets:

- `research_read`
- `active_trading_preview` / `active_trading_write`
- `unsupported_or_unverified`

It is intentionally honest:

- public read/research mappings come from the verified public Robinhood MCP source
- active-trading mappings come from local SharpEdge beta/delegate flows
- it does **not** authenticate to the hosted Robinhood bridge or run live `tools/list`

## Workflow

```text
.github/workflows/robinhood_capability_probe.yml
```

## Manual run

From the GitHub Actions tab, run **Robinhood Capability Probe** and pass a capability like:

- `get_portfolio`
- `get_equity_positions`
- `order_submit`
- `create_watchlist`

## Local run

```bash
python scripts/utils/robinhood_mcp_capability_probe.py get_portfolio
```

Outputs:

- `outputs/robinhood_capability_probe.json`
- `outputs/robinhood_capability_probe.txt`

## Why this exists

We need a repo-backed way to separate:

- safe research fetches
- active-trading intents
- fake or unverified names

before pretending a candidate belongs on the broker side.
