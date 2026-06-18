# Trade Journal Hints

```bash
python scripts/agents/trade_journal_hints.py
```

Outputs:

- `outputs/trade_journal_hints.json`
- `outputs/trade_journal_hints.txt`

## What it does

This script reads:

- the manual trade DB at `~/trade_journal/trades.db` by default
- markdown notes under repo `trade_journal/`

It builds a small historical-context artifact for operator-facing app logic.

## What it extracts

- top repeating setup + VWAP-behavior clusters
- win/loss and PnL summaries for closed journal trades
- average R and hold duration when available
- field coverage hints for logging discipline
- future metric backlog from markdown notes
- research hypotheses already written in the journal

## What it is allowed to do

- help the brief/dashboard show historical context
- suggest what to monitor or log better
- remind the operator about low-sample patterns

## What it is **not** allowed to do

- override `agent_v1_decision.json`
- open trade permission
- change broker permissions
- increase risk sizing

## Environment overrides

- `TRADE_JOURNAL_DB`
- `TRADE_JOURNAL_NOTES_DIR`
- `TRADE_HINT_MIN_PATTERN_SAMPLE_N`
- `TRADE_HINT_TOP_PATTERN_LIMIT`

## Current integration

When `outputs/trade_journal_hints.json` exists, `operator_brief.py` and
`morning_open_dashboard.py` can surface a compact `historical_hints` summary.
That summary is intentionally secondary context only.