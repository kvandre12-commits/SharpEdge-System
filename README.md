# SharpEdge Systems

## What is this?

SharpEdge-System is the signal-generation and trade-gate analytics repo for
SharpEdge.

It produces `sharpedge.signal.v1` artifacts such as `outputs/signal.json` and
operator-facing decision-support objects.

## Who uses it?

- The operator running SharpEdge market reads.
- Phone Companion, which packages contracts for mobile use.
- SharpEdge Android, which renders the signal contract.
- Agents working on analytics, cockpit, contracts, and orchestration artifacts.

## What does it own?

- Market/cockpit signal generation.
- `trade_permission` / Trade Gate analytics.
- Operator artifacts and approval-state objects.
- Phone Companion contract packaging.
- Generated runtime proof artifacts.

## What does it not own?

- Native Android UI rendering.
- Browser rendering internals.
- Broker execution.
- Autonomous live orders.

## How do I test it?

Focused checks:

```bash
python -m pytest tests/test_trade_permission.py tests/test_phone_companion_golden_loop.py -q
python phone_companion/export_signal_to_android_viewer.py
```

Architecture entrypoints for agents:

```text
docs/architecture/OWNERSHIP_MAP.md
docs/architecture/SYSTEM_OVERVIEW.md
docs/architecture/CURRENT_STATE.md
docs/architecture/REPO_INVENTORY.md
docs/architecture/CONTRACTS.md
```

---

SharpEdge Systems is a systematic market data and regime analysis engine focused on
clarity, discipline, and decision-quality over noise.

Built from a background in precision craft, this project applies the same principles
of sharp tools, clean structure, and deliberate execution to financial data:

- Automated multi-source market data ingestion (Alpaca, FINRA, FRED)
- Feature engineering and liquidity/regime classification
- Backtested signal generation with calibrated options execution logic
- Deterministic trade cards designed to reduce emotional decision-making

The goal is simple:

**Create a reliable edge through structure, not speculation.**

---

## Architecture Overview

Pipeline layers:

1. **Truth Layer** – Raw market data ingestion and normalization  
2. **Feature Layer** – Derived signals, regimes, and structural context  
3. **Decision Layer** – Backtested rules, calibrated DTE selection, and trade plans  

All processes are automated via scheduled workflows and reproducible SQLite state.

---

## Purpose

SharpEdge Systems is both:

- A personal discipline framework for systematic trading  
- A production-style data engineering portfolio project  

It represents the transition from physical precision craft → data system design.

## Local Quality Gate

Ruff may require a Rust build on Android/Termux, so this repo includes a
stdlib-only fallback quality gate:

```bash
python scripts/utils/lint_python.py scripts
```

Optional stricter style audit, currently advisory while old debt is cleaned up:

```bash
python scripts/utils/lint_python.py scripts --strict-style
```

## FINRA Runtime Control

The FINRA darkpool overlay uses persisted `ats_weekly` state. Routine runs rebuild
daily overlays from SQLite and skip FINRA network calls while the cache is fresh.

Useful overrides:

```bash
FINRA_FORCE_REFRESH=1 python scripts/ingest_finra_darkpool_overlay.py
FINRA_CACHE_TTL_HOURS=24 python scripts/ingest_finra_darkpool_overlay.py
FINRA_REFRESH_LOOKBACK_WEEKS=8 python scripts/ingest_finra_darkpool_overlay.py
```

## Layer 1 Cache + State Controls

Layer 1 ingestion emits state breadcrumbs under `outputs/health/*_state.json`.
Routine runs now avoid unnecessary network or recompute work when persisted state is
fresh.

Useful overrides:

```bash
DAILY_FORCE_REFRESH=1 python scripts/ingest_spy_daily.py
DAILY_CACHE_TTL_HOURS=6 python scripts/ingest_spy_daily.py
DAILY_INCREMENTAL_PERIOD=30d python scripts/ingest_spy_daily.py

FRED_FORCE_REFRESH=1 python scripts/ingest_fred_overlays.py
FRED_MAX_LAG_DAYS=2 python scripts/ingest_fred_overlays.py

OPTIONS_POSITIONING_FORCE_REBUILD=1 DTE_MIN=0 DTE_MAX=1 \
  python scripts/aggregate_options_positioning_metrics.py
```

## Operator Brief MVP

The operator brief is a thin compression layer over the existing local-only
artifacts. It gives one fast stand-down / monitor / review summary without
changing the safety contract.

```bash
python scripts/agents/operator_brief.py
```

Outputs:

- `outputs/operator_brief.json`
- `outputs/operator_brief.txt`
- `outputs/operator_watchlist.json`
- `outputs/operator_journal_append.jsonl`
- `outputs/operator_session_review.json`
- `outputs/operator_session_review.txt`
- `outputs/morning_open_dashboard.json`
- `outputs/morning_open_dashboard.txt`
- `outputs/robinhood_beta_execution.json`
- `outputs/robinhood_beta_execution.txt`
- `outputs/trade_journal_hints.json`
- `outputs/trade_journal_hints.txt`
- `outputs/workflow_state.json`
- `outputs/workflow_state.txt`
- `outputs/execution_plan.json`
- `outputs/execution_plan.txt`
- `outputs/approval_decision.json`
- `outputs/approval_decision.txt`
- `outputs/journal.json`
- `outputs/journal.txt`

Extra operator artifacts:

```bash
python scripts/agents/operator_session_review.py
python scripts/agents/morning_open_dashboard.py
python scripts/agents/robinhood_beta_execution.py
python scripts/agents/trade_journal_hints.py
python scripts/agents/agent_language_objects.py
```

Design notes:

- `docs/operator_breadcrumbs.md`
- `docs/robinhood_beta_execution.md`
- `docs/permission_and_confidence_map.md`
- `docs/trade_journal_hints.md`
- `docs/agent_language_protocol.md`

## Results Summary

| Metric | Value |
|--------|-------|
| Signals Generated | TBD |
| Win Rate | TBD |
| Avg Expectancy (R) | TBD |
| Max Drawdown | TBD |
| Latest Data Date | 2026-02-06 |
