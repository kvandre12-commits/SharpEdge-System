# SharpEdge Systems

[![CI](https://github.com/kvandre12-commits/SharpEdge-System/actions/workflows/ci.yml/badge.svg)](https://github.com/kvandre12-commits/SharpEdge-System/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

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

## See it in action

Every run emits a **deterministic decision card** — and, crucially, it will *refuse
to trade* when the evidence isn't there. Discipline is the feature, not an
afterthought:

```
SHARPEDGE AGENTIC AI V1 DECISION
Symbol: SPY
Decision: hold
Trade allowed: False
Risk state: PROBE
Blocking reasons: controller_hold, monitor_no_trade, sample_n_below_30, stale_or_missing_inputs
Risk flags: broker_integration_unavailable, freshness_gate_failed, low_sample, monitor_blocks_trade
Orders remain blocked unless an operator manually confirms outside this contract.
```

Under the hood it ranks **regime x pressure x DTE** buckets by backtested
expectancy, with a minimum sample size enforced before any bucket counts:

| regime | pressure | n | win | exp | sharpe | maxDD |
|---|---|---:|---:|---:|---:|---:|
| high_vol / rising_voltrend | NORMAL | 28 | 82.1% | 0.0054 | 3.62 | -2.01% |
| mid_vol / rising_voltrend | NORMAL | 44 | 72.7% | 0.0031 | 3.89 | -1.10% |
| mid_vol / falling_voltrend | NORMAL | 49 | 63.3% | 0.0022 | 2.84 | -2.10% |

*Full artifacts live in [`outputs/`](outputs/) — trade cards, expectancy matrices,
gate sweeps, and execution attribution.*

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

Extra operator artifacts:

```bash
python scripts/agents/operator_session_review.py
python scripts/agents/morning_open_dashboard.py
python scripts/agents/robinhood_beta_execution.py
```

Design notes:

- `docs/operator_breadcrumbs.md`
- `docs/robinhood_beta_execution.md`

## Results

Rather than one headline number, SharpEdge reports **per-regime expectancy** (see
the table above and the full matrices in [`outputs/`](outputs/)). Top backtested
buckets show roughly 63-82% win rates with controlled drawdowns (around 2% or less)
at enforced minimum sample sizes — and the live decision layer still gates every
order behind freshness, sample-size, and monitor checks before anything executes.
