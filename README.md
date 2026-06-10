# SharpEdge Systems

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

Extra operator artifacts:

```bash
python scripts/agents/operator_session_review.py
python scripts/agents/morning_open_dashboard.py
```

Design note:

- `docs/operator_breadcrumbs.md`

## Results Summary

| Metric | Value |
|--------|-------|
| Signals Generated | TBD |
| Win Rate | TBD |
| Avg Expectancy (R) | TBD |
| Max Drawdown | TBD |
| Latest Data Date | 2026-02-06 |
