---
name: sharpedge-analytics-pipeline
description: Map of SharpEdge's research/analytics layer (scripts/ + data/*.db) — how raw market/options/macro data is ingested into SQLite, turned into features, regimes, expectancy matrices, confidence weights, calibration, and daily signal strength, then validated by backtests and published as trade cards. Use when working on any build_*/ingest_*/backtest_* script, the spy_truth.db schema, or tracing where a daily number comes from.
---

# SharpEdge Analytics / Research Pipeline

Map of `SharpEdge-System/scripts/` (37 scripts) over the SQLite backbone.
This is the **offline research/daily-build** layer that feeds the live cockpit and
the operator contract. Not a live executor.

## Data backbone (SQLite)
- **`data/spy_truth.db`** — canonical truth store (**28 tables**). Set via env
  `SPY_DB_PATH`.
- **`data/market_data.db`** — secondary/raw store (23 tables, overlapping schema).
Key tables: `bars_daily`, `bars_intraday_15m`, `ats_weekly`, `features_daily`,
`liquidity_regime_events`, `open_resolution_regime`, `intraday_trendday_prob`,
`exhaustion_transition_state`, `auction_expectancy_events`,
`conditional_expectancy_matrix`, `confidence_matrix`, `dte_calibration`,
`edge_regime_pressure_dte_lut`, `execution_state_daily`, `options_chain_snapshot`.

## The layers (ingest → derive → validate → publish)

**1. Ingest (raw data in)**
- `ingest_spy_daily.py` (yfinance → `bars_daily`)
- `ingest_spy_intraday_alpaca.py` (→ `bars_intraday_15m`)
- `ingest_alpaca_options_chain_snapshots.py`, `ingest_alpaca_options_open_interest_daily.py`
- `ingest_finra_darkpool_overlay.py` (FINRA ATS weekly → `ats_weekly`)
- `ingest_fred_overlays.py` (FRED macro), `seed_tariff_overlays.py`

**2. Features**
- `build_features_spy_daily.py` → `features_daily`
- `join_macros_into_features.py` (fold FRED/macro overlays in)
- `aggregate_options_positioning_metrics.py` (options positioning → SQLite)

**3. Regimes (classify the day)**
- `build_regime_spy_daily.py`, `build_liquidity_regime_daily.py`
- `build_open_resolution_regime_intraday.py`,
  `build_intraday_trendday_prob_1130_fused.py`, `build_intraday_tr_gate.py`
- `build_exhaustion_transition_state.py`, `build_options_surface_state.py`

**4. Expectancy + backtests (does the edge pay?)**
- `build_auction_expectancy_events.py` → `auction_expectancy_events`
- `measure_gap_excursions.py`, `classify_fill_paths.py` (path-aware gap-fill outcomes)
- `build_conditional_expectancy_matrix.py` → `conditional_expectancy_matrix`
- `backtest_same_day_breakouts.py`, `backtest_failed_inside_slow_regime.py`
- `calibrate_dte_from_backtest.py` → `dte_calibration`;
  `load_edge_regime_pressure_lut.py` → `edge_regime_pressure_dte_lut`

**5. Decision synthesis (the daily SharpEdge number)**
- `build_confidence_weights.py` ("Confidence Weighting Layer") → `confidence_matrix`
- `build_signal_strength_daily.py` → `latest_signal_strength.csv` (feeds
  `controller_agent`)
- `build_risk_decision_layer.py`, `build_execution_state_daily.py` →
  `execution_state_daily`
- `build_intraday_execution_card.py`, `build_robinhood_fvg_monitor.py` →
  `robinhood_fvg_monitor.json` (the monitor the operator layer reads)

**6. Publish / present**
- `print_today_trade_card.py`, `print_gamma_close_card_300.py` (3pm gamma card)
- `publish_sharpedge_2_report.py`, `send_trade_card_to_discord.py`,
  `run_playbook.py`, `bootstrap_trade_execution_log.py`

## How it connects to the rest of the stack
`build_signal_strength_daily` + `build_robinhood_fvg_monitor` are the bridge into
the **operator-agents** layer (`controller_agent` reads `latest_signal_strength.csv`;
`agent_v1_decision`/`operator_brief`/`robinhood_beta_execution` read
`robinhood_fvg_monitor.json`). The **cockpit** is a separate real-time view
(Yahoo+CBOE) — see `sharpedge-cockpit-pipeline`.

## Working in this layer
- Most scripts are standalone CLIs reading/writing `spy_truth.db` (`SPY_DB_PATH`).
- No Makefile/orchestrator: run in layer order (ingest → features → regime →
  expectancy → confidence → execution_state → signal_strength → monitor → cards).
- Inspect schema: `sqlite3 data/spy_truth.db '.tables'` /
  `python3 -c "import sqlite3; ..."`.
- Backtests/expectancy are **research** — they inform confidence weights, never
  bypass the `approval_decision` authority gate or the Bridge's confirm gate.

## Pairs with
`sharpedge-operator-agents` (consumes signal_strength + monitor),
`sharpedge-cockpit-pipeline` (real-time sibling),
`sharpedge-market-microstructure` + `sharpedge-execution-governance` (live path).
