# SharpEdge Dirty Tree Workstream Split

This file breaks the current dirty tree into logical workstreams so the repo is
not treated like one giant undifferentiated branch.

Use this before deciding what to review, commit, test, or defer.

## Summary

The current tree is carrying at least five distinct workstreams:

1. Core cockpit decision-engine extraction and scoring expansion
2. Dashboard and cockpit-consumer rendering surfaces
3. Android / Phone Companion export and packaging
4. Data ingestion, analytics pipeline, and migration churn
5. Generated artifacts and runtime evidence churn

These should not be reviewed or landed as if they are one thing.

## Workstream 1: Core cockpit decision engine

Risk: high

Zone alignment:

- Red zone from `docs/architecture/CHANGE_ZONES.md`

Purpose:

- Extract and clarify the cockpit decision brain
- Separate scoring, weighting, context, setup conviction, targeting, and receipt logic
- Expand the visible Trade Gate surface without losing behavioral integrity

Files in this workstream:

- `cockpit/make_cockpit.py`
- `cockpit/setups.py`
- `cockpit/trade_permission.py`
- `cockpit/balance.py`
- `cockpit/context_chart_focus.py`
- `cockpit/decision_receipts.py`
- `cockpit/execution_vector_context.py`
- `cockpit/execution_vector_engine.py`
- `cockpit/execution_vector_primitives.py`
- `cockpit/execution_vector_weights.py`
- `cockpit/gate_workflows.py`
- `cockpit/setup_conviction.py`
- `cockpit/setup_event_lifecycle.py`
- `cockpit/targeting.py`
- `cockpit/trade_permission_context.py`
- `docs/architecture/CONTRACTS.md`
- `docs/trade_permission_vertical_audit_map.md`
- `tests/test_trade_permission.py`
- `tests/test_cockpit_balance.py`
- `tests/test_cockpit_volatility_coil.py`
- `tests/test_decision_receipts.py`

Why this is a distinct stream:

- `trade_permission.py` is now a wrapper, so the real math moved into the
  execution-vector modules
- setup quality, execution burden, target semantics, and decision receipts are
  being treated as first-class decision objects
- the acceptance / rejection / trap seam is explicitly important and should not
  be merged or simplified casually

Review posture:

- review as financial-math and decision-semantics work
- do not bury this under UI or artifact churn
- require explicit proof for behavioral changes

Suggested proof set:

```bash
python -m pytest tests/test_trade_permission.py tests/test_decision_receipts.py -q
```

Add the balance / volatility-coil tests when those surfaces are involved.

## Workstream 2: Dashboard and cockpit consumers

Risk: medium

Zone alignment:

- Mostly yellow zone, with some red-adjacent coupling because these surfaces
  display core decision objects

Purpose:

- Render the expanded cockpit decision objects into operator-facing views
- Improve local-first fallback behavior when external bridge logic is absent
- Add new charts and viewer context panels without changing contract meaning

Files in this workstream:

- `cockpit/live_chart_svg.py`
- `cockpit/live_read_view.py`
- `cockpit/dashboard_runtime.py`
- `cockpit/make_command_deck.py`
- `cockpit/make_pilot_board.py`
- `cockpit/make_options.py`
- `cockpit/make_overlay.py`
- `cockpit/make_price_volume.py`
- `cockpit/monthly_context_chart.py`
- `cockpit/weekly_context_chart.py`
- `tests/test_dashboard_runtime.py`
- `tests/test_live_read_view.py`

Why this is a distinct stream:

- these files mostly consume `signal.json`, Trade Gate objects, and related
  decision receipts rather than defining the canonical math
- rendering and local fallback should be reviewable separately from scoring
  behavior

Review posture:

- verify contract preservation
- verify renderer behavior
- do not confuse visible table/order changes with score-logic changes

Suggested proof set:

```bash
python -m pytest tests/test_dashboard_runtime.py tests/test_live_read_view.py -q
```

## Workstream 3: Android / Phone Companion export and packaging

Risk: medium

Zone alignment:

- Orange and yellow

Purpose:

- Carry the richer cockpit/signal contract into Android-facing packaging
- Export native-viewer bundle artifacts
- Add launcher/share helpers for Android import flows

Files in this workstream:

- `phone_companion/README.md`
- `phone_companion/export_signal_to_android_viewer.py`
- `phone_companion/export_operator_packet_to_android.py`
- `phone_companion/launchers/README.md`
- `phone_companion/launchers/run_phone_companion_android_operator_import.sh`
- `phone_companion/launchers/run_phone_companion_android_signal_import.sh`
- `phone_companion/launchers/share_operator_packet_to_android.py`
- `phone_companion/launchers/share_signal_to_android_viewer.py`
- `tests/test_android_viewer_export.py`
- `tests/test_android_operator_import_launcher.py`
- `tests/test_android_operator_packet_export.py`
- `tests/test_android_signal_import_launcher.py`

Why this is a distinct stream:

- these changes package and transport the contract rather than define the trade
  math itself
- breakage here is mostly contract drift, import/export breakage, or launcher
  UX failure

Review posture:

- audit exported field expectations
- keep source-of-truth in `outputs/signal.json` semantics
- verify Android bundle behavior separately from cockpit scoring review

Suggested proof set:

```bash
python -m pytest tests/test_android_viewer_export.py \
  tests/test_android_operator_import_launcher.py \
  tests/test_android_operator_packet_export.py \
  tests/test_android_signal_import_launcher.py -q
```

## Workstream 4: Data pipeline, ingestion, and migration churn

Risk: medium to high depending on which script is touched

Zone alignment:

- Mixed: infrastructure, feature pipeline, and some decision-input shaping

Purpose:

- Refresh ingestion paths, pipeline health, CSV exports, and supporting data
- Extend options/FINRA/daily-bar state handling
- Update migrations and support scripts used by the larger analytics pipeline

Files in this workstream:

- `scripts/build_auction_expectancy_events.py`
- `scripts/build_overlay_context_daily.py`
- `scripts/exports/export_liquidity_regime_csv.py`
- `scripts/ingest_alpaca_options_open_interest_daily.py`
- `scripts/ingest_cboe_options_chain_snapshots.py`
- `scripts/ingest_finra_darkpool_overlay.py`
- `scripts/ingest_spy_daily.py`
- `scripts/ingest_spy_intraday_alpaca.py`
- `scripts/join_macros_into_features.py`
- `scripts/options_snapshot_store.py`
- `scripts/print_gamma_close_card_300.py`
- `scripts/print_today_trade_card.py`
- `scripts/publish_sharpedge_2_report.py`
- `scripts/send_trade_card_to_discord.py`
- `scripts/utils/audit_pipeline.py`
- `scripts/audit_endpoint_freshness.py`
- `scripts/audit_workspace_endpoints.py`
- `scripts/update_decision_receipt_outcomes.py`
- `sql/migrations/001_playbook.sql`
- `sql/migrations/002_options_chain_snapshots.sql`
- `tests/test_finra_darkpool_overlay.py`
- `tests/test_ingest_spy_daily.py`
- `tests/test_market_data_sources.py`
- `tests/test_options_snapshot_enrichment.py`
- `tests/test_send_trade_card_to_discord.py`
- `tests/test_layer1_cache_controls.py`
- `tests/test_overlay_context.py`

Why this is a distinct stream:

- these files feed or audit the system, but they are not the same review lane as
  cockpit decision-engine math
- some changes may alter inputs to the cockpit indirectly, so they still deserve
  sober review

Review posture:

- separate data-freshness / pipeline-health work from cockpit scoring review
- be explicit when a pipeline change is expected to alter downstream signal
  behavior

Suggested proof set:

- run only the tests directly tied to the changed scripts
- do not treat generated CSV diffs as sufficient proof

## Workstream 5: Generated artifacts and runtime evidence churn

Risk: low as source, high as review noise

Zone alignment:

- Not source logic; derived artifacts

Files in this workstream:

- `data/spy_truth.db`
- `outputs/auction_expectancy_events.csv`
- `outputs/health/daily_bars_state.json`
- `outputs/health/finra_state.json`
- `outputs/latest_overlay_context_daily.csv`
- `outputs/liquidity_regime_events.csv`
- `outputs/overlay_context_daily.csv`
- `outputs/sharpedge_2_report.md`
- `outputs/spy_features_daily_with_macro.csv`
- `outputs/spy_finra_ats_weekly.csv`
- `outputs/spy_truth_daily.csv`
- `outputs/endpoint_audit_20260625.md`
- `outputs/endpoint_audit_latest.json`
- `outputs/endpoint_audit_latest.md`
- `outputs/execution_vector_burden_audit_20260625.md`
- `outputs/financial_code_gate_audit_20260625.md`
- `outputs/health/alpaca_options_open_interest_state.json`
- `outputs/health/cboe_options_chain_state.json`
- `outputs/pandas_numpy_audit_latest.md`
- `outputs/permission_receipts_spy.jsonl`
- `outputs/run_local_dashboard.log`
- `outputs/sharpedge_android_operator_import.json`
- `outputs/workspace_endpoint_audit_latest.json`
- `outputs/workspace_endpoint_audit_latest.md`

Why this is a distinct stream:

- these files are useful evidence and runtime state
- they are terrible as primary review material for architecture intent
- they can overwhelm the real code changes if not separated mentally

Review posture:

- review only when validating output consequences of source changes
- do not let artifact churn hide source-level risk
- avoid treating these files as doctrine

## Recommended review order

When triaging the current dirty tree, review in this order:

1. Workstream 1: core cockpit decision engine
2. Workstream 3: Android / Phone Companion export and packaging
3. Workstream 2: dashboard and cockpit consumers
4. Workstream 4: data pipeline and migrations
5. Workstream 5: generated artifacts

Reasoning:

- the core engine is the gem and the main risk surface
- Android / Phone Companion depends on stable contract meaning
- dashboard surfaces depend on both of those
- pipeline changes can be reviewed separately as upstream input work
- generated artifacts should come last, not first

## Recommended commit strategy

Do not land this tree as one blob.

Preferred split:

1. documentation / audit maps
2. core cockpit engine extraction and tests
3. dashboard consumer updates and tests
4. Android / Phone Companion export surfaces and tests
5. data pipeline changes and their focused tests
6. optional artifact refresh, if the repo intends to track them

## Working doctrine

If a file changes how the system decides:

- it belongs in Workstream 1 until proven otherwise

If a file changes how the decision is exported or rendered:

- it belongs in Workstream 2 or 3

If a diff is mostly CSV, JSON, SQLite, or logs:

- it belongs in Workstream 5 unless it is being used as evidence for another stream
