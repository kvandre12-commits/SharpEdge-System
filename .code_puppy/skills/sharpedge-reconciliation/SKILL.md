---
name: sharpedge-reconciliation
description: SharpEdge's nightly model-vs-reality reconciliation — how the gate's predictions are scored against what each day actually did (regime accuracy, direction accuracy, Brier, calibration), why honest n= labels matter, and the "build the scoreboard before optimizing" discipline. Use when interpreting reconcile_summary, judging whether the model has edge, or adding a calibration metric.
---

# SharpEdge Reconciliation (the honest scoreboard)

`scripts/reconcile_model_vs_reality.py`, run nightly in `ingest.yml`. It turns
one-off anecdotes into an accumulating, auditable track record. **Read-only on
sources; writes `outputs/reconcile_daily.csv` + `reconcile_summary.{txt,json}`.**

> **Discipline #1: build the scoreboard BEFORE optimizing the game.** Without it,
> every "the model is better now" claim is unprovable. With it, the system audits
> itself nightly and evidence compounds honestly.

## What it compares

For each session: the gate's **prediction** (`final_bias` + `prob_trend_fused` from
`execution_state_daily`) vs **reality** (`day_type` label + `sign(ret_1d)` from
`features_daily`).

- `predicted_regime(final_bias)`: `EXPANSION_FOLLOW*`→trend; `RANGE_FADE`/`PIN_FADE`→range;
  else **neutral** (excluded from regime accuracy — only *decisive* calls are scored).
- `predicted_direction(final_bias)`: `LONG`→+1, `SHORT`→−1, else 0 (fades are
  regime-only until the execution layer writes its chosen side back).

## Metrics (and how to read them honestly)

- **regime accuracy** over *decisive* calls. Compare to the **base rate** (~0.68 if you
  always said "range"), not to 0.5. Current ~0.75 on 359 days = modest real signal.
- **direction accuracy** — **carries an `n=` label and a `(thin!)` flag when n<20.**
  As of build time n≈1–2: a story, NOT evidence. Never quote it without the n=.
- **Brier** = mean((prob − y)²), lower better; compared to a **constant-base-rate
  baseline**. Beating the baseline (e.g. 0.196 < 0.217) means the probabilities add
  information.
- **calibration table**: predicted-prob buckets vs observed trend frequency. Well
  calibrated in the bulk low buckets; high-conviction buckets are thin (small n) —
  that's the honest state, surfaced not hidden.

## Invariants / rules

- **The n= caveats render in the output AND on the command deck** — the dashboard
  cannot flatter itself. Keep it that way; never strip an n= label.
- Pure mapping (`predicted_regime`/`predicted_direction`) is **unit-tested**
  (`tests/test_reconcile.py`). Changing the `final_bias` enum? Update the mapping +
  tests.
- The binding constraint on knowing whether there's edge is **sample size**. The
  pipeline accrues one labeled row per night; thin high-conviction buckets only fill
  with time. Don't force conclusions early.

## How this closes the loop

Reconciliation is the acceptance test for everything upstream:
- a new feature (`sharpedge-feature-engineering`) should *improve* Brier / accuracy here;
- a candidate context layer (e.g. the overlay) is only "gold" if it lifts the
  reconciled hit rate over time — not because the data is rich.
Run it after any model/gate change and check Brier + regime accuracy moved the
right way before claiming a win.
