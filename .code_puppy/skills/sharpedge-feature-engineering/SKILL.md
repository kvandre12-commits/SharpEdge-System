---
name: sharpedge-feature-engineering
description: How to engineer, VALIDATE, and promote intraday after-open features for SharpEdge's trend/range model — the OOS walk-forward discipline, the validated winners (signed_vol, lower_wick) with exact formulas, the proven duds to NOT re-test, and the exact promotion steps. Use whenever adding/judging a feature for build_intraday_trendday_prob_1130_fused.py or reasoning about model edge.
---

# SharpEdge Feature Engineering (validated, OOS-first)

Hard-won rules for adding signal to the 11:30 trend/range model. **Validated against
code + walk-forward results, not from memory.** Producer:
`scripts/build_intraday_trendday_prob_1130_fused.py`. Lab:
`scripts/analysis/intraday_feature_lab.py`.

> **Law #1: judge a feature by OUT-OF-SAMPLE walk-forward AUC, never train AUC.**
> Train AUC here is ~optimistic by +0.02–0.05. The lab's `walk_forward_auc()` is
> the only number that counts.

## The discipline (how we add a feature)

1. **Prototype in the lab, never in prod.** `intraday_feature_lab.py` is read-only,
   reuses the prod loaders/fit via importlib so lab and prod can't drift.
2. **Screen per-feature** (`baseline + ONE`), then check **stacking** (does it add
   over what we already have, or is it redundant?).
3. **Robustness:** require a positive delta across fold counts (4/5/6/8/10), not one
   lucky split.
4. **Multiple-testing caution:** every extra feature tested against the same folds
   raises the odds of a fake winner. Demand a **strong economic prior** + robustness.
   A +0.004 delta is noise; we promote only clear, theory-backed lifts.
5. **YAGNI:** promote the *single* best, not a kitchen sink. The all-features set has
   repeatedly UNDER-performed one good feature.

Realized OOS ceiling on this task ≈ **0.66** (market is ~68% range; ~30% trend base
rate). That's "weak rank signal," not an edge by itself. Don't oversell it.

## Validated WINNERS (in production)

Both computed over the open→11:30 cutoff bars (`g_cut`); both are after-open and
contain **no lookahead**.

| Feature | Formula | Economic read | OOS lift |
|---|---|---|---|
| `lower_wick` | `(min(open,close) − low) / (high − low)` of the session-so-far candle | long lower tail = buyers **absorbing** selling at the lows (rejection) → trend | **best single**, baseline 0.600 → 0.666 |
| `signed_vol` | `Σ(sign(close−open)·vol) / Σ vol` over the bars | volume concentrated one-directionally = trend; balanced = range | 0.600 → 0.647 |

`lower_wick` ≈ `signed_vol` in info (both read directional pressure); keeping both →
0.6665 (harmless, slightly robust). Effect of adding `lower_wick`: Brier 0.200→0.196,
prob_trend std 0.122→0.136, and the FIRST `EXPANSION_FOLLOW_LONG` state appeared.

## Proven DUDS — do not re-test (cost us; OOS-negative or noise)

- **Overlay context** (macro VIX/rates, dark-pool z, contango): every column −; ALL → 0.574.
  Overlay is real data but **not an intraday directional feature** (maybe a risk/sizing layer).
- **Gap / prior-day**: `gap_open_pct`, `gap_abs_pct`, `prior_range` hurt; `prior_ret_1d`/`prior_trend`
  only +0.002–0.004 (noise).
- **OHLC vol estimators**: Parkinson / Garman-Klass / Rogers-Satchell ≈ 0 to −.
- **Bar anatomy**: `IBS`, `CLV`, `upper_wick`, `body_ratio`, `close_loc`, `max_run_frac`
  (`max_run_frac` actively −0.023), `vwap_above_frac`, `vol_slope`, `late_accel` — all duds.

## Promoting a feature into production (6 edit points)

In `build_intraday_trendday_prob_1130_fused.py`:
1. **compute** it in `compute_intraday_features` (add to the per-session row dict).
2. add to **`feature_names`** (the training/scoring list).
3. add a **table column** in `ensure_table`'s CREATE.
4. add a **safe ALTER migration**: `if "<col>" not in existing: ALTER TABLE ... ADD COLUMN`.
5. add to the **upsert** INSERT column list **and** the VALUES placeholder count **and**
   the `ON CONFLICT DO UPDATE SET` list.
6. add to the **params tuple** (cast `float(r["<col>"])`).

Then rebuild + verify: `prob` spread, `frac>=0.65`, Brier via
`reconcile_model_vs_reality.py`. Run with the **system python**
(`/data/data/com.termux/files/usr/bin/python3` — has numpy/pandas/tzdata); the PATH
`python3` is a venv without them.

## Direction vs Magnitude — the efficient-market frontier (validated)

Tested on 359 days at the 11:30 horizon:

- **DIRECTION (back-half up/down) is ~unforecastable.** Best single feature OOS
  AUC **0.52** (lower_wick/wick_asym); most ~0.50; `ret_open_to_cutoff` 0.47
  (morning momentum slightly reverses). **`vs_vwap`/`vwap_proxy` is 0.46 —
  ANTI-predictive**, yet it's what `decide()` uses to pick EXPANSION direction.
  Don't fish for directional alpha; it isn't there (returns ≈ martingale).
- **MAGNITUDE (rest-of-day |move|) IS strongly forecastable.** Morning
  Garman-Klass / Parkinson vol → afternoon |move| with **Spearman IC ~0.40
  (0.21 OOS)**. Calibration: `expected |move|% ≈ 2.54 · morning_GK%`.

Note the inversion: the vol estimators are **duds for trend-classification** but
the **best magnitude predictors**. Target choice decides the winner.

**Strategy implication:** the edge is **structural + magnitude**, not directional
forecasting — gamma walls, the pin/magnet, fade-the-edge reversion, and sizing to
the expected move (surfaced on the deck's EXPECTED MOVE panel). The runner-long
*directional* call is the shakiest part of the system.

## Coil / squeeze (tested 3 ways — NO breakout edge in SPY; it FADES)

The compression→expansion coil does NOT work as a breakout in SPY, at any tested
horizon (don't re-chase it):
- daily: P(next-day expansion) 0.049 after compression vs 0.232 base.
- intraday width: narrowest channel quintile → SMALLEST forward range (monotonic =
  pure vol persistence, not coil).
- intraday squeeze→breakout (n=540): follow-through −0.014%, P(continue)=0.48 →
  **breaks REVERT.**

**Why (and it's consistent with everything):** SPY is a dealer-pinned, positive-
gamma index — long-gamma hedging sells rallies / buys dips and **suppresses
breakouts**. So a coil that pokes out gets sold back = the cockpit's `STICKY DAY →
FADE the edges`. The coil is real but it's a **FADE signal, not a breakout** — use
it to *strengthen* the fade side (`sharpedge-decision-gate`), never to chase a break.

## Calibration gotcha (already fixed, keep in mind)

`fit_logreg` adds the L2 penalty straight to the gradient → it's brutally strong.
`LOGREG_L2=1.0` censored the model to the base rate (max prob 0.355, no directional
state could fire). Default is now **0.01**. If predictions ever collapse to ~base
rate with near-zero spread, suspect over-regularization first.
