---
name: sharpedge-decision-gate
description: How SharpEdge turns analytics + the live signal into a trade decision — the daily compounding engine (execution_score + final_bias state machine) and the Bridge decide() runner/fade playbooks, always 1 option contract, with the symmetric tighten-only analytics veto. Use whenever changing decide(), the final_bias states, position instrument, or the veto logic.
---

# SharpEdge Decision Gate

The decision brain, **validated against code**. Two layers:
daily **compounding engine** (`scripts/build_execution_state_daily.py`) →
live **decide()** (`SharpEdge-Robinhood-Bridge/.../trade_intent.py`).
**Interpretation/planning only. Every live order is operator-confirmed (Tier C).**

## Layer 1 — the compounding engine (daily)

`build_execution_state_daily.decide(row)` compounds the daily inputs into one
`execution_score` (0–100) + a `final_bias` state. Inputs: `prob_trend_fused`,
`prob_range_fused`, `dealer_state_hint`, `dist_to_wall_pct`, `compression_flag`,
`signal_strength`, and the intraday sign carriers `vwap_proxy`, `ret_open_to_cutoff`.

**Score (key rule):** base `prob_trend·70`, **minus `RANGE_DRAG(40)·max(0, prob_range−prob_trend)`**
— range probability is compounded IN (it used to be computed then ignored). Plus
dealer/compression/ignition nudges, clipped 0–100.

**`final_bias` state machine (direction-aware):**
- near wall (`dist≤0.25`) or dealer `pin` → **`PIN_FADE`**
- `prob_trend ≥ PROB_TREND_STRONG(0.65)` → **`EXPANSION_FOLLOW_LONG/SHORT`**
  (direction = `vwap_proxy` sign, tiebreak `ret_open_to_cutoff`; `EXPANSION_FOLLOW`
  if sign unavailable)
- range-dominant (`prob_range ≥ 0.60` and `prob_range−prob_trend ≥ 0.15`) → **`RANGE_FADE`**
- `prob_trend ≤ PROB_TREND_WEAK(0.45)` → **`WHIP_WAIT`**
- else **`BALANCED_SMALL`**

> Directional states only fire when the prob model produces conviction. If they
> never fire, suspect the model is censored — see `sharpedge-feature-engineering`
> (the L2=1.0 → 0.01 fix). High-conviction days are genuinely rare (~0.5%).

## Layer 2 — the live decision: `decide(signal, analytics=None)`

Dispatch on `gamma_regime`. **Always emits exactly ONE option contract** (never
equity), premium-priced limit (`build_option_leg` → BS premium ×(1+markup); risk
math prices off premium ×100, NOT the underlying).

**Negative gamma → `_runner_decision` (breakout → long):**
volume ≥ `MIN_VOL_MULT`; not pinned to a wall; `vs_vwap>MIN` AND `mom15>MIN` →
**long ATM call**.

**Positive gamma → `_fade_decision` (sticky → mean-revert) — THE validated trade:**
price must be AT an edge (`within FADE_EDGE_PCT(0.30)` of a wall). At/near the
**call wall → fade SHORT with a PUT**; at/near the **put wall → fade LONG with a
CALL**. Mid-range = nothing to fade → stand down. Conviction inputs (all validated
this session):
- **ROOM** gate: need `>= FADE_MIN_ROOM_PCT(0.08)` spot→magnet distance, else
  'nothing to fade'.
- **COIL**: tight intraday channel (`micro.ch_width_pct <= FADE_COIL_MAX_WIDTH_PCT
  0.30`) STRENGTHENS the fade — SPY coils FADE (dealer long-gamma suppresses
  breakouts; tested 3 ways). Coils are a fade signal, never a breakout.
- **MAGNITUDE-aware grade**: `HIGH` only if coiled AND the expected rest-of-day
  move (`magnitude.exp_move_realized_pct`) can actually reach the magnet; else
  `standard`. Target = the pin/magnet.

> This is the system's edge: structural mean-reversion, not direction/breakout.
> Direction is ~unforecastable (0.52); magnitude is forecastable (IC 0.40). Make
> the fade good; don't chase the runner-long directional call (shakiest part).

**Unknown regime → stand down.**

## The analytics veto — TIGHTEN ONLY, and symmetric

> **Design law: analytics can only veto/annotate a trade, never create one. Stale
> analytics (>max_age_days) is ignored WITH A NOTE, never silently trusted.**

- Runner long is **vetoed** if fresh and `prob_range − prob_trend > RANGE_VETO_MARGIN(0.20)`
  (a range day → don't chase the breakout). `_range_favored()`.
- Fade is **vetoed** if fresh and `prob_trend − prob_range > TREND_VETO_MARGIN(0.20)`
  (a trend day → the edge is likelier to break than revert). `_trend_favored()`.

This is why a fresh `RANGE_FADE` daily read *allows* a fade (it agrees) but kills a
runner long.

## Gotchas / invariants

- **One option contract, always.** No equity path exists in `decide()`. If you see
  `asset="equity"`, it's a regression.
- **Limit prices off premium, not spot.** A spot-priced option limit trips the
  notional ceiling instantly ($75k vs ~$30).
- **Backtests must be hermetic.** `backtest.grade_signal` passes an explicit
  *unavailable* AnalyticsContext when none is given — never let `decide()` reach
  into the live execution-state file during a backtest (lookahead).
- Tests: `tests/test_trade_intent.py` (runner+fade+veto), `tests/test_backtest.py`.
  Run with the system python; keep them green before committing.
