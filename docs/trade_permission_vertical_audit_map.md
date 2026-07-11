# Trade Permission Vertical Audit Map

Scope: **historical audit map** for `sharpedge.trade_permission.v1` cockpit math.
This document does **not** change scoring behavior, weights, UI rendering, or tests.

Status note, 2026-07-02: this map predates removal of standalone `candle_score`.
Use `docs/execution_vector_formula_taxonomy_audit.md` and
`cockpit/execution_vector_taxonomy.py` as the current taxonomy authority.

Related files:

- Wrapper: `cockpit/trade_permission.py`
- Score engine: `cockpit/execution_vector_engine.py`
- Context helpers: `cockpit/trade_permission_context.py`
- Weight doctrine: `cockpit/execution_vector_weights.py`
- UI table/render order: `cockpit/live_read_view.py`
- Labels/thresholds: `cockpit/execution_vector_primitives.py`

## Aggregation rules

### 1) Public entry point

- `score_trade_permission(...)` in `cockpit/trade_permission.py`
- Delegates to `ExecutionVectorEngine.build_card(...)`

### 2) Part construction order

`ExecutionVectorEngine.build_parts(...)` constructs the current visible verticals in this order:

1. `structure_score`
2. `acceptance_score`
3. `rejection_score`
4. `trend_score`
5. `volume_score`
6. `location_score`
7. `pressure_score`
8. `time_of_day_score`
9. `volatility_score`
10. `candle_score`
11. `opening_auction_score`
12. `exhaustion_score`
13. `trap_score`
14. `dealer_gamma_score`
15. `regime_score`
16. `compression_score`
17. `balance_context_score`

That insertion order is preserved by `serialize_parts(...)`, and `render_permission_section(...)`
renders the table by iterating `permission["scores"]` in that same order.

### 3) Weighted score rule

`_weighted_score(...)` uses `DEFAULT_BASE_WEIGHTS`, but with one important seam:

- the **score bucket named `acceptance_score`** is not just raw acceptance
- it is:

```python
max(
    parts["acceptance_score"].score,
    parts["rejection_score"].score,
    parts["trap_score"].score,
)
```

So:

- `rejection_score` and `trap_score` do **not** have standalone score weights
- but they can still affect the final permission score **indirectly** by winning that max()
- both still remain separately visible in the UI table

### 4) Weighted bias rule

`_weighted_bias(...)` uses `DEFAULT_BASE_BIAS_WEIGHTS` directly on the named parts.
That means:

- `rejection_score` contributes directly to bias
- `trap_score` contributes directly to bias
- `acceptance_score` contributes directly to bias
- all three can be visible at once even though score aggregation partially merges them

Special case:

- `compression_score` is skipped from bias aggregation when its reason is
  `"no compression/coil read"`

### 5) Label thresholds

From `cockpit/execution_vector_primitives.py`:

- Gate label:
  - `PERMIT` if score `>= 72`
  - `CAUTION` if score `>= 58`
  - `BLOCK` otherwise
- Bias label:
  - `CALLS` if weighted bias `>= 0.20`
  - `PUTS` if weighted bias `<= -0.20`
  - `NEUTRAL` otherwise

---

## Vertical map

Legend for **Affects**:

- **Score** = direct contribution to `_weighted_score(...)`
- **Bias** = direct contribution to `_weighted_bias(...)`
- **Both** = direct contribution to both
- **Both\*** = direct bias contribution, plus **indirect** score contribution through the acceptance/rejection/trap `max(...)` bucket

| Vertical | Source function | Score wt | Bias wt | Formula summary | Overlap cluster | Affects |
|---|---|---:|---:|---|---|---|
| `structure_score` | `ExecutionVectorEngine._score_structure()` | 0.15 | 0.10 | Compares latest swing highs/lows for HH/HL vs LH/LL; fallback compares first-half vs second-half session structure. | Structure / Trend / Regime | Both |
| `acceptance_score` | `ExecutionVectorEngine._score_acceptance()` | 0.15 | 0.12 | Looks for 3 closes accepted above/below the nearest full level with buffer; fallback uses `vs_vwap`. | **Acceptance / Rejection / Trap** | Both |
| `rejection_score` | `ExecutionVectorEngine._score_rejection()` | — | 0.10 | Reads last-candle wick rejection using wick/body ratio and close position in range. | **Acceptance / Rejection / Trap**, Candle | **Both\*** |
| `trend_score` | `ExecutionVectorEngine._score_trend()` | 0.15 | 0.14 | Uses recent 6-close slope + `vs_vwap` + `mom15`; favors aligned VWAP control and short-term momentum. | Trend / Regime / Structure | Both |
| `volume_score` | `ExecutionVectorEngine._score_volume()` | 0.10 | — | Thresholds on `vol_mult` (`>=1.5`, `>=1.0`, `>=0.7`, else weak). | Volume / Pressure / Regime | Score |
| `location_score` | `ExecutionVectorEngine._score_location()` | 0.15 | — | Prefers proximity to nearest decision level; fallback uses balance state, balance position, and day-range position. | Location / Balance / Exhaustion / Regime | Score |
| `pressure_score` | `ExecutionVectorEngine._score_pressure()` | 0.08 | 0.05 | Uses last 4 bars: directional closes, close position, `vol_mult`, and displacement through recent range. | Pressure / Trend / Candle / Volume | Both |
| `time_of_day_score` | `ExecutionVectorEngine._score_time_of_day()` | 0.05 | — | Minutes since open bucketed into opening auction, morning continuation, midday chop, power hour, or neutral window. | Time / Opening Auction | Score |
| `volatility_score` | `ExecutionVectorEngine._score_volatility()` | 0.05 | — | Uses ATM IV regime plus `premium_read` (`cheap` / `rich`) to adjust follow-through quality. | Volatility / Dealer Gamma / Compression | Score |
| `candle_score` | `ExecutionVectorEngine._score_candle()` | 0.05 | 0.07 | Uses last-candle close position and wick/body ratios for strong close or wick-rejection reads. | Candle / Rejection / Pressure | Both |
| `opening_auction_score` | `score_opening_auction(...)` + `_opening_auction_decay(...)` | 0.10 | 0.04 | Reads gap vs prior close (`PDC`) and whether it is accepting, failing, reclaiming, or unresolved; then decays its influence later in the session. | Opening Auction / Time / Acceptance | Both |
| `exhaustion_score` | `ExecutionVectorEngine._score_exhaustion()` | 0.07 | 0.07 | Uses distance from VWAP + EMA20, range extreme, wick rejection, and OR proximity to detect stretch/exhaustion. | Exhaustion / Location / Candle / Regime | Both |
| `trap_score` | `ExecutionVectorEngine._score_trap()` | — | 0.09 | Searches recent failed breaks around ORH/ORL/PDH/PDL and checks for close back through the level. | **Acceptance / Rejection / Trap** | **Both\*** |
| `dealer_gamma_score` | `ExecutionVectorEngine._score_dealer_gamma()` | 0.05 | 0.06 | Uses gamma regime, pin distance, wall proximity, and `premium_read`; positive gamma suppresses, negative gamma favors expansion. | Dealer Gamma / Regime / Volatility | Both |
| `regime_score` | `ExecutionVectorEngine._score_regime()` | 0.12* | 0.08* | Uses VWAP relation, `mom15`, `vol_mult`, range position, and session drift to classify balance vs trend-day behavior. | Regime / Trend / Location / Volume | Both |
| `compression_score` | `score_compression(...)` | 0.07 | 0.15 | Uses `volatility_structure` / coil state / trigger proximity to rate squeeze, contraction, expansion, or post-selloff coil setups. | Compression / Volatility / Setup State | Both |
| `balance_context_score` | `score_balance_context(...)` | 0.08 | 0.12 | Uses balance confluence, disagreement, and flip state from `pa`; disagreement caps score and forces neutral bias. | Balance / Location / Regime | Both |

Notes:

- `regime_score` weights are conditionally reduced by `_regime_weight_multiplier(...)` when trend and regime already agree. In the aligned case, the multiplier is `0.6` for both score and bias weight maps.
- `compression_score` contributes to bias only when it has a real read. If its reason is `"no compression/coil read"`, `_weighted_bias(...)` skips it.

---

## Special audit note: acceptance / rejection / trap

This is the most important overlap cluster in the current math.

### Current behavior

#### Visible in UI as separate rows

All three are independently serialized and rendered:

- `acceptance_score`
- `rejection_score`
- `trap_score`

#### Score aggregation

Only **one** score bucket is used in `_weighted_score(...)`:

- the bucket named `acceptance_score`
- but its numeric value is `max(acceptance, rejection, trap)`

That means:

- raw `rejection_score` can raise total permission score
- raw `trap_score` can raise total permission score
- even though neither has its own direct score weight column

#### Bias aggregation

In `_weighted_bias(...)`, all three participate separately:

- `acceptance_score` with bias weight `0.12`
- `rejection_score` with bias weight `0.10`
- `trap_score` with bias weight `0.09`

### Audit consequence

The current model is **not** a simple one-row-per-math-axis system for this cluster.
It is a hybrid:

- **score side:** partially merged
- **bias side:** separate
- **UI side:** separate

That is the key financial-math seam to preserve and audit before any future refactor.

### Intentionality status

This asymmetry is **not automatically wrong**.

It may be intentional if the design goal was:

- treat acceptance / rejection / trap as one shared **permission-strength** bucket on the score side
- while still letting their directional readouts remain separately expressive on the bias side

But the current code does **not** carry an explicit doctrine comment proving that intent.
So the safe audit posture is:

- **do not change it casually**
- **do not assume it is accidental**
- **do treat it as the first investigation target** if score/bias alignment is ever questioned

---

## Render-order vs math-order note

The visible table order comes from `build_parts(...)` insertion order, not from weight rank,
bias rank, or importance rank.

So the UI table is best read as:

- **"feature inventory in construction order"**

not:

- **"importance order"**
- **"weight order"**
- **"best-to-worst order"**

The separate “Top reasons to trade / wait” blocks come from sorting by **current score**, not from table order.

---

## Step-by-step investigation sequence

### Step 1 — acceptance / rejection / trap

Status: **traced**

Confirmed current behavior:

- all three are constructed as separate visible rows
- `_weighted_score(...)` partially merges them by using
  `max(acceptance, rejection, trap)` inside the `acceptance_score` score bucket
- `_weighted_bias(...)` keeps all three separate and weighted independently

Why this stays the first scrutiny target:

- score and bias do not follow the same grouping rule here
- that may be intentional, but current code does not prove intent explicitly
- future changes should not "clean this up" without deciding whether the asymmetry is doctrine or drift

### Step 2 — trend / regime

Status: **traced**

Confirmed current behavior:

- `trend_score` is a **short-horizon directional alignment** read
  - inputs: recent 6-close slope, `vs_vwap`, `mom15`
  - weights: score `0.15`, bias `0.14`
- `regime_score` is a **session-shape / tape-state** read
  - inputs: `vs_vwap`, `mom15`, `vol_mult`, `rng_pos`, first-half vs second-half drift
  - weights: score `0.12`, bias `0.08`
- both therefore share meaningful inputs:
  - `vs_vwap`
  - `mom15`
  - directional drift logic

Current anti-double-count seam:

- `_regime_weight_multiplier(...)` reduces `regime_score` weight to `0.6x`
- but **only** when:
  - `trend_score.bias` is non-neutral
  - `regime_score.bias` is non-neutral
  - and both biases agree

That means the effective `regime_score` weights become:

- score: `0.12 -> 0.072`
- bias: `0.08 -> 0.048`

Important limit of the current mitigation:

- if either side is `NEUTRAL`, the multiplier stays `1.0`
- so overlapping **neutral / chop / unclear** reads are **not** de-weighted
- if the two lenses disagree, the multiplier also stays `1.0`

Audit read:

- this is a **partial** anti-double-count seam, not a full conceptual separation
- the current code distinguishes:
  - **trend** = local directional alignment
  - **regime** = broader session condition
- but in practice they still share enough ingredients that they deserve continued scrutiny

Current safe conclusion:

- the overlap is **mitigated**, not eliminated
- the multiplier is real and meaningful
- but it is not broad enough to prove the two lenses are fully independent

### Step 3 — rejection / candle / pressure

Status: **traced**

Confirmed current behavior:

- `rejection_score` is a **pure last-bar wick rejection** read
  - inputs: `bar_personality(last_bar)`
  - uses: `body`, `upper_wick`, `lower_wick`, `close_pos`
  - output style: yes/no style rejection call with strong directional bias
  - weights: score `—` (indirect via acceptance bucket), bias `0.10`
- `candle_score` is a **pure last-bar candle personality** read
  - inputs: `bar_personality(last_bar)`
  - uses: same four fields as rejection
  - output style: strong close, wick rejection, or ordinary candle
  - weights: score `0.05`, bias `0.07`
- `pressure_score` is a **4-bar sequence / follow-through** read
  - inputs: last 4 closes, last-bar `close_pos`, `vol_mult`, range displacement
  - output style: persistent buying/selling pressure vs mixed pressure
  - weights: score `0.08`, bias `0.05`

Overlap read:

- `rejection_score` and `candle_score` are the closest pair
  - both use the exact same primitive: `bar_personality(last_bar)`
  - both evaluate wick/body structure on the same final bar
  - both can fire bullish/bearish on the same wick event
- `pressure_score` is more distinct
  - it still uses the last bar's `close_pos`
  - but it adds multi-bar persistence, directional close count, displacement, and volume confirmation
  - so it is not just a second copy of candle personality

Important absence:

- there is **no explicit anti-double-count seam** here like the trend/regime multiplier
- there is also **no grouping seam** here like acceptance/rejection/trap
- so any separation depends entirely on the conceptual difference in the underlying formulas

Audit read:

- `rejection_score` and `candle_score` are only weakly separated doctrinally
  - rejection = wick rejection lens
  - candle = broader bar-quality lens
- but in practice the current implementations overlap heavily because both are driven by the same last-bar wick/body anatomy
- `pressure_score` is the most defensibly separate member of this cluster

Current safe conclusion:

- this cluster is **not fully duplicate**, because pressure brings sequence and displacement
- but `rejection_score` vs `candle_score` remains a meaningful overlap seam
- if a future cleanup happens, this pair deserves scrutiny before pressure does

### Step 4 — location / balance context / exhaustion

Next question:

- are structural placement, balance placement, and stretch/exhaustion acting as complementary lenses,
  or are they compressing into one overcrowded "where are we?" family?

---

## Quick code-reference index

- Wrapper: `cockpit/trade_permission.py`
- Weight doctrine: `cockpit/execution_vector_weights.py`
- Parts build: `cockpit/execution_vector_engine.py`, `build_parts(...)`
- Score aggregation: `cockpit/execution_vector_engine.py`, `_weighted_score(...)`
- Bias aggregation: `cockpit/execution_vector_engine.py`, `_weighted_bias(...)`
- Regime anti-double-count seam: `cockpit/execution_vector_engine.py`, `_regime_weight_multiplier(...)`
- Compression helper: `cockpit/trade_permission_context.py`, `score_compression(...)`
- Opening auction helper: `cockpit/trade_permission_context.py`, `score_opening_auction(...)`
- Balance helper: `cockpit/trade_permission_context.py`, `score_balance_context(...)`
- UI render order: `cockpit/live_read_view.py`, `render_permission_section(...)`
- Label thresholds: `cockpit/execution_vector_primitives.py`, `gate_label(...)`, `bias_label(...)`
