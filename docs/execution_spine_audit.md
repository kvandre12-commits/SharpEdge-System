# SharpEdge Execution Spine Audit

Date: 2026-07-11

## Scope

This audit covers the canonical **core execution spine** packaged in `cockpit/execution_hierarchy.py`:

- `structure_score`
- `acceptance_score`
- `trend_score`
- `location_score`
- `volume_score` (displayed as Participation)
- `time_of_day_score`
- `dealer_gamma_score`

It also audits:

- how these parts are built in `cockpit/execution_vector_engine.py`
- how they are weighted in `cockpit/bucket_conditioned_spine.py`
- how they become permission in `cockpit/execution_card_builder.py`
- where measured facts end and interpretation/authority begins

## Primary code paths reviewed

- Raw price/options inputs:
  - `cockpit/market_data_sources.py`
  - `cockpit/make_cockpit.py`
  - `cockpit/volume_profile.py`
  - `cockpit/gamma.py`
- Spine scoring:
  - `cockpit/execution_vector_engine.py`
  - `cockpit/execution_location_score.py`
  - `cockpit/trade_permission_context.py`
  - `cockpit/execution_vector_context.py`
  - `cockpit/execution_vector_primitives.py`
  - `cockpit/execution_vector_weights.py`
  - `cockpit/execution_vector_taxonomy.py`
- Packaging / authority:
  - `cockpit/execution_hierarchy.py`
  - `cockpit/day_bucket.py`
  - `cockpit/bucket_conditioned_spine.py`
  - `cockpit/execution_card_builder.py`
  - `cockpit/trade_permission.py`
- Tests reviewed:
  - `tests/test_trade_permission.py`
  - `tests/test_volume_profile.py`
  - `tests/test_execution_location_score.py`
  - related execution/view tests

---

## Executive findings

1. **The spine is real, but not fully purified.**
   Several verticals measure what they claim, but some leak into neighboring concepts.

2. **Structure is only partially “structure.”**
   It mostly scores swing sequence and half-session HH/HL or LH/LL behavior. It does **not** currently measure reclaims, failed breaks, compression, or explicit acceptance.

3. **Acceptance is partly auction, partly context blend.**
   It checks 3 closes above/below the nearest mapped level, but the mapped level set includes `VWAP`, `CALL_WALL`, and `PUT_WALL` via `execution_vector_context.level_map()`. That means acceptance is not purely auction acceptance around classic price references.

4. **Trend is not independent.**
   It is VWAP + 15-minute momentum + 6-bar slope. That is a clean momentum read, but it overlaps heavily with structure and acceptance. The repo already admits this in `execution_vector_taxonomy.py`.

5. **Location is the dirtiest core vertical.**
   It should answer “where are we?” but one of its sub-branches uses `vs_vwap` and `mom15`, which means trend leaked into location.

6. **Participation is improved but still incomplete.**
   `volume_score` correctly measures confirmation/participation, not all possible fuel. The newer expansion-fuel surface exists specifically because low volume can still coincide with large moves for non-volume reasons.

7. **Time of day is a governor, not a signal engine.**
   In code it behaves as session-quality context, not a directional signal. That is probably the correct boundary.

8. **Dealer/gamma is the most mixed vertical.**
   Parts are measured from delayed options chain data, but the key behavioral meaning is inferred. Worse: current `dealer_gamma_score` also uses `premium_read`, so it is not a pure dealer-positioning vertical.

9. **The 0-100 scale is mostly doctrinal, not calibrated.**
   The numbers are discrete hand-authored buckets. `72` and `58` matter because `gate_label()` uses them for `PERMIT` and `CAUTION`, not because an empirical model proved those exact cutoffs.

10. **Predictive validation is thin.**
    I found unit tests proving deterministic behavior and doctrinal intent. I did **not** find a cockpit-local backtest/calibration framework for the core spine verticals themselves. The one explicit out-of-sample validation note I found is for `read_magnitude()` in `make_cockpit.py`, which is not one of the seven core spine verticals.

---

## Data chain: raw input -> feature -> fact -> interpretation -> score -> permission

| Vertical | Raw input | Derived feature | Market fact | Interpretation | Execution impact |
|---|---|---|---|---|---|
| Structure | 1m OHLC bars | swing highs/lows, session-half highs/lows | sequence or mixed structure | tape is orderly or messy | weighted core spine input |
| Acceptance | 1m closes + mapped levels | 3-close stack above/below level with buffer | accepted above/below some level | auction is accepting or not | weighted core spine input |
| Trend | closes + VWAP-derived `vs_vwap` + `mom15` | 6-bar slope, VWAP relation, 15m momentum | short drift aligned or not | continuation quality | weighted core spine input |
| Location | spot + levels + balance fields | nearest level distance, balance position, range extreme | near edge / in middle / stretched | where price sits in the battlefield | weighted core spine input |
| Participation | OHLCV bars | local/session volume mult, aligned share, path efficiency | move is confirmed / participating / mixed / missing | whether participation confirms the move | weighted core spine input |
| Time of day | session minute | time bucket | opening / morning / midday / power hour | execution environment quality | weighted core spine input |
| Dealer/gamma | delayed options chain + spot + premium read | regime, pin distance, wall proximity, cheap/rich premium | positive/negative gamma context | damped vs expansive tape | weighted core spine input |

The chain is real, but several verticals still blend **market fact** and **interpretation** too early.

---

## Core spine authority boundary

### Descriptive layer
Each vertical returns `ScorePart(score, bias, reason)`.

### Authority layer
The parts are not merely displayed. They are fed into:

- `ExecutionVectorEngine._weighted_score()`
- `build_bucket_conditioned_spine()`
- `gate_label()` -> `BLOCK` / `CAUTION` / `PERMIT`
- `recommended_action` -> `stand_down` / `watch_edges` / `candidate_calls` / `candidate_puts`

### Conclusion
Each vertical is **descriptive evidence**, but not purely passive. Once weighted, it directly affects trading permission. So the audit standard must be higher than “sounds plausible.”

---

# Vertical-by-vertical audit

## 1) Structure

**Code:** `ExecutionVectorEngine._score_structure()`

### Purpose
Measure price structure / sequence quality.

### Source
Only 1-minute OHLC bars.

### Features
- `ctx.swing_points(self.bars, window=2)`
- latest two swing highs
- latest two swing lows
- fallback: compare high/low behavior between first and second session halves

### Logic
- HH + HL -> `82 / BULLISH`
- LH + LL -> `82 / BEARISH`
- mixed swings -> `46 / NEUTRAL`
- not enough swing points but session halves trend up/down -> `70 / directional`
- otherwise -> `45 / NEUTRAL`

### Score behavior
- `82`: strong clean sequence
- `70`: weaker fallback trend-like structure
- `46/45`: mixed or insufficient structure
- There is no fine-grained structural continuum here; it is a branchy state machine wearing a score costume.

### Validation found
- Unit tests indirectly cover resulting permission behavior.
- No direct predictive calibration for structure quality found.

### Failure modes
- Swing-point detection is coarse and sensitive to windowing.
- Can miss valid intraday structure if the tape is smooth rather than pivot-heavy.
- Fallback session-half logic is trend-ish, so structure degrades into drift detection when swing points are sparse.

### Dependencies
- Only bars.

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**Is Structure really measuring structure, or have trend and momentum leaked into it?**
- Mostly structure in the primary branch.
- But the fallback branch (`session halves show higher high/higher low`) is already a mild trend proxy.
- It does **not** currently include reclaims, failed breaks, acceptance, or compression.

### Verdict
Keep it, but rename expectations honestly:
- current structure = **sequence quality**, not full market structure.

---

## 2) Acceptance

**Code:** `ExecutionVectorEngine._score_acceptance()`

### Purpose
Measure auction acceptance around levels.

### Source
- recent 3 closes from bars
- `self.full_levels`

### Important dependency detail
`self.full_levels` comes from `execution_vector_context.level_map()` and includes:
- classical levels: ORH/ORL/PDH/PDL/PDC
- `VWAP`
- `CALL_WALL`
- `PUT_WALL`

That means acceptance is not limited to price-structure references.

### Features
- `ctx.recent_closes(self.bars, 3)`
- nearest candidate levels sorted by proximity to last close
- `buffer_for_price(level)`

### Logic
For candidate levels, first one that fits wins:
- all 3 closes > level + buffer -> `78 / BULLISH`
- all 3 closes < level - buffer -> `78 / BEARISH`
- otherwise fallback:
  - above VWAP -> `60 / BULLISH`
  - below VWAP -> `60 / BEARISH`
  - no clean acceptance -> `35 / NEUTRAL`

### Score behavior
- `78`: clean multi-close acceptance
- `60`: directional lean via VWAP, without stacked level acceptance
- `35`: no clean acceptance

### Validation found
- Tests assert bullish acceptance can score high and affect permission.
- No direct predictive calibration found.

### Failure modes
- Because `full_levels` includes VWAP and walls, acceptance can silently absorb trend/location/dealer context.
- “First matching nearest level wins” can cause path dependence.
- Acceptance may say “accepted above” a dealer wall or VWAP when that is semantically different from accepting above ORH/PDH.

### Dependencies
- Bars
- VWAP from `read_price_action()`
- options walls from `read_options_surface()`

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**Does this belong here or inside Structure?**
- It belongs separate from Structure.
- But it should be tightened to pure auction acceptance around a more explicit reference set.

### Verdict
Acceptance is a valid independent concept, but its current implementation is too permissive about what counts as a reference.

---

## 3) Trend

**Code:** `ExecutionVectorEngine._score_trend()`

### Purpose
Measure short-term directional drift / momentum alignment.

### Source
- closes from bars
- `pa['vs_vwap']`
- `pa['mom15']`

### Features
- recent 6 closes
- slope = `recent[-1] - recent[0]`
- VWAP relation
- 15-minute momentum

### Logic
- `vs_vwap > 0.05` and `mom15 > 0` and `slope > 0` -> `82 / BULLISH`
- mirror bearish branch -> `82 / BEARISH`
- hugging VWAP -> `38 / NEUTRAL`
- otherwise directional lean -> `58 / directional`

### Score behavior
- `82`: strong alignment
- `58`: partial alignment
- `38`: chop / VWAP magnet

### Validation found
- Tests confirm trend can stay strong while pressure stays mixed.
- No direct predictive calibration found.

### Failure modes
- Highly correlated with acceptance and structure.
- No EMA usage, despite people often mentally mapping this vertical to “trend + EMAs.” That is **not** what current code does.
- Very short-horizon; can overreact to recent drift.

### Dependencies
- bars
- VWAP
- mom15

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**Is Trend independent, or just another expression of Structure?**
- Not independent.
- It is a separate momentum-style read, but it overlaps heavily with structure and acceptance.
- The repo’s own taxonomy labels it `MOMENTUM` with overlap into `PRICE_STRUCTURE`.

### Verdict
Trend is legitimate, but it is best understood as **short-horizon momentum/VWAP alignment**, not some independent truth source.

---

## 4) Location

**Code:** `cockpit/execution_location_score.py`

### Purpose
Measure where price is relative to actionable geography.

### Source
- `pa['spot']`
- `full_levels`
- balance fields from `build_balance_stack()`
- `rng_pos`
- also `vs_vwap` and `mom15` in one branch

### Features
Three ordered sub-reads:
1. edge proximity
2. balance position
3. stretch state

### Logic
Priority order is explicit:
- near level <= 0.08% -> `82 / NEUTRAL`
- near level <= 0.20% -> `68 / NEUTRAL`
- above/below balance with confirming `vs_vwap` + `mom15` -> `74 / directional`
- above/below balance without confirmation -> `52 / NEUTRAL`
- top/bottom of balance -> `58 / directional counter-lean`
- middle of balance -> `36 / NEUTRAL`
- range extreme -> `58 / NEUTRAL`
- middle of nowhere -> `34 / NEUTRAL`

### Score behavior
This is not one coherent formula. It is a prioritized cascade.

### Validation found
- Dedicated location tests exist.
- They verify deterministic composition order, not predictive power.

### Failure modes
- The balance-position branch uses `vs_vwap` and `mom15`, which means trend leaked into location.
- “Near level” is neutral, even when the level itself is directional in context.
- Range-extreme logic overlaps stretch/exhaustion.

### Dependencies
- price action
- balance engine
- mapped levels
- VWAP/momentum

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**Is this measuring where we are rather than what price is doing?**
- Not consistently.
- Edge proximity: yes.
- Balance position: partly.
- Directional acceptance above/below balance: no, because that is partly what price is doing.

### Verdict
Location is the best candidate for cleanup.
It should be purified to geography and stop borrowing `vs_vwap` and `mom15`.

---

## 5) Participation / Volume

**Code:**
- `volume_profile.build_volume_profile()`
- `ExecutionVectorEngine._score_volume()`

### Purpose
Measure whether participation confirms the move.

### Source
Only OHLCV bars.

### Features
From `build_volume_profile()`:
- recent 5-bar average volume
- session median volume
- local baseline median volume
- 15-bar move direction
- aligned volume share
- path efficiency
- composite participation multiple

### Logic
Volume profile classification:
- confirmed
- participating
- mixed
- missing

Spine score mapping:
- confirmed -> `85`
- participating -> `64`
- mixed -> `42`
- missing -> `25`
- fallback also exists using raw `vol_mult`

### Score behavior
- `85`: participation confirms move
- `64`: good enough, not dominant
- `42`: mixed
- `25`: absent

### Validation found
- Strongest validation among the core verticals is here, but still only deterministic/relative tests.
- `tests/test_volume_profile.py` confirms stronger aligned participation yields higher score and higher permission than weak participation.
- No return-based calibration found.

### Failure modes
- Low volume can still accompany large moves due to thin liquidity, dealer chasing, stop cascades, or structural acceptance.
- Current spine treats weak participation as weaker conviction, which is reasonable, but it is not the whole story.
- The newer expansion-fuel lane exists because participation alone does not explain travel.

### Dependencies
- bars only

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**Can low volume still produce big moves? If so, how should that affect scoring?**
- Yes.
- Current doctrine already admits this through `execution_expansion_potential` and the “participation confirmation vs expansion fuel” split.
- Therefore volume should stay a **confirmation** vertical, not a total explanation vertical.

### Verdict
This vertical is conceptually right. The remaining work is doctrinal: keep it explicitly in the confirmation lane and never let it pretend it explains all expansion.

---

## 6) Time of day

**Code:** `ExecutionVectorEngine._score_time_of_day()`

### Purpose
Measure session-window quality for execution.

### Source
Only session minute derived from bar timestamp index.

### Features
- minutes since open
- fixed session buckets

### Logic
- first 30 min -> `52 / NEUTRAL`
- first 2 hours -> `74 / NEUTRAL`
- midday -> `42 / NEUTRAL`
- >= 330 min -> `68 / NEUTRAL`
- otherwise -> `58 / NEUTRAL`

### Score behavior
Always neutral bias. This is a pure conviction modifier.

### Validation found
- Tests confirm opening-auction context decays later in the day, but that is a different governor (`opening_auction_score`).
- No predictive calibration for time buckets found.

### Failure modes
- Hard-coded session windows may not generalize across event days, OPEX, FOMC, etc.
- By itself it cannot distinguish good power-hour trend from sloppy headline chop.

### Dependencies
- bar minute only

### Authority status
- Descriptive governor that feeds permission.

### Audit answer
**Should Time modify conviction, or should it generate signals?**
- In current code it modifies conviction.
- That is the correct boundary.
- It should remain a governor unless there is robust validation for time-specific signal families.

### Verdict
Keep as governor, not signal generator.

---

## 7) Dealer / Gamma

**Code:**
- measured packet: `gamma.gamma_profile()`
- options surface: `market_data_sources.read_options_surface()`
- spine score: `ExecutionVectorEngine._score_dealer_gamma()`

### Purpose
Measure dealer-positioning context: pinning vs expansion, plus wall gravity.

### Source
Delayed CBOE options chain:
- per-strike gamma
- open interest
- bid/ask
- IV
- spot from CBOE current price/close fallback

### Measured components
From `gamma_profile()`:
- nearest expiry
- `net_gamma = sum(call_gamma*OI - put_gamma*OI)`
- `pin = strike with max(call_gamma*OI + put_gamma*OI)`
- `max_pain`
- `pin_dist`
- gamma data quality

From `read_options_surface()`:
- `call_wall` = highest call OI strike >= spot
- `put_wall` = highest put OI strike <= spot
- ATM IV and related option fields

### Inferred components
- positive gamma -> dealers dampen moves / chop risk
- negative gamma -> dealers amplify moves / expansion risk
- call wall near spot -> upside resistance
- put wall near spot -> downside support
- pin as a “magnet”

These are not directly measured. They are market-structure interpretations layered on measured options fields.

### Spine features actually used
`_score_dealer_gamma()` uses:
- `gp['regime']`
- `gp['pin']`
- `op['call_wall']`
- `op['put_wall']`
- `magnitude['premium_read']`

### Important contamination
This vertical is **not pure dealer positioning** because it also uses `premium_read`, which comes from `read_magnitude()` in `make_cockpit.py`.
That `premium_read` is a realized-vs-implied move comparison, not dealer positioning.

### Logic
- positive gamma + near pin (<= 0.25%) -> `38`
- negative gamma -> `72` if cheap premium else `62`
- otherwise if pin distance known -> `55`
- else -> `50`

Wall proximity sets bias but does not dominate score.

### Score behavior
- `38`: pinning / chop risk
- `72` or `62`: negative gamma expansion context
- `55`: mild dealer gravity context
- `50`: no read

### Validation found
- Tests check that positive gamma pinning dampens score and that reasons mention pinning.
- No historical validation of gamma regime / wall / pin logic found inside cockpit tests.

### Failure modes
- Delayed options data can be stale intraday.
- Dealer positioning is inferred from public options chain and conventions, not observed directly.
- `net_gamma` sign can oversimplify distributional effects across strikes.
- `pin` and `max_pain` are useful abstractions, but they are not guaranteed live magnets.
- Mixing `premium_read` into this vertical blurs the boundary between dealer positioning and option-pricing richness.

### Dependencies
- delayed options chain
- magnitude model (`premium_read`)
- spot

### Authority status
- Descriptive input that feeds permission.

### Audit answer
**What part is measured, and what part is inferred?**
- Measured: chain fields, OI, gamma, IV, strikes, spot, ATM IV, walls by OI.
- Derived but objective: net gamma sign, pin, max pain, pin distance.
- Inferred: dealer damping/amplification behavior, wall directional meaning, magnet interpretation, cheap/rich effect inside this vertical.

### Verdict
This deserves a split:
- **dealer_position_facts**
- **dealer_position_interpretation**
- and `premium_read` should probably move out of the dealer vertical.

---

# Scoring audit

## Current score doctrine

Scores are mostly **discrete authored buckets**, not calibrated probabilities.
Examples:
- `25` missing participation
- `35` no acceptance / weak neutral state
- `38` VWAP hug or gamma pinning
- `58` mild directional lean / caution state
- `72` permit threshold neighborhood
- `78/82/85` strong evidence buckets

## Why this is not binary
The system wants graded conviction, not pure yes/no checks.
That is reasonable.

## Why the scale is still problematic
1. Many verticals are branch-based state machines, not continuous estimators.
2. The same score can come from different market states.
3. The meanings of 58, 72, and 82 are doctrinal rather than empirically calibrated.
4. `gate_label()` hard-codes:
   - `>= 72` -> `PERMIT`
   - `>= 58` -> `CAUTION`
   - else `BLOCK`

So exact numbers matter mechanically even when they are not statistically grounded.

## Answering the calibration questions

### Why is this 72 instead of 65?
Usually because the author hand-assigned a bucket that “felt like strong enough directional evidence.” I found no cockpit-local calibration artifact proving 72 is optimal.

### Why is 90 rare?
Because the authored scale clusters strong states around 78-85 and leaves 90+ for very exceptional corroboration. This is a design style, not a demonstrated statistical distribution.

### Why isn’t it binary?
Because the card needs ranked evidence and a final weighted permission score. Binary parts would be too brittle.

### Can two different states produce the same score?
Yes, constantly.
Examples:
- very different directional contexts can both return 58
- structure and trend can both return 82 from unrelated mechanics

### Should some of these be states instead of scores?
Yes.
Best candidates:
- `time_of_day_score`
- dealer/gamma regime components
- acceptance state
- location state
- level-state and trap/reclaim semantics

A clean pattern would be:
- state first
- score second only if needed for aggregation

---

# Evidence audit by vertical

## Structure
- Raw input: OHLC bars
- Derived feature: swing highs/lows; session-half highs/lows
- Market fact: sequence is improving, degrading, or mixed
- Interpretation: price structure is orderly or messy
- Execution impact: weighted core spine feature

## Acceptance
- Raw input: closes + mapped levels
- Derived feature: 3-close stack above/below buffered level
- Market fact: recent closes are accepted relative to a reference
- Interpretation: auction is proving direction
- Execution impact: weighted core spine feature

## Trend
- Raw input: closes + VWAP + mom15
- Derived feature: 6-bar slope, VWAP relation, 15m momentum
- Market fact: short-term drift is aligned or not
- Interpretation: directional continuation quality
- Execution impact: weighted core spine feature

## Location
- Raw input: spot + levels + balance fields + range position
- Derived feature: nearest level distance, balance position, extreme detection
- Market fact: price is near edge / mid / extreme
- Interpretation: geography is favorable or not
- Execution impact: weighted core spine feature

## Participation
- Raw input: OHLCV bars
- Derived feature: local/session mult, aligned volume share, efficiency
- Market fact: participation confirms / participates / mixed / missing
- Interpretation: move has or lacks confirmation
- Execution impact: weighted core spine feature

## Time of day
- Raw input: session minute
- Derived feature: time bucket
- Market fact: session window type
- Interpretation: follow-through quality is better/worse here
- Execution impact: weighted core spine feature

## Dealer/gamma
- Raw input: options chain + spot + premium read
- Derived feature: net gamma sign, pin, wall proximity, cheap/rich premium
- Market fact: gamma regime, pin distance, wall proximity
- Interpretation: tape likely pinned vs expansive
- Execution impact: weighted core spine feature

---

# Validation audit

## What validation exists
- Deterministic unit tests for behavior and score relationships.
- Taxonomy metadata documenting overlap risk.
- One explicit OOS note exists in `read_magnitude()` comment, but that is not a core spine vertical.

## What validation does not currently exist in the cockpit layer
I did **not** find:
- vertical-by-vertical expectancy studies
- calibration curves for score buckets
- out-of-sample hit-rate reports for structure/acceptance/trend/location/participation/time/dealer-gamma
- evidence that `72` and `58` are optimal gate thresholds

## Honest conclusion
The current spine is **explainable and test-covered**, but not yet **empirically audited** at the vertical level.

---

# Dependency / leakage map

## Cleanest verticals
- Participation
- Time of day

## Moderately entangled
- Structure
- Trend
- Acceptance

## Most entangled
- Location
- Dealer/gamma

## Specific leakage issues
1. `location_score` uses `vs_vwap` and `mom15` in balance-position logic.
2. `acceptance_score` scans `VWAP`, `CALL_WALL`, `PUT_WALL`, not just reference levels.
3. `dealer_gamma_score` uses `premium_read`, which belongs to move-pricing context.
4. `structure_score` fallback branch behaves like trend.

---

# Recommended cleanup order

## Tier 1: purity fixes
1. **Purify `location_score`**
   - Remove `vs_vwap` and `mom15` from location branches.
   - Keep it about geography only.

2. **Purify `dealer_gamma_score`**
   - Remove `premium_read` from this vertical.
   - Split measured facts from dealer interpretation.

3. **Narrow `acceptance_score` reference set**
   - Separate static/session references from VWAP/walls.
   - Consider distinct acceptance families.

## Tier 2: state-first architecture
4. Convert major verticals to explicit states first, then score:
   - structure state
   - acceptance state
   - location state
   - dealer-state packet

5. Use `level_state_engine` and related facts wherever possible before interpretation layers score them.

## Tier 3: empirical audit
6. Build a vertical validation harness:
   - freeze snapshots of raw inputs
   - compute vertical states/scores
   - compare against forward intraday outcomes
   - report hit rates / calibration by bucket

7. Audit thresholds:
   - is `72` really permit?
   - should some verticals be capped harder?
   - should others be qualitative only?

---

# Bottom line

The execution spine is no longer random score soup. It has real structure.
But it is also **not yet a fully purified evidence stack**.

The biggest remaining truth problems are:
- location mixing geography with momentum
- acceptance mixing auction with VWAP/walls
- dealer/gamma mixing measured options structure with broader inference and premium richness
- score numbers behaving like calibrated values when they are mostly doctrinal buckets

So the next stage is not “add more cleverness.”
The next stage is:

**purify vertical boundaries, state-first the evidence, and only then validate the scoring empirically.**
