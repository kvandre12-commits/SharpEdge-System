# Execution vector formula taxonomy audit

Date: 2026-07-02

Update: `candle_score` has been removed from the active vector contract after
this audit identified it as the most redundant score-soup seam. Last-bar candle
anatomy remains visible through `rejection_score`, `pressure_score` overlap, and
`exhaustion_score` wick context, but it no longer has a standalone score/bias
vote.

Update: `location_score` now remains one weighted vector but is composed from
named sub-reads: edge proximity, balance position, stretch state, and no-edge
fallback. Level quality is documented as a future distinct concept because the
current inputs do not provide an independent quality signal.

Purpose: audit the red-taped execution-vector formulas against the newer
`execution_vector_taxonomy.py` contract.

This document treats the older vertical audit as evidence, not authority. The
current authority candidate is the taxonomy:

- `category` = role in the reasoning stack
- `correlation_family` = primary phenomenon measured
- `overlap_families` = known secondary correlation risk

No scoring constants, thresholds, weights, caps, floors, gates, or bias math were
changed for this audit.

## Layer invariant

```text
Market
↓
Evidence
↓
Interpretation
↓
Governance
↓
Readiness
↓
Packaging
↓
Explanation
↓
Human
```

The cockpit pipeline should be monotonic: information flows one way through
contracts. View does not influence packaging. Packaging does not influence
governance. Governance does not create evidence. Evidence does not render UI.

```text
Formula = measure phenomenon
Taxonomy = declare role + overlap risk
Hierarchy = organize evidence
Grammar/playbook = interpret and advise execution
Permission = act on interpretation
```

Formula reasons should describe what was measured. They should not say what the
operator should do next; that belongs in grammar/playbook `reason` and `needs`.

## Pipeline doctrine

```text
Vector engine produces evidence
Grammar interprets evidence
Permission governor constrains authority
Card builder packages
View explains
```

`cockpit/execution_grammar.py` currently contains both the interpreter and the
permission governor. That is acceptable while they evolve together:

- interpretation maps break/dealer state into thesis, bias, and authority
- governance maps live-trigger readiness into permission caps/floors

Do not split the file merely because these are distinct concepts. Split only if
interpreter doctrine and governor doctrine start changing independently, or if
new playbooks turn the thesis resolver into a broad decision tree that wants
named resolvers.

## EdgeTokenPosition protocol doctrine

`edge_token_manager.py` produces EdgeTokenPosition, a local shadow-policy state
protocol.

```text
DecisionReceipt + trade_permission + previous shadow state
↓
EdgeTokenPosition
```

Doctrine:

```text
EdgeTokenPosition models local policy state.
EdgeTokenPosition does not model broker state.
```

It is the highest protocol in the internal reasoning stack. No downstream
reasoning protocol may grant external execution authority. External execution,
broker approval, Android automation, Robinhood connectors, or APIs begin after
EdgeTokenPosition and must add their own explicit approval/broker authority.

```text
suggested_action != approved_order
contracts_delta != broker fill
position_state != brokerage position
```

The reasoning pipeline ends with EdgeTokenPosition. External execution begins
after it.

## BehaviorAnnotation protocol doctrine

`regime_refinement.py` produces BehaviorAnnotation-style observations. These are
structured diagnostics about tensions between existing protocols, not inputs that
change authority.

Examples:

```text
sticky_trend_conflict
trend_score_overstating_edge
magnet_target_unreachable_today
trap_candidate_waiting_confirmation
```

Behavior annotations answer:

```text
What interesting tension or confirmation should the human notice?
```

They do not answer:

```text
Should permission change?
Should a playbook fire?
Should a token be minted, approved, or executed?
```

Keep this token boundary intact:

```text
Annotations can explain token context but do not mint, approve, or execute tokens.
```

## TargetPlan protocol doctrine

`targeting.py` produces a TargetPlan-style protocol, not merely a target label.
Its conceptual fields are:

```text
Objective
Direction
Strategic destination
Reachable destination
Reachability assessment
Explanation
```

Doctrine:

```text
reachable != executable
```

The cockpit now separates three questions:

```text
Can price get there?        -> TargetPlan / reachability
Should this playbook exist? -> day_bucket legal search space
Is the playbook live now?   -> TriggerResult readiness
Should authority be capped? -> permission governance
```

Targeting may estimate destination and travel plausibility. Targeting may not
authorize execution, change permission, select live timing, or replace readiness.

Protocol families currently recognized:

```text
Market        -> market_day
Evidence      -> ScorePart
Execution     -> TriggerResult
Decision      -> trade_permission
Planning      -> TargetPlan
Annotation    -> BehaviorAnnotation
Shadow policy -> EdgeTokenPosition
Historical    -> DecisionReceipt / Outcome / PermissionTrend
```

Protocol kinds:

```text
Descriptive -> ScorePart, market_day, BehaviorAnnotation, TargetPlan
Normative   -> TriggerResult, trade_permission, EdgeTokenPosition
Historical  -> DecisionReceipt, Outcome, PermissionTrend
```

Do not introduce a TargetPlan class/dataclass just for symmetry. Formalize only
if downstream consumers need validation, versioning, or stricter serialization.

## View explanation doctrine

`execution_flow_view.py` and the cockpit view layer explain completed protocols
to the operator.

```text
View may derive presentation.
View may not derive meaning.
```

Allowed presentation derivations:

- format
- summarize
- truncate
- group
- color
- reorder for readability

Not allowed:

- infer market state
- reinterpret evidence
- change authority
- change readiness
- change permission
- mutate protocols

The view is the only layer that intentionally compresses information for human
cognition. Truncating matched evidence for readability is presentation; changing
`WAIT` to `CONTEXT_MATCH` would be reasoning and belongs upstream.

## Card-builder packet doctrine

`execution_card_builder.py` is the protocol assembly layer.

```text
Card builder may derive.
Card builder may not infer.
```

Allowed derivations:

- gate label from final permission score
- display bias from grammar authority that already exists
- execution-flow packet from market-day, grammar, and permission contracts
- serialized evidence/hierarchy from already-produced ScoreParts

Not allowed:

- infer market state
- choose or revise bucket identity
- resolve thesis/playbook readiness
- apply caps/floors
- change permission because the packet "looks like" a better setup

Display resolution is not market interpretation. If grammar already established
`authority`, `bias`, and `thesis`, the card may display the highest-authority
bias without creating new authority.

## Day-bucket protocol doctrine

`day_bucket.py` is the battlefield/environment protocol. Its contract is larger
than `{"bucket": "...", "allowed_playbooks": [...]}`:

```text
This is the market environment.
These playbook families are legal to search.
This is the risk posture for that environment.
```

Rules:

- Bucket may describe environment.
- Bucket may constrain legal playbook families.
- Bucket may not authorize execution.
- Bucket may not determine timing.
- Bucket may not replace readiness evaluation.

Directional setup buckets are acceptable when they describe the dominant market
phenomenon, e.g. failed-breakdown behavior dominates the battlefield. They become
invalid only if they mean "take this trade now." Legal playbooks still require
`TriggerResult` readiness and permission governance.

Boundary review question:

```text
Are all paths into this bucket describing the same battlefield?
```

This matters most for union buckets such as `range_balance_day`, where positive
gamma, balance disagreement, and near-VWAP magnet context currently converge. If
future inputs such as overnight inventory, dealer pin dominance, or narrow
expected move are added, the review should prove they describe the same
environment, not merely that they share playbooks.

## Playbook readiness doctrine

`live_trigger_check.py` has two valid growth paths:

```text
Direct thesis playbook
↓
generic readiness: thesis allowed by bucket -> TRIGGER_MATCH
```

```text
Contextual playbook
↓
dedicated evaluator: conditions + missing evidence + needs -> TriggerResult
```

Do not turn every direct thesis into a tiny evaluator just for symmetry. Use a
custom evaluator only when the playbook has real readiness concepts such as edge
proximity, confirmation, or defined risk.

`TriggerResult` is the execution-layer sibling of `ScorePart`:

- `ScorePart` describes scored evidence
- `TriggerResult` describes execution readiness evidence

Both are stable contracts at different abstraction levels.

Future optional metadata, not needed yet:

```text
Playbook kind: direct_thesis | context_evaluator
Requires: bucket / edge / confirmation / defined risk
Produces: TRIGGER_MATCH | CONTEXT_MATCH | WAIT
```

## Evidence composition doctrine

```text
One phenomenon
↓
One public vector
↓
Many internal readers, if needed
↓
One ScorePart
```

Compose because concepts exist, not because helpers can exist.

A public vector should represent one primary phenomenon. If the taxonomy shows
multiple overlap families, multiple distinct reasoning paths, hard-to-explain
behavior, or tests naturally splitting into named phenomena, consider internal
composition. Do not split a clean formula merely because it has branches or could
be decomposed into helpers.

Review questions before adding a new branch to an existing vector:

1. Does this belong inside an existing internal reader?
2. Is this actually a new reader under the same phenomenon?
3. Is this a different primary phenomenon that deserves a separate vector or a
   different downstream layer?
4. Would adding it create a new weighted vote for evidence already represented
   elsewhere?

## Cluster findings

### 1. Auction family: acceptance / rejection / trap / opening auction

Parts:

- `acceptance_score`: primary `auction`, category `core_structural`
- `rejection_score`: primary `auction`, overlaps `tactical_candle`
- `trap_score`: primary `auction`, category `tactical_confirmation`
- `opening_auction_score`: primary `auction`, overlaps `session_context`

Formula read:

- `acceptance_score` measures multi-close acceptance above/below nearby levels.
- `rejection_score` measures last-bar wick rejection.
- `trap_score` measures failed breaks around OR/PD levels.
- `opening_auction_score` measures gap behavior against prior close, then decays
  later through `_opening_auction_decay(...)`.

Existing mitigation:

- Score side partially merges `acceptance_score`, `rejection_score`, and
  `trap_score` by using only the highest of the three in the acceptance score
  bucket.
- Bias side still lets all three speak separately.

Audit result:

- Keep current math.
- Keep the asymmetry explicitly documented. It is not automatically wrong.
- Resolved: standalone `candle_score` was removed after this audit. Last-bar
  anatomy remains represented by rejection and pressure/stretch overlaps.

### 2. Momentum family: trend / pressure / regime

Parts:

- `trend_score`: primary `momentum`, category `core_structural`
- `pressure_score`: primary `momentum`, category `suspect_drift_voice`
- `regime_score`: primary `momentum`, category `suspect_drift_voice`

Formula read:

- `trend_score` uses recent slope, VWAP relationship, and 15-minute momentum.
- `pressure_score` uses last-4-bar close persistence, displacement, close
  position, and volume participation.
- `regime_score` uses VWAP relation, momentum, volume, range position, and
  session drift.

Existing mitigation:

- `pressure_score` is damped when it agrees with `trend_score`.
- `regime_score` is damped when it agrees with `trend_score`.

Audit result:

- Keep current damping.
- The formulas measure related but not identical phenomena.
- `pressure_score` and `regime_score` should remain `suspect_drift_voice`, not
  core structural authority.
- Watch item: neutral/chop overlap is not damped by the current multipliers;
  this is acceptable for now but should be remembered before adding more
  momentum-ish features.

### 3. Tactical candle seam: rejection / pressure

Parts:

- `rejection_score`: primary `auction`, overlaps `tactical_candle`
- `pressure_score`: primary `momentum`, overlaps `participation`,
  `tactical_candle`

Formula read:

- `rejection_score` calls `bar_personality(last_bar)`.
- `pressure_score` also uses last-bar close position, but adds a four-bar
  sequence, displacement, and volume participation.

Audit result:

- Resolved: the standalone wick/body `candle_score` vote was removed.
- Do not add another wick/body last-bar score unless the new feature declares why
  rejection/pressure/exhaustion do not already measure the same phenomenon.
- `pressure_score` is more defensibly separate, and taxonomy records its candle
  and participation overlap.

### 4. Placement family: location / balance context / exhaustion / regime

Parts:

- `location_score`: primary `location`, overlaps `balance`, `momentum`, `stretch`
- `balance_context_score`: primary `balance`, overlaps `location`
- `exhaustion_score`: primary `stretch`, overlaps `location`, `momentum`,
  `tactical_candle`
- `regime_score`: primary `momentum`, overlaps `location`, `participation`,
  `balance`

Formula read:

- `location_score` first measures proximity to levels. If not near a level, it
  falls back to balance position/state and range position. The implementation is
  now a composer of named sub-reads rather than one opaque branching blob.
- `balance_context_score` consumes already-computed confluence/disagreement/flip
  state from `pa`.
- `exhaustion_score` measures VWAP/EMA stretch, range extreme, wick rejection,
  and OR proximity.
- `regime_score` uses range position as one ingredient in trend/range labeling.

Audit result:

- `location_score` is the broadest formula in this cluster.
- It still earns primary `location`, but its fallback behavior justifies the
  added `balance`, `momentum`, and `stretch` overlaps.
- No weighted-score split recommended yet.
- Refactor completed: `location_score` is now composed from edge proximity,
  balance position, stretch state, and no-edge fallback while preserving the
  single weighted vector contract.
- Future improvement candidate: add real level-quality input only if a separate
  feed/metric exists. Do not fake level quality from distance alone.

### 5. Volatility/dealer family: volatility / compression / dealer gamma

Parts:

- `volatility_score`: primary `volatility`, overlaps `dealer_positioning`
- `compression_score`: primary `volatility`
- `dealer_gamma_score`: primary `dealer_positioning`, overlaps `volatility`

Formula read:

- `volatility_score` measures ATM IV and premium cheap/rich context.
- `compression_score` measures squeeze/contraction/expansion and coil trigger
  proximity.
- `dealer_gamma_score` measures gamma regime, pin distance, and wall proximity.

Audit result:

- The overlap is real but acceptable.
- `volatility_score` and `compression_score` are context governors, not triggers.
- `dealer_gamma_score` is core structural dealer context, but grammar/playbook
  layers remain responsible for execution interpretation.

### 6. Session family: time of day / opening auction

Parts:

- `time_of_day_score`: primary `session_context`, category `context_governor`
- `opening_auction_score`: primary `auction`, overlaps `session_context`

Formula read:

- `time_of_day_score` only buckets the session window.
- `opening_auction_score` measures open/gap behavior and then decays after the
  auction window.

Audit result:

- The taxonomy correction to `time_of_day_score = context_governor` is correct.
- The execution hierarchy may display it in the core-spine group for readability,
  but formula role is governance/context, not price structure.

## Execution-language leak check

Removed from formula reasons:

- `do not chase without acceptance`
- `do not puke without acceptance`
- `wait for breakdown or reclaim`
- `wait for breakout or failure`
- `countertrend unless accepted`
- `poor R:R`

Replaced with phenomenon descriptions such as:

- `upside continuation acceptance not proven`
- `downside continuation acceptance not proven`
- `breakdown/reclaim direction unresolved`
- `breakout/failure direction unresolved`
- `acceptance not proven`
- `balance chop zone`

The remaining execution phrase `do not chase away from the edge` lives in
`live_trigger_check.py` under playbook `needs`, which is the correct layer.

## Current action list

1. Do not change scoring math from this audit.
2. Do not promote taxonomy or hierarchy into a decision authority.
3. Before adding any vector part, require:
   - category
   - primary correlation family
   - overlap families
   - whether it is independent or intentionally correlated
4. Do not reintroduce standalone candle/body scoring without a taxonomy review.
5. Watch `location_score` if balance/stretch/location explanations become hard
   to reason about.
