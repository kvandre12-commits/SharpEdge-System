# Transition Pressure Roadmap

## Best idea

The next major SharpEdge evolution should be a **Transition Pressure Engine**.

This should sit **above** the current vector stack and estimate:

> how close the market is to leaving its current state.

That is a better problem than direct price prediction.

Do **not** ask first:
- bullish or bearish?
- where will price go?
- how much volume is showing?

Ask first:
- is energy being stored?
- are constraints tightening?
- are multiple independent vectors improving together?
- is permission rising before price releases?
- is balance running out of places to persist?

---

## Why this is the right next move

Current SharpEdge vectors mostly describe **state**:
- structure
- acceptance
- trend
- participation
- location
- dealer gamma

The next edge is to model **change of state**:
- vector velocity
- vector acceleration
- vector confidence change
- latent energy build
- transition pressure

This matches what the system has already started to discover:
- permission trend is more useful than naive directional prediction
- some large moves happen without huge volume
- price is often the last thing to react
- nonlinear interactions matter more than isolated scores

---

## Architecture shift

Old mental model:

`Market State -> Direction -> Trigger -> Trade`

New mental model:

`Current State -> State Deltas -> Potential Energy -> Nonlinear Interactions -> Transition Pressure -> Trigger -> Execution Permission`

Price should not dominate the first layers.
Forward-looking does **not** mean predicting price.
It means detecting **state transition pressure** before price fully resolves.

---

## Proposed stack

### 1. State layer
Keep the current vector engine.

This remains the descriptive base:
- structure
- acceptance
- trend
- participation
- location
- dealer gamma
- time of day
- balance context

Do not throw this away.

### 2. Delta layer
Add first- and second-derivative features for core vectors.

Examples:
- trend velocity
- trend acceleration
- acceptance velocity
- participation velocity
- gamma constraint change
- permission slope
- permission acceleration

This answers whether conditions are strengthening, weakening, stalling, or rolling over.

### 3. Potential energy layer
Model latent variables that lead price.

This is the most important new layer.

Candidate features:
- compression intensity
- failed-break persistence
- repeated auction rejection density
- trapped inventory accumulation
- liquidity thinning / drying-up proxies
- narrowing rotation range
- acceptance forming beneath/above a key decision area
- unstable dealer positioning / pin fragility

This layer should answer:
- how much energy is stored?
- what kind of release is becoming more likely?

### 4. Interaction layer
Explicitly model nonlinear interactions instead of relying on isolated scores.

Priority interactions:
- trend × gamma
- compression × acceptance
- location × dealer positioning
- participation × time of day
- liquidity × trigger quality
- failed-break persistence × trapped side

Rule: interactions should be first-class packets, not buried as prose.

### 5. Transition pressure layer
This is the actual new crown jewel.

It should estimate whether the market is nearing a state transition.

It should score things like:
- multiple vectors improving together
- permission rising while balance options shrink
- failed breaks getting shorter or more frequent
- compression increasing while acceptance improves
- directional asymmetry building without clean release yet
- conflicting forces resolving toward one side

This is not directional pressure.
This is **resolution pressure**.

### 6. Trigger / execution layer
Only after transition pressure is elevated should triggers matter.

This preserves the current SharpEdge execution discipline:
- trigger quality still matters
- setup conviction still matters
- bucket-conditioned execution still matters

The new engine should improve *when* those things become interesting.

---

## Smallest useful v1

Build the first version as a small package:

```text
sharpedge/transition_pressure/
  __init__.py
  deltas.py
  potential_energy.py
  interactions.py
  pressure.py
  packet.py
```

### First v1 inputs
Start with the smallest useful set:
- permission_delta
- trend_delta
- acceptance_delta
- participation_delta
- compression_score
- failed_auction_score
- location_pressure
- gamma_constraint

This is enough to detect transition pressure without pretending the system is already omniscient.

### First v1 output packet
Output one simple packet:

```text
transition_pressure_score: 0-100
transition_state: dormant | building | pressurized | release_candidate | resolving
directional_bias: upside | downside | two_way | unclear
reason: permission rising, acceptance building, balance narrowing, gamma constraint active
```

This is the first packet worth wiring into the cockpit.

---

## Operating rule

**Transition pressure should not create trades.**

It should upgrade attention.

Attention ladder:
- `low pressure` -> ignore
- `building` -> watch
- `pressurized` -> prepare
- `release_candidate` -> require trigger
- `resolving` -> execution engine takes over

This rule is critical.
Without it, transition pressure becomes just another noisy score.

---

## Best implementation sequence

### Phase 1 — Delta packets
Add explicit delta packets for current vectors and permission.

Proposed packet families:
- `sharpedge.vector_deltas.v1`
- `sharpedge.permission_deltas.v1`

Minimum fields:
- current
- previous
- velocity
- acceleration
- confidence_change
- status (`strengthening`, `weakening`, `flat`, `reversing`)

Start with:
- trend
- acceptance
- participation
- location
- dealer gamma constraint
- permission score

### Phase 2 — Potential energy packet
Create a first explicit latent-energy model.

Proposed packet:
- `sharpedge.potential_energy.v1`

Core sub-surfaces:
- compression_energy
- failed_auction_energy
- trapped_inventory_energy
- acceptance_energy
- liquidity_tension
- dealer_instability

Output:
- total_energy_score
- dominant_energy_source
- release_bias
- release_quality
- release_fragility

This is likely the highest-value single addition.

### Phase 3 — Interaction engine
Create explicit interaction surfaces.

Proposed packet:
- `sharpedge.vector_interactions.v2`

This should supersede simple “good combos / bad combos” by adding:
- interaction strength
- interaction direction
- interaction stability
- interaction novelty
- interaction worsening / improving state

Do not stop at static classification.
Track whether the interaction itself is strengthening.

### Phase 4 — Transition pressure engine
Build the main model.

Proposed packet:
- `sharpedge.transition_pressure.v1`

Core outputs:
- transition_pressure_score
- transition_state
- directional_bias
- reason
- pressure_sources[]
- pressure_conflicts[]
- invalidation_conditions[]

Suggested progression:
- dormant
- building
- pressurized
- release_candidate
- resolving

This packet should become the primary forward-looking brain.

### Phase 5 — Cockpit integration
Add a dedicated cockpit block:
- **TRANSITION PRESSURE**

It should show:
- score
- state
- directional bias
- dominant reasons
- whether attention should upgrade
- whether permission is leading price

This should likely sit above or beside timeframe agreement.

### Phase 6 — Bucket integration
Let bucket selection use transition pressure as a conditioning input.

Not to replace existing buckets immediately.
Instead:
- annotate bucket confidence
- annotate release context
- distinguish stable trend from unstable transition
- distinguish sticky drift from primed break

Eventually, buckets can evolve from static day labels toward transition-aware state families.

---

## What not to do

Do not:
- replace the current vector engine wholesale
- collapse everything into one mega-score
- pretend raw volume explains every move
- make pressure a synonym for bullishness
- let price become the lead variable again

The whole point is to model what leads price, not what price has already done.

---

## Concrete recommendation

If only one thing is built next, build this:

## `sharpedge.transition_pressure.v1`

But do it in this order:
1. permission/vector deltas
2. potential energy packet
3. interaction strengthening
4. transition pressure synthesis

That sequence is DRY, testable, and consistent with the current SharpEdge architecture.

The most important early edge to protect is this:

> **permission rising before price moves**

That is probably the first real forward-looking component SharpEdge can exploit cleanly.

---

## Next refinement order

Refine v1 in this order:
1. directional_bias
2. pressure persistence
3. interaction strengthening
4. permission-leading-price detection
5. threshold calibration

This order matters.
Do not start with threshold polish before the packet semantics are smarter.

### Directional bias refinement
Do not collapse bias into plain bullish/bearish yet.
Use auction-directional outcomes instead:
- `upside_release_possible`
- `downside_release_possible`
- `two_way_compression`
- `failed_upside_release`
- `failed_downside_release`
- `unclear`

That keeps the model focused on release structure, not naive direction-calling.

### Pressure persistence refinement
Add explicit persistence tracking so the system describes pressure behavior over time.

Suggested persistence states:
- `building`
- `holding`
- `decaying`
- `recycling`

Example target read:
- `transition_pressure: building`
- `score: 45`
- `persistence: holding_3_bars`
- `bias: two_way_compression`
- `attention: watch`

This is better than a naked score because it tells the operator whether the system is coiling, stalling, fading, or reloading.

### Core forward-looking read to protect
The next real edge is not just:
- `pressure is 45`

It is:
- `pressure rising while price is still balanced near a decision level with compression + gamma constraint active`

That is the forward-looking contract.
The goal is to detect transition pressure before price fully resolves.

---

## Short version

Best next move:

> Stop trying to predict price directly.
> Start estimating whether the market is nearing a transition.

That means SharpEdge should evolve from a **state reader** into a **state-transition engine**.
