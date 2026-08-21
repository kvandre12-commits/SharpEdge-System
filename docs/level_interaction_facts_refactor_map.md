# Level Interaction Facts Refactor Map

## Why this exists

SharpEdge currently has the right instinct but an incomplete seam.

We already have one opinion-free primitive in `cockpit/failed_break_facts.py`, but
multiple interpretation layers still recompute overlapping low-level mechanics:

- `cockpit/level_state_engine.py`
- `cockpit/acceptance_state_engine.py`
- `cockpit/location_state_engine.py`
- parts of `cockpit/execution_grammar.py`

That means the real smell is **not only duplicated acceptance logic**.
The deeper smell is:

> there is not yet one clearly-owned, opinion-free interaction-facts primitive
> underneath level state, acceptance, and nearby level-aware consumers.

This document defines the correct ownership split and a staged refactor path.

---

## Doctrine

### What must never happen

1. **Execution spine must never feed level state.**
   - authority composition is downstream
   - level interpretation is upstream

2. **Level state must not become the owner of acceptance mechanics.**
   - that would remove duplication by misplacing ownership
   - level state answers: _what is price doing at this particular level?_
   - acceptance answers: _is auction/value being established there?_

3. **Location must not use semantic level-state labels for base geometry.**
   - location owns where price is
   - not what a level “means”

### Correct dependency direction

```text
raw market data
  ↓
mechanical fact builders
  ↓
domain interpretations
  ├─ structure
  ├─ level state
  ├─ acceptance
  ├─ trend
  ├─ location
  ├─ participation
  ├─ time
  └─ dealer
  ↓
execution spine / authority composition
  ↓
setup permission + operator presentation
```

### Ownership split

- **Level interaction facts** = mechanical observations at a level
- **Level state** = local level interpretation
- **Acceptance** = auction/value interpretation
- **Location** = geometry / distance / proximity / area status
- **Execution grammar** = higher-order interpretation that may consume one or more
- **Execution spine** = authority composition across verticals

---

## Proposed new primitive

Create a shared primitive module, likely:

- `cockpit/level_interaction_facts.py`

This should either absorb and supersede `failed_break_facts.py`, or wrap it and
expand it without breaking existing callers during migration.

### Primitive contract goal

This layer must stay:

- mechanical
- deterministic
- opinion-free
- non-scoring
- non-biasing
- non-authoritative

### Candidate packet shape

```text
schema: sharpedge.level_interaction_facts.v1
level_name
level_price
role
buffer
window_used
latest_bar_index
current_close
current_close_relation
current_high_relation
current_low_relation
closes_above_count
closes_below_count
closes_at_level_count
acceptance_window_used
recent_breach_above
recent_breach_below
breach_above_latest_index
breach_below_latest_index
breach_above_highest_high
breach_below_deepest_low
breach_above_extension_pct
breach_below_depth_pct
reclaim_above_level_index
reject_below_level_index
bars_since_reclaim_above_level
bars_since_reject_below_level
hold_above_count
hold_below_count
first_bar_above_index
first_bar_below_index
```

Not every field must ship in phase 1, but the packet should be designed so that:

- level state can interpret tests / holds / failed breaks
- acceptance can interpret acceptance persistence and direction
- grammar can interpret failed/accepted break states
- location can reuse relation/buffer geometry if helpful

---

## What each domain should consume

### 1. Level state engine

`cockpit/level_state_engine.py`

Should consume shared interaction facts and answer:

- `testing_support`
- `holding_above_support`
- `failed_break_reclaimed`
- `lost_support`
- `testing_resistance`
- `holding_below_resistance`
- `failed_break_rejected`
- `accepted_above_resistance`
- `accepted_above_reference`
- `accepted_below_reference`

It may expose acceptance-related observations inside its packet because those are
local facts about the level, but it must not become the global owner of auction
acceptance doctrine.

### 2. Acceptance state engine

`cockpit/acceptance_state_engine.py`

Should consume the same shared interaction facts and answer:

- acceptance direction
- acceptance strength
- acceptance persistence
- accepted level count
- representative level
- auction status

Acceptance may select from multiple level facts, rank them, or summarize them,
but the low-level close/buffer/breach mechanics should no longer live privately
inside this engine.

### 3. Location state engine

`cockpit/location_state_engine.py`

Should continue to own geometry:

- nearest reference
- above/below/at relation
- proximity class
- between references / above all / below all
- decision-area status

It may optionally reuse a shared relation helper or shared geometry facts, but
its base scoring/logic must not depend on semantic labels like:

- `FAILED_BREAKDOWN_RECLAIMED`
- `TESTING_SUPPORT`
- `HOLDING_BELOW_RESISTANCE`

Those may be added later as **contextual explanation**, not as base location
ownership.

### 4. Execution grammar

`cockpit/execution_grammar.py`

This is exactly where consuming interpreted level-state outputs makes sense.
Grammar may consume:

- level-state interpretations
- acceptance interpretations
- dealer interpretations
- setup cards

This is the correct place to decide whether a failed break matters to authority.

### 5. Execution spine

`cockpit/execution_vector_engine.py`

The spine should consume domain interpretations, not raw authority shortcuts.

Possible future use of level-state outputs inside the spine is allowed, but only
as **selected contextual evidence**. Examples:

- acceptance score may rely on acceptance-state built from shared facts
- location score may mention that the nearest reference is currently being tested
- trap/grammar layers may weight level-state evidence separately

But each vertical must still own its own judgment.

---

## Current repo mapping

### Already mechanical enough

- `cockpit/failed_break_facts.py`
  - good seed primitive
  - explicitly opinion-free
  - already computes breach/reclaim mechanics

### Currently duplicating mechanics

- `cockpit/level_state_engine.py`
  - recomputes close relation
  - recomputes acceptance counts/window
- `cockpit/acceptance_state_engine.py`
  - separately recomputes close-vs-level acceptance
- `cockpit/location_state_engine.py`
  - separately recomputes relation/buffer geometry

### Current downstream consumers of level-aware logic

- `cockpit/setups.py`
- `cockpit/execution_grammar.py`
- `cockpit/transition_pressure/potential_energy.py`
- `cockpit/level_state_view.py`
- `cockpit/live_chart_svg.py`

---

## Refactor stages

## Stage 0 — freeze behavior with tests

Before moving ownership around, add or tighten tests around:

- `failed_break_facts.py`
- `level_state_engine.py`
- `acceptance_state_engine.py`
- `location_state_engine.py`
- `execution_grammar.py`

### Add cases for

- support reclaimed after downside breach
- resistance rejected after upside breach
- accepted above resistance without failed-break rejection
- accepted below support without reclaim
- at-level / in-buffer ambiguity
- no clean acceptance
- multiple candidate levels with one representative level
- location geometry staying stable even when semantic level meaning changes

Goal: lock behavior before deduping mechanics.

---

## Stage 1 — extract shared level interaction facts

Create:

- `cockpit/level_interaction_facts.py`

### Phase 1 rules

- no callers removed yet
- no scoring changes
- no presentation changes
- no doctrine changes
- build the primitive first

### Implementation approach

1. Start from `failed_break_facts.py`
2. Expand packet shape to include reusable close-relation / close-count mechanics
3. Keep names boring and mechanical
4. Add `build_level_interaction_facts_for_levels(...)`
5. Leave `failed_break_facts.py` as:
   - thin wrapper
   - compatibility shim
   - or deprecated alias

### Deliverable

A reusable primitive that both level state and acceptance can consume without one
owning the other.

---

## Stage 2 — rebase level state engine on shared facts

Update `cockpit/level_state_engine.py` so it:

- consumes `level_interaction_facts`
- stops privately computing close-relation and acceptance-window mechanics
- keeps ownership of local semantic labels only

### Allowed outputs

- `event_state`
- `failed_break_candidate`
- `summary`
- `actionable`

### Not allowed

- authority scores
- global acceptance doctrine
- permission bias

This should be a mostly internal refactor if stage 0 tests are good.

---

## Stage 3 — rebase acceptance state engine on shared facts

Update `cockpit/acceptance_state_engine.py` so it:

- consumes the same interaction facts packets
- ranks or summarizes them according to acceptance doctrine
- stops owning raw close/buffer mechanics privately

### Acceptance remains free to decide

- which levels count
- how many accepted levels matter
- what representative level means
- whether auction acceptance is broad, weak, narrow, or absent

That is doctrinal ownership.
Mechanical close counts are not.

---

## Stage 4 — optionally share geometry helpers with location

This stage is optional and should be conservative.

Possible cleanup:

- extract shared relation helpers:
  - above / below / at reference
  - buffer usage
  - distance / distance_pct

But do **not** let location import semantic level-state labels as base logic.

Safe pattern:

- location builds geometry first
- optional explanation layer may append:
  - “nearest reference is currently testing reclaimed support”

Unsafe pattern:

- “location is bullish because level state says failed breakdown reclaimed”

That is no longer pure location.

---

## Stage 5 — rebase grammar and setup consumers where useful

After the primitive is stable:

- simplify `cockpit/execution_grammar.py`
- simplify `cockpit/setups.py`
- simplify any transition-pressure consumers

Goal:

- shared facts feed multiple interpreters
- interpreters do not each rebuild the same breach/reclaim mechanics

This is where the architecture starts feeling actually coherent instead of merely
less duplicated.

---

## Stage 6 — only then revisit spine/context composition

After the primitive + domain consumers are stable, evaluate whether the spine
should consume selected level-state or acceptance outputs differently.

Examples:

- acceptance score enriched by persistence from shared facts
- location explanation enriched by local level-state note
- trap/grammar surfaces promoted into supporting evidence lanes

Do **not** do this early.

If done too soon, the repo will re-couple interpretation and authority before
ownership boundaries are settled.

---

## Guardrails

### Keep these hard boundaries

#### Level interaction facts
- no score
- no bias
- no trade language
- no setup names
- no permission verdicts

#### Level state
- local level meaning only
- may emit semantic labels
- must not own global auction doctrine

#### Acceptance
- owns auction/value interpretation
- may choose representative level
- must not privately rebuild core mechanics once primitive exists

#### Location
- owns geometry only
- semantic enrichment optional and secondary

#### Spine
- consumes interpretations
- does not back-drive lower layers

---

## Suggested test plan after each stage

Focused checks:

```bash
pytest -q \
  tests/test_level_state_engine.py \
  tests/test_execution_grammar.py \
  tests/test_trade_permission.py \
  tests/test_live_read_view.py
```

If new tests are added for the primitive, include:

```bash
pytest -q tests/test_level_interaction_facts.py
```

Also run the cockpit-adjacent safety checks when view payloads or packets change.

---

## Success criteria

The refactor is successful when:

1. Level state and acceptance no longer duplicate low-level close/buffer/breach mechanics.
2. Neither engine becomes the owner of the other’s doctrine.
3. Location remains geometrically pure.
4. Execution grammar gets simpler, not more magical.
5. Execution spine consumes clearer domain packets, not blurrier ones.
6. The cockpit explanation becomes easier to trust because ownership is obvious.

---

## Sharp verdict

The real fix is **not**:

- “make acceptance depend on level state”

The real fix is:

- **extract a shared level interaction facts primitive**
- **let level state and acceptance interpret it independently**
- **keep execution authority downstream of those judgments**

That is the clean version.
Anything else is just prettier ownership drift.
