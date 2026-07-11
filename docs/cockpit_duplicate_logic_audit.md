# SharpEdge Cockpit Duplicate Logic Audit

Date: 2026-07-11

## Scope

This audit focuses on **duplicate or semantically duplicated logic** inside the
`cockpit/` stack — especially where the same market fact is:

- computed in multiple places,
- interpreted twice under different names,
- or promoted into more than one authority surface.

This is **not** a broad architecture audit. It is a cleanup map for overlap,
drift risk, and refactor priority.

---

## Executive summary

The cockpit does **not** look like random copy-paste spaghetti. Good news.

But it **does** have several real duplicate-logic seams:

1. **State packet -> score mapping is duplicated across multiple vector methods.**
2. **Dealer/gamma interpretation is duplicated in both the vector engine and execution grammar.**
3. **Session/time doctrine is split across multiple files with overlapping authority.**
4. **VWAP posture logic is repeated all over the repo with slightly different thresholds and meanings.**
5. **Acceptance-like close stacking exists in more than one place.**
6. **Trend/regime/setup synthesis reuse the same inputs to tell very similar stories.**

Big picture:
- there is less raw line-for-line duplication than feared,
- but there is a lot of **semantic duplication**,
- which is the more dangerous kind because it quietly drifts while looking “modular.”

---

## Findings

### 1) State packet -> score mapping is duplicated

**Severity:** High  
**Type:** real duplicate interpretation logic

The new state engines are good. They produce explicit packets:

- `build_structure_state()`
- `build_acceptance_state()`
- `build_trend_state()`
- `build_time_state()`
- `build_location_state()`
- `build_dealer_state()`

But the score interpretation still lives separately inside
`cockpit/execution_vector_engine.py`:

- `_score_structure()`
- `_score_acceptance()`
- `_score_trend()`
- `_score_time_of_day()`
- `_score_dealer_gamma()`
- `_score_location()` delegates to `execution_location_score.py`

### Why this is duplication

Each vertical now has **two doctrine surfaces**:

1. the state packet naming / semantics
2. the score mapping from that state into permission influence

That split is fine **only if score mapping is treated as a first-class adapter**.
Right now it is scattered inside the vector engine methods, so doctrine can drift.

### Evidence

- `cockpit/structure_state_engine.py`
- `cockpit/acceptance_state_engine.py`
- `cockpit/trend_state_engine.py`
- `cockpit/time_state_engine.py`
- `cockpit/location_state_engine.py`
- `cockpit/dealer_state_engine.py`
- `cockpit/execution_vector_engine.py`
- `cockpit/execution_location_score.py`

### Why it matters

If the state packet changes but the `_score_*()` method does not, the UI and the
permission engine can silently disagree while both remain “correct” in isolation.

### Recommendation

Create one dedicated adapter layer, something like:

- `cockpit/execution_state_scores.py`

with functions like:

- `score_structure_state(packet)`
- `score_acceptance_state(packet)`
- `score_trend_state(packet)`
- `score_time_state(packet)`
- `score_location_state(packet)`
- `score_dealer_state(packet)`

Then `ExecutionVectorEngine` becomes orchestration, not interpretation storage.

---

### 2) Dealer/gamma interpretation is duplicated

**Severity:** High  
**Type:** real duplicate score/meaning mapping

Dealer state is computed in:

- `cockpit/dealer_state_engine.py`

But then dealer meaning is remapped at least twice:

1. `ExecutionVectorEngine._score_dealer_gamma()`
2. `execution_grammar.build_dealer_gamma_state()`

### Evidence

`cockpit/execution_vector_engine.py`
- `positive_gamma_gravity -> 38`
- `positive_gamma_context -> 48`
- `negative_gamma_expansion -> 72`

`cockpit/execution_grammar.py`
- `build_dealer_gamma_state()` repeats the same mapping again

### Why this is bad

That is real duplication, not just healthy reuse.
If one mapping changes and the other does not, the grammar and permission score
start living in parallel universes. Cute in sci-fi, bad in trading software.

### Recommendation

Move dealer state scoring/normalization into one place:

- either `dealer_state_engine.py`
- or shared adapter module like `execution_state_scores.py`

Then both grammar and vector engine should consume the same normalized dealer packet.

---

### 3) Session/time doctrine is split across overlapping surfaces

**Severity:** Medium-High  
**Type:** semantic duplication / authority overlap

Time logic currently lives across several places:

- `time_state_engine.py`
- `execution_vector_engine._score_time_of_day()`
- `execution_vector_engine._opening_auction_decay()`
- `execution_vector_engine._get_minutes_since_open()`
- `trade_permission_context.score_opening_auction()`

### What is duplicated

Not exact line duplication — worse, **session doctrine is fragmented**:

- one place classifies session windows,
- one assigns time score values,
- one decays opening-auction significance later in the day,
- one separately scores opening-gap context.

These are related concepts with overlapping authority over “how much time should matter now.”

### Drift risk

A future tweak to session windows can easily update the state engine while leaving
opening-auction decay assumptions untouched.

### Recommendation

Centralize session doctrine constants and boundaries in one module, e.g.:

- `cockpit/session_doctrine.py`

That module should own:

- minute buckets
- window names
- opening influence decay thresholds

Scoring can still differ by consumer, but the **time ontology** should be singular.

---

### 4) VWAP posture logic is repeated across many files

**Severity:** High  
**Type:** semantic duplication with threshold drift risk

VWAP-related posture appears in many places:

- `make_cockpit.py::synthesize()`
- `day_bucket.py::_vwap_context()`
- `trend_state_engine.py`
- `execution_vector_engine._score_regime()`
- `setups.py`
- several display surfaces

### Evidence

Examples:

- `make_cockpit.py` uses `vs_vwap > 0.05` / `< -0.05` for “bulls in control”
- `day_bucket.py` uses:
  - `VWAP_MAGNET_BAND_PCT = 0.08`
  - `VWAP_ACCEPTANCE_BAND_PCT = 0.05`
- `trend_state_engine.py` uses `VWAP_FLAT_PCT = 0.05`
- `execution_vector_engine._score_regime()` uses `abs(vs_vwap) <= 0.05`
- `setups.py` has multiple independent VWAP acceptance/stretch checks

### Why this matters

Some of this reuse is valid. VWAP is a core fact. But right now the repo has
multiple slightly different answers to:

- hugging VWAP
- accepted above VWAP
- stretched from VWAP
- above/below VWAP directional control

That is a doctrine-drift trap.

### Recommendation

Extract one canonical VWAP posture helper, e.g.:

- `cockpit/vwap_posture.py`

It should emit a packet like:

- `state`: `hugging_vwap` / `near_vwap` / `above_vwap` / `below_vwap` / `stretched_above` / `stretched_below`
- `distance_pct`
- `acceptance_state`
- `stretch_state`

Then day-bucket, trend, setups, and synthesize can consume that instead of each
reinventing a small private VWAP religion.

---

### 5) Acceptance-like close stacking exists in more than one place

**Severity:** Medium  
**Type:** partial duplicate feature logic

The clean acceptance engine is here:

- `acceptance_state_engine.py`

But similar close-stacking logic also appears in setup logic, especially in:

- `setups.py` negative-gamma continuation handoff path

That path independently checks whether recent bars were accepted above VWAP.

### Why this matters

This is not automatically wrong. Setup logic often needs stricter, setup-specific proof.
But if “accepted above VWAP” is a real concept used elsewhere, the implementation
should come from a shared helper rather than a bespoke local count.

### Recommendation

Extract a tiny reusable helper for repeated close-based acceptance tests, e.g.:

- `accepted_above_reference(closes, level, window, min_closes, buffer)`
- `accepted_below_reference(...)`

Then keep setup-specific thresholds local, but not the mechanic itself.

---

### 6) Trend, regime, and setup detectors reuse the same directional story

**Severity:** Medium-High  
**Type:** semantic duplication / narrative overlap

Several places use overlapping inputs:

- `vs_vwap`
- `mom15`
- short-horizon drift
- `vol_mult`
- range position

These appear in:

- `trend_state_engine.py`
- `execution_vector_engine._score_regime()`
- `make_cockpit.py::synthesize()`
- `setups.py::detect_negative_gamma_continuation()`
- `setups.py::detect_sticky_noise()`

### Why this is risky

Even if the modules serve different jobs, they are often describing the same market
fact in different costumes:

- “trend aligned up”
- “bulls in control”
- “trend day regime”
- “negative gamma continuation candidate”

This is where the cockpit can accidentally overweight one reality four times.

### Recommendation

Do **not** merge these blindly. They have different authority levels.
Instead:

1. declare a canonical low-level directional packet
2. let higher layers cite it
3. stop recomputing the same directional posture from scratch where possible

Good candidate packet:

- `short_horizon_direction_state`

with fields like:
- `vwap_relation`
- `momentum_state`
- `path_slope_state`
- `participation_state`

Then trend/regime/setup can specialize from one fact base.

---

### 7) Location doctrine is cleaner now, but old narrative surfaces still overlap it

**Severity:** Medium  
**Type:** residual semantic duplication

`location_state_engine.py` is properly spatial now.
Good puppy.

But older narrative surfaces still tell location-ish stories using:

- range position
- balance location
- VWAP relation

Examples include:
- `make_cockpit.py::synthesize()`
- `execution_vector_engine._score_regime()`

### Why this matters

Location should answer:
- where are we relative to tracked references?

Not:
- whether momentum is good,
- whether balance favors fading,
- whether price is stretched.

That broader battlefield interpretation belongs elsewhere.

### Recommendation

Keep `location_state_engine.py` as the canonical location fact source and gradually
rewrite older narrative surfaces to reference it instead of restating location ad hoc.

---

## What is acceptable overlap vs bad duplication?

### Acceptable overlap

These can use the same raw facts without being considered bugs:

- trend reading uses `vs_vwap`
- setup logic uses `vs_vwap`
- bucket classification uses `vs_vwap`

**if** they are consuming a canonical posture packet or clearly documented shared doctrine.

### Bad duplication

These are the ones worth fixing first:

1. same state -> same score mapping in more than one file
2. same concept thresholds copied with slightly different numbers and no owning module
3. same market fact promoted into multiple authority surfaces without a canonical fact packet

---

## Refactor priority

### Priority 1 — fix immediately

1. **Unify state-packet scoring adapters**
   - create `execution_state_scores.py`
   - remove direct score mapping duplication from vector engine and grammar

2. **Unify dealer/gamma score mapping**
   - one source of truth for dealer state normalization

### Priority 2 — next cleanup pass

3. **Centralize session doctrine**
   - minute buckets
   - labels
   - opening influence decay thresholds

4. **Centralize VWAP posture doctrine**
   - hugging / near / accepted / stretched

### Priority 3 — careful, not reckless

5. **Extract acceptance helpers for repeated close tests**
6. **Refactor old narrative surfaces to cite canonical packets**
7. **Reduce trend/regime/setup recomputation where authority truly overlaps**

---

## Suggested implementation sequence

1. Add `cockpit/execution_state_scores.py`
2. Move dealer score mapping there first
3. Move structure/acceptance/trend/time/location mappings there
4. Add `cockpit/session_doctrine.py`
5. Add `cockpit/vwap_posture.py`
6. Slowly migrate:
   - `day_bucket.py`
   - `make_cockpit.py`
   - `setups.py`
   - `execution_vector_engine.py`

This keeps the blast radius sane and preserves doctrinal clarity.

---

## Bottom line

The cockpit’s biggest duplicate-logic risk is **not** raw copy-paste.
It is **fact interpretation drift**:

- same truth,
- multiple modules,
- slightly different thresholds,
- all sounding authoritative.

That’s the kind of bug that passes tests and still lies to the operator.

So yeah — the repo is ready for a **canonical state adapter pass** and a
**VWAP/session doctrine consolidation pass** next.
