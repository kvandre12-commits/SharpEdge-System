# Range Posture Repo Grep Audit

Date: 2026-07-11
Scope: `SharpEdge-System` source audit for remaining raw range/VWAP posture threshold logic after canonical `cockpit/range_posture.py` adoption.

## Verdict

This audit is now a **resolved historical snapshot**.

The repo-level posture cleanup identified here has been completed:
- `cockpit/setups.py` exhaustion detection was refactored onto canonical `build_range_posture()` semantics
- `cockpit/make_cockpit.py` range narration was refactored onto canonical `build_range_posture()` semantics

As of the follow-up cleanup pass, the repo is **clean for this specific duplication seam**.

Validation performed after the refactor:
- `ruff check --fix` passed
- focused setup/narration regression slice: **63 passed**
- broader cockpit regression pack: **140 passed**
- post-change grep found no remaining cockpit-source matches for:
  - `rng_pos <= 22`
  - `rng_pos >= 78`
  - `abs(pa["vs_vwap"]) >= 0.4`
  - `rp >= 80`
  - `rp <= 20`

---

## Resolved seams

### `cockpit/setups.py`

#### `detect_exhaustion()` — resolved

Previous issue:
- raw `rng_pos <= 22` / `rng_pos >= 78`
- raw `abs(pa["vs_vwap"]) >= 0.4`

Resolution:
- exhaustion edge classification now uses canonical posture semantics via `build_range_posture()`
- the stricter exhaustion-specific VWAP stretch rule was preserved intentionally as a named setup constant:
  - `EXHAUSTION_STRETCH_MIN_PCT = 0.40`

Why this is the right shape:
- generic edge/upside/downside doctrine now comes from the canonical posture helper
- setup-specific strictness stays local and explicit instead of masquerading as generic doctrine

### `cockpit/make_cockpit.py`

#### `synthesize()` range narration — resolved

Previous issue:
- raw `rp >= 80` / `rp <= 20` narration for:
  - `At day HIGHS`
  - `At day LOWS`
  - `Mid-range`

Resolution:
- those labels now derive from canonical `build_range_posture()` semantics instead of standalone `80/20` literals

Why this matters:
- operator-facing narration now stays aligned with the same posture model used elsewhere
- less drift bait, less threshold folklore, fewer future cleanup laps

---

## Low-priority / likely intentional local heuristics

### `cockpit/setups.py`

#### 3) `HANDOFF_MIN_RNG_POS = 45`
Relevant line:
- `setups.py:40`, used at `setups.py:498`

Assessment:
- This does **not** look like duplicate extreme/edge doctrine.
- It looks like a setup-specific promotion gate for exhaustion -> runner handoff.
- Leave it alone unless behavior review says otherwise.

#### 4) continuation / sticky-noise thresholds
Examples:
- `CONTINUATION_MIN_MOM = 0.05`
- `CONTINUATION_MIN_VOL_MULT = 1.2`
- `PIN_PROXIMITY_PCT = 0.10`

Assessment:
- These are setup-owned heuristics, not generic range-posture semantics.
- Not part of the same duplication seam.

---

## Noise / non-actionable grep hits

### Canonical owners
- `cockpit/range_posture.py`
- `cockpit/vwap_posture.py`

These are supposed to contain thresholds. Obviously. That’s the point.

### Already reworked consumers
Confirmed posture consumers now include:
- `cockpit/execution_vector_engine.py`
- `cockpit/regime_refinement.py`
- `cockpit/live_trigger_check.py`
- `cockpit/execution_expansion_potential.py`
- `cockpit/execution_vector_interactions.py`
- `cockpit/spine_phase_model.py`
- `cockpit/timeframe_agreement.py`
- `cockpit/transition_pressure/pressure.py`
- `cockpit/transition_pressure/potential_energy.py`

### Validation/test/generated noise
Examples:
- `scripts/validate_asymmetric_execution_logic.py`
- `tests/test_transition_pressure.py`
- `tests/test_range_posture.py`
- `outputs/history/...`
- `outputs/*.json`, `outputs/*.html`

These should not drive refactor priority.

### Historical docs
Examples:
- `docs/cockpit_duplicate_logic_audit.md`
- `docs/execution_spine_audit.md`

These describe prior duplication; they are not live-code seams.

---

## Current priority order

1. No immediate action required for this seam
2. Re-run a repo grep only if a new operator-facing drift issue appears
3. Treat the remaining numeric thresholds as canonical-owner or setup-local unless proven otherwise

---

## Historical note

This file originally captured the remaining open posture-duplication seams.
Those seams have now been closed.

Do **not** use this artifact as evidence that `setups.py` or `make_cockpit.py` still need the old cleanup pass unless a fresh grep/behavior review reopens the issue.
