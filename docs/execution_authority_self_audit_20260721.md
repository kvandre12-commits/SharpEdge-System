# Execution Authority Self-Audit — 2026-07-21

## Decision

Demote the cockpit execution score spine from final-authority language to a
**diagnostic/advisory execution read**.

The cockpit still computes and displays:

- bucket-conditioned spine score
- trade gate / bias / recommended action
- raw vector score
- core state packets and hierarchy rows
- core adjunct vectors for pressure and balance context

But these are no longer labeled as final execution permission. Final authority
is explicitly delegated to:

```text
approval_decision_plus_operator
```

## Why

The existing execution-spine audit found the score spine is explainable but not
calibrated tightly enough to be treated as final authority:

- full-stack legacy mode blends core, secondary, context, and suspect drift voices
- score thresholds are authored buckets, not empirical probability cutoffs
- location / acceptance / dealer-gamma still have known leakage seams
- even cleaner ACE core-only mode still uses authored bucket offsets and gates

That means the score is useful cockpit context, but not a broker/permission
contract. Cute score. Short leash.

## Code changes

- Added `cockpit/authority_self_audit.py`
- `trade_permission` cards now include `authority_self_audit`
- `authority_adjudication.we_are_doing_this` now carries:
  - `score_spine_role`
  - `final_authority_source`
- Live Read headline changed from `FINAL EXECUTION PERMISSION` to `EXECUTION READ`
- CLI changed from `execution gate` / `spine authority` to:
  - `execution read`
  - `spine diagnostic`

## Current doctrine

```text
score_spine_role = diagnostic_advisory
final_authority_source = approval_decision_plus_operator
```

## 2026-07-21 main-spine adjunct update

Promoted two modest-weight vectors into the main diagnostic spine:

- `pressure_score` — short-horizon follow-through pressure
- `balance_context_score` — balance/value confluence or disagreement

Rationale:

- Pressure is cleaner than trap/rejection because it is a multi-bar pressure read,
  not setup identity.
- Balance context makes disagreement/confluence visible next to location instead
  of burying it as a secondary governor.
- Both remain diagnostic/advisory and do not change the final authority source.

Rejected for now:

- `trap_score` / `rejection_score`: setup corroboration, too easy to double-count
  with acceptance.
- `opening_auction_score`: decaying context governor, not a persistent execution
  vector.
- `regime_score`: still a broad suspect drift voice with known trend/location
  overlap.

The next legitimate promotion path is not prettier UI text. It is validation:
vertical calibration, threshold review, and governance sign-off.
