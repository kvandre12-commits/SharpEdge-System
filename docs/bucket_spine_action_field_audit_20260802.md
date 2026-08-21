# Bucket Spine Action-Field Audit

Date: 2026-08-02

## Question

Should `bucket_conditioned_spine` emit anything action-shaped at all?

Short answer:

- **Canonical answer:** no, not as the primary semantic.
- **Practical migration answer:** keep `recommended_action` only as a deprecated
  compatibility alias while descriptive consumers move to `diagnostic_posture`.

## Why

The bucket-conditioned spine is a **diagnostic execution read**. It does not own
final authority. When it emits a field named `recommended_action`, downstream
code is tempted to treat a context read like a command source.

That creates doctrine bleed:

- score spine becomes advisory **and** imperative at the same time
- descriptive surfaces inherit command-voice language
- later consumers start using a compatibility field as if it were canonical

## Consumer audit

### Safe descriptive consumers

These can and should prefer `diagnostic_posture`:

- `cockpit/execution_flow_view.py`
- `cockpit/make_cockpit.py`
- `cockpit/execution_card_builder.py`
- `cockpit/timeframe_agreement.py`
- `cockpit/trade_permission.py` (Ace advisory stance text)
- `cockpit/ace_authority.py` summary metadata

### Remaining compatibility residue

After the migration sweep, the singular `recommended_action` alias is confined
mainly to:

- `cockpit/bucket_conditioned_spine.py` (the raw compatibility alias)
- historical docs describing the deprecation path
- older output artifacts in `outputs/`

## Fix applied

### Contract changes

`bucket_conditioned_spine.py` now emits:

- `diagnostic_posture`
- `advisory_only=True`
- `authority_role="diagnostic_advisory"`
- `recommended_action_status="deprecated_compatibility_alias"`

`recommended_action` remains for back-compat only.

### Consumer changes

- `timeframe_agreement.py` now uses `diagnostic_posture`
- `trade_permission.py` Ace advisory summaries now use `diagnostic_posture`
- `ace_authority.py` summary metadata now stays on `diagnostic_posture`
- execution-flow packet and render surfaces no longer mirror the alias
- tests were migrated off singular `recommended_action`

## Recommendation

1. Treat `diagnostic_posture` as the only canonical descriptive field.
2. Reserve imperative action fields for explicit approval/operator layers only.
3. Keep `recommended_action` only in the raw spine packet until a versioned contract change removes it.
4. After the schema bump, remove the alias in a deliberate versioned contract change rather
   than letting it linger forever like cursed attic wiring.
