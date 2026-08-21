# Cockpit Vector / Authority Audit

Date: 2026-08-02

## Problem statement

The vector score card and bucket-conditioned spine are supposed to be
**diagnostic execution reads**. They are not supposed to claim final authority.

But the cockpit had drifted into double-duty language:

- `authority_self_audit.py` correctly said the score spine is `diagnostic_advisory`
- `bucket_conditioned_spine.py` still emitted `recommended_action`
- `execution_flow_view.py` rendered `PRIMARY EXECUTION AUTHORITY`
- `execution_card_builder.py` summarized the score lane as `we_are_doing_this`

So the packet doctrine and the human-facing language were disagreeing.

## Root cause

This was mainly a **language/contract leak**, not a math leak.

The underlying score and bucket logic can stay useful as a read, but the UI and
adjudication packet were presenting that read like a decision source.

## Fix applied

### Packet changes

`bucket_conditioned_spine.py`
- kept `recommended_action` for compatibility
- added `diagnostic_posture`
- added `advisory_only=True`
- added `authority_role="diagnostic_advisory"`

`execution_card_builder.py`
- execution flow now carries the advisory metadata
- authority adjudication now publishes canonical `cockpit_read`
- legacy `we_are_doing_this` remains as a compatibility alias
- summaries now say `Cockpit read posture`

### Render changes

`execution_flow_view.py`
- `PRIMARY EXECUTION AUTHORITY` -> `DIAGNOSTIC EXECUTION READ`
- `Bucket-conditioned spine action` -> `Bucket-conditioned diagnostic posture`
- `AUTHORITY ADJUDICATION` -> `COCKPIT READ ADJUDICATION`
- `Authority action` -> `Cockpit read posture`

`make_cockpit.py`
- console summary now prints `posture=` instead of `action=`

## Why this is the right fix

It preserves compatibility for downstream consumers that still read
`recommended_action`, while making the canonical cockpit language honest:

- score spine = read
- approval/operator = authority

No fake purity refactor. Just less doctrinal lying.

## Follow-up

1. Gradually migrate downstream consumers from `recommended_action` to
   `diagnostic_posture` where the field is only descriptive.
2. Reserve truly imperative action fields for approval/operator layers only.
3. If the repo wants a hard break later, deprecate `we_are_doing_this` in favor
   of `cockpit_read` after downstream surfaces are updated.
