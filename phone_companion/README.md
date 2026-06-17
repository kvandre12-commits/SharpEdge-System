# Phone Companion

Phone Companion is the phone-facing shell for SharpEdge-System.

It exists to keep mobile/operator presentation and launch behavior separate from:
- trading domain logic in `cockpit/`
- section-gap domain logic in `ops/section_gap/`
- Android execution logic in DroidPuppy

## Hierarchy

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

## Folder intent

Planned subfolders:

- `contracts/` — request and observation schemas
- `views/` — phone-facing view builders by domain
- `launchers/` — local server + Brave / share wrappers
- `observations/` — action result records
- `configs/` — route and target config

## Rules

- this folder owns **phone shell behavior**, not domain truth
- it may read domain artifacts, but must not recompute domain authority
- it may request DroidPuppy actions, but must not impersonate DroidPuppy
- it must emit observations for every meaningful action

If a file here starts talking like SPY gamma and empty produce sections at the same time, bonk it.
