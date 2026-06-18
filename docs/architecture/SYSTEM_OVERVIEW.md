# SharpEdge System Overview

SharpEdge is split into bounded systems so agents and humans can load only the
context they need.

## Golden path

```text
Signal generation
  → contract packaging
  → native rendering
  → optional Android execution
  → broker handoff only after approval
```

## System boundaries

```text
SharpEdge-System
  owns signal generation, trade gate analytics, operator artifacts

Phone Companion
  owns moving contracts between systems and preserving request/view evidence

SharpEdge-Android
  owns native Android rendering of SharpEdge contracts

DroidPuppy
  owns optional Android execution/observation utilities

SharpEdge-Robinhood-Bridge
  owns broker capability routing and approval-gated execution planning
```

## Canonical trading viewer flow

```text
SharpEdge-System outputs/signal.json
  → phone_companion/requests/golden_loop_request.json
  → phone_companion/views/trading/golden_loop_view_model.json
  → SharpEdge-Android app/src/main/assets/sample_signal.json during local builds
  → native Compose Trade Gate card
```

## What not to do

- Do not put Android UI logic in SharpEdge-System.
- Do not put trading analytics in SharpEdge-Android.
- Do not let Phone Companion recompute trade authority.
- Do not rely on Brave DevTools/CDP as the primary viewer proof.
- Do not treat generated proof artifacts as stable source code.

## Agent onboarding

Read these first:

1. `docs/architecture/OWNERSHIP_MAP.md`
2. `docs/architecture/SYSTEM_OVERVIEW.md`
3. `docs/architecture/CURRENT_STATE.md`
4. `docs/architecture/REPO_INVENTORY.md`
5. `docs/architecture/CONTRACTS.md`

Then work only inside the repo/layer that owns the requested change.
