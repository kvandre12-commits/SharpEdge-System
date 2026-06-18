# Section Gap Hierarchy Plan

## Why this exists

SharpEdge now has enough power to do many things.
That is exactly why the hierarchy must get stricter, not looser.

The current `cockpit/` lane was built for **market decision support**.
It is a good implementation pattern, but it is **not** the right domain home for store-floor section monitoring.

So the rule is simple:

- reuse the **pattern**
- do **not** mix the **domain**

## Control path mantra

```text
Intent ↓ Contract ↓ Code Puppy ↓ DroidPuppy ↓ Phone Action ↓ Observation
```

This is the clean execution spine.

- **Intent** — what the operator wants
- **Contract** — the structured shape of that request/result
- **Code Puppy** — reasoning, planning, artifact generation
- **DroidPuppy** — Android-native execution and routing
- **Phone Action** — the real side effect on-device
- **Observation** — what actually happened, captured back into state

If one layer reaches upward and tries to impersonate another, the architecture gets sloppy fast.

## Hard separation rule

### Keep in `cockpit/`
Only trading-specific logic:
- market data fetch
- signal generation
- command deck / cockpit rendering for trading
- operator trading review artifacts

### Put in `ops/section_gap/`
Only store-floor section monitoring logic:
- fresh map / section definitions
- reference photos
- observed photos
- section-gap event generation
- ownership routing
- alert packaging
- repeat-event analytics

If a file needs both SPY gamma walls and empty produce sections, the file is cursed and wrong.

## The hierarchy

### L0 — Assets / Ground Truth
This is raw reality.

Examples:
- fresh map output
- reference photos you already took
- future Zamboni or route photos
- manually confirmed statuses

This layer answers:
- what place are we talking about?
- what does normal look like?
- what evidence do we have?

### L1 — Domain Model
This is the first clean abstraction over reality.

Objects for the section-gap system should be things like:
- `Section`
- `ReferenceView`
- `Observation`
- `SectionGapEvent`
- `OwnerRoute`
- `AlertRecord`

This layer must know nothing about Brave, Teams buttons, or pretty dashboards.
It only knows the truth model.

### L2 — Decision / Classification
This is where we convert observations into operational meaning.

Examples:
- `normal`
- `low`
- `empty`
- `needs_review`

And:
- confidence
- severity
- suspected issue type
- who should own the fix

This layer is the brain of the section pilot.
It should be deterministic and boring before it becomes fancy.

### L3 — Presentation
This is where the existing cockpit/command-deck pattern becomes useful.

The dashboard should render:
- pilot spot
- reference image
- latest observation
- status
- owner
- alert state
- repeat count

Important:
this layer reads artifacts from L1/L2.
It does not invent them.

### L4 — Delivery / Orchestration
This is the DroidPuppy / Android lane.

Examples:
- open the dashboard in Brave
- share an alert to Teams
- route to Outlook
- use UI steering if needed

This layer is how work moves.
It is not where truth is decided.

## Mapping current SharpEdge assets

### Existing asset: `cockpit/make_cockpit.py`
Use as a **pattern reference only**.

What to steal:
- static artifact generation
- one-command rebuild flow
- browser-friendly single-page output

What not to steal:
- market data assumptions
- trading terminology
- market-specific signal logic

### Existing asset: `cockpit/make_command_deck.py`
This is the best template for a section pilot dashboard.

Why:
- already thinks in `artifact in -> dashboard out`
- already designed as a compressed operator surface
- already phone/browser friendly

Repurpose the pattern, not the market semantics.

### Existing assets: `run_cockpit.sh` and `run_command_deck.sh`
These belong to L4 delivery.

They are launch wrappers:
- local static server
- open Brave
- refresh artifacts in a loop

That behavior is reusable.
The trading-specific filenames are not.

## Proposed repo boundary

```text
SharpEdge-System/
  cockpit/                  # trading only
  ops/
    section_gap/
      README.md
      pilot_config.example.json
      refs/
      observations/
      events/
      build_section_signal.py
      build_section_dashboard.py
      run_section_dashboard.sh
```

## First pilot constraints

The pilot stays intentionally tiny:
- one store
- one known spot
- 1-3 reference images
- 1 alert destination
- manually review borderline detections

This is not a whole-store omniscience engine.
It is one trustworthy monitored zone.

## Data flow for v1

```text
reference photos + observed photos
  -> section observation record
  -> status classification
  -> owner routing
  -> alert artifact
  -> dashboard artifact
  -> repeat-event history
```

## What not to mix

Do not mix these concerns in one file:
- image classification
- owner mapping
- HTML rendering
- Android launch logic

That is how prototypes become haunted.

## Recommended next implementation order

1. define pilot config
2. define section-gap event schema
3. build manual observation -> status artifact
4. render section dashboard
5. add alert packaging
6. only then add smarter image logic

## One-sentence rule

**Trading cockpit stays trading. Store-floor section monitoring becomes its own ops lane, borrowing the artifact-and-dashboard pattern without inheriting the domain mess.**
