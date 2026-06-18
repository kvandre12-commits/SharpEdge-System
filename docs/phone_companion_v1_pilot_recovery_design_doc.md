# Phone Companion v1 Pilot Recovery Design Document

## Read this first

If you have lost all context, start here.

This document is the recovery map for **Phone Companion v1**.
It explains what the system is, what it is not, where the important files live,
how the layers talk to each other, and how to restart work without inventing a
fresh pile of architectural nonsense.

---

## One-sentence definition

**Phone Companion v1 is the phone-facing shell for SharpEdge-System: it reads approved domain artifacts, packages mobile-facing views, invokes DroidPuppy for on-device work, and records observations about what actually happened.**

That is the whole game.
Not more. Not less.

---

## The canonical execution spine

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

Read it like this:

- **Intent** — what the operator wants
- **Contract** — the structured shape of the request/result
- **Code Puppy** — reasoning, planning, artifact selection, view-model building
- **DroidPuppy** — Android-native execution strategy
- **Phone Action** — the real side effect on the device
- **Observation** — the record of what actually happened

### Why this matters

This spine is the anti-spaghetti rule.

If any layer impersonates another, the system becomes sloppy:
- if Code Puppy starts pretending to be DroidPuppy, device logic leaks upward
- if DroidPuppy starts deciding business truth, execution leaks into domain logic
- if Phone Action happens without Observation, the system becomes a ghost story

---

## What Phone Companion v1 is for

Phone Companion v1 exists to make SharpEdge usable from the phone **without**
mixing together:

- domain truth
- phone presentation
- Android execution
- audit/observation

It is a **shared phone shell** for two lanes:

### Lane 1 — trading
Use existing trading artifacts from `cockpit/` and `outputs/` to:
- open cockpit views on the phone
- open command deck views on the phone
- compress operator-facing summaries into a mobile-friendly launch path

### Lane 2 — section-gap ops
Use pilot artifacts from `ops/section_gap/` to:
- show the state of one monitored section
- compare reference vs observed state
- package alerts to Teams or Outlook
- record acknowledgement and follow-up observations

### Core design rule

**Reuse the phone shell pattern. Do not merge the domains.**

Trading stays trading.
Section-gap stays section-gap.
Phone Companion sits between them and the phone.

---

## What Phone Companion v1 is not

Phone Companion v1 is **not**:

- the trading domain brain
- the section-gap classification authority
- the Android executor itself
- a permission mint
- an auto-trader
- a whole-store omniscience engine
- a folder for random mobile experiments

If you catch it trying to do two or three of those jobs at once, bonk it.

---

## System boundaries

```text
SharpEdge-System/
  cockpit/           trading-only domain builders and pages
  ops/section_gap/   section-gap-only domain truth and pilot artifacts
  phone_companion/   shared phone-facing shell

Code Puppy           planning / artifact assembly / view-model selection
DroidPuppy           Android-native execution layer
Android apps         Brave / Teams / Outlook / camera / photos
Observation          result records and support artifacts
```

### Ownership summary

| Area | Owns |
|---|---|
| `cockpit/` | trading artifacts and render/build logic |
| `ops/section_gap/` | section-gap domain artifacts and pilot truth |
| `phone_companion/` | phone shell contracts, view packaging, launcher conventions, observations |
| Code Puppy | planning, validation, view-model generation, flow selection |
| DroidPuppy | Android launch/share/UI execution |

### Most important boundary

**Phone Companion may read domain truth, but it must not become domain truth.**

---

## Where the important files live

## Architecture docs

Read in this order if you need to recover context fast:

1. `docs/phone_companion_v1_pilot_recovery_design_doc.md` ← this file
2. `docs/phone_companion_v1_architecture_map.md`
3. `docs/phone_companion_v1_implementation_map.md`
4. `docs/section_gap_hierarchy_plan.md`

## Phone Companion home

```text
phone_companion/
  README.md
  contracts/
    phone_companion_request_v1.json
    phone_companion_view_model_v1.json
    phone_companion_observation_v1.json
  configs/
    phone_companion_routes.example.json
  views/
    trading/
    section_gap/
  launchers/
    run_phone_companion_trading.sh
    run_phone_companion_section_gap.sh
  observations/
```

## Trading domain inputs

Look here for trading-side artifacts and patterns:

- `cockpit/make_cockpit.py`
- `cockpit/make_command_deck.py`
- `cockpit/run_cockpit.sh`
- `cockpit/run_command_deck.sh`
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `outputs/execution_plan.json`
- `outputs/journal.json`

## Section-gap domain inputs

Look here for section-gap pilot work:

- `ops/section_gap/README.md`
- `ops/section_gap/pilot_config.example.json`
- `ops/section_gap/refs/`
- `ops/section_gap/observations/`
- `ops/section_gap/events/`

---

## Folder-by-folder purpose

## `phone_companion/contracts/`
This is the seam between layers.

Required contracts:
- `phone_companion_request_v1.json`
- `phone_companion_view_model_v1.json`
- `phone_companion_observation_v1.json`

Contracts are not decorative JSON.
They are what keeps the shell testable and swappable.

## `phone_companion/configs/`
This holds non-authoritative routing preferences.

Examples:
- which launcher is default for trading
- which browser/channel is preferred
- which package name maps to Teams/Outlook/Brave

Config here steers delivery behavior.
It does **not** grant authority.

## `phone_companion/views/`
This is where Code Puppy-facing mobile view packaging belongs.

Subfolders:
- `views/trading/`
- `views/section_gap/`

These folders may read artifacts and produce view state.
They must not redefine authority or invent new domain truth.

## `phone_companion/launchers/`
These are thin wrappers.

Their job:
- ensure required artifacts exist
- optionally serve a local page
- open Brave or hand off to a share target
- emit an observation record

They should stay dumb and small.
A launcher is not a business brain.

## `phone_companion/observations/`
This holds result records from phone actions.

If there is no observation, there is no trustworthy record of what happened.

---

## Required contracts

## 1. Request contract

File:
- `phone_companion/contracts/phone_companion_request_v1.json`

Purpose:
- the structured request entering Phone Companion

Minimum meaning:
- what is being requested
- which domain it belongs to
- what artifacts are needed
- what delivery preference applies

Required fields:
- `request_id`
- `intent_type`
- `domain`
- `artifact_inputs`

Optional fields:
- `preferred_view`
- `preferred_channel`
- `requires_confirmation`
- `constraints`
- `notes`

## 2. View-model contract

File:
- `phone_companion/contracts/phone_companion_view_model_v1.json`

Purpose:
- the mobile-facing state produced after planning but before launch

Required fields:
- `view_id`
- `request_id`
- `domain`
- `view_type`
- `headline`
- `status`
- `data`

Optional fields:
- `actions`
- `artifacts`
- `warnings`
- `notes`

This is what the phone shell should consume, not raw business guts from random scripts.

## 3. Observation contract

File:
- `phone_companion/contracts/phone_companion_observation_v1.json`

Purpose:
- the record of what happened after attempting a phone action

Required fields:
- `observation_id`
- `request_id`
- `status`
- `action_type`
- `target`
- `started_at`
- `ended_at`

Optional fields:
- `result_summary`
- `artifacts_created`
- `operator_notes`
- `fallback_used`

This is how failures stop becoming folklore.

---

## Required artifacts by lane

## Trading lane

Phone Companion should **consume**, not invent, the following upstream artifacts:

- `cockpit/cockpit.html`
- `cockpit/command_deck.html` when present
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `outputs/execution_plan.json`
- `outputs/journal.json`

### Trading lane rule

Phone Companion can display or route based on these artifacts.
It cannot overrule them.

## Section-gap lane

Phone Companion should consume:

- `ops/section_gap/pilot_config*.json`
- `ops/section_gap/refs/*`
- `ops/section_gap/observations/*.json`
- `ops/section_gap/events/*.json`
- future section-gap dashboard artifacts

### Section-gap lane rule

Phone Companion displays or routes what the section-gap lane has already decided.
It does not silently recalculate the domain truth in the launcher.

## Internal Phone Companion artifacts

Phone Companion itself should produce or carry:

- request artifact
- view-model artifact
- observation artifact
- routing config

---

## End-to-end layer data flow

## Layer 1 — Intent

Input:
- operator goal
- upstream automation goal

Output:
- a clear requested outcome

Examples:
- `open_trading_dashboard`
- `open_section_gap_dashboard`
- `send_section_gap_alert`
- `record_alert_acknowledgement`

Intent is always the **WHAT**.
Never the tap path.

## Layer 2 — Contract

Input:
- intent statement

Output:
- `phone_companion_request_v1`

Flow:

```text
Intent
  -> normalize the goal
  -> package the request contract
```

## Layer 3 — Code Puppy

Input:
- request contract
- authoritative upstream artifacts

Output:
- chosen flow
- built view model
- launcher target
- packaged data for DroidPuppy

Flow:

```text
request contract
  + upstream artifacts
  -> validate blockers and readiness
  -> build phone_companion_view_model_v1
  -> choose delivery strategy
```

This is where thinking happens.
Do it once, clearly.

## Layer 4 — DroidPuppy

Input:
- chosen launcher target
- chosen channel/browser/app
- packaged request or view state

Output:
- Android-native execution path

Flow:

```text
view model + launch intent
  -> URL open / text share / Android intent / UI fallback
```

Preferred order:
1. direct handoff
2. URL open in Brave
3. structured share
4. UI fallback only if needed

## Layer 5 — Phone Action

Input:
- DroidPuppy execution path

Output:
- real device side effect

Examples:
- Brave opened a dashboard
- Teams compose/share opened
- Outlook compose opened
- camera/photos review path opened

## Layer 6 — Observation

Input:
- action result

Output:
- `phone_companion_observation_v1`

Flow:

```text
phone action result
  -> summarize what happened
  -> record timestamps, target, status, fallback use
  -> persist observation artifact
```

Observation is truth.
Not intention.

---

## Two canonical v1 flows

## Flow A — trading dashboard open

```text
Intent:
  open_trading_dashboard

Contract:
  phone_companion_request_v1

Code Puppy:
  read trading artifacts
  build trading view model
  choose trading launcher

DroidPuppy:
  open Brave to the chosen local page

Phone Action:
  trading dashboard is visible on the phone

Observation:
  success/failure recorded
```

## Flow B — section-gap alert route

```text
Intent:
  send_section_gap_alert

Contract:
  phone_companion_request_v1

Code Puppy:
  read pilot config + latest section event
  build alert-facing view model and channel package

DroidPuppy:
  hand off text to Teams or Outlook

Phone Action:
  the share/compose target opens on-device

Observation:
  success/failure/fallback recorded
```

---

## What to build next

If you are reviving the pilot, the safest implementation order is:

1. keep the folder boundaries intact
2. keep the contracts authoritative
3. implement the trading launcher thinly
4. implement the section-gap launcher thinly
5. add observation writing for both
6. only then add smarter UX or smarter section-gap logic

### Why this order

Because launcher chaos without contracts is garbage,
and smart UX without observations is a lie.

---

## What not to mix

Do **not** mix these concerns in one file:
- domain classification
- owner routing
- HTML rendering
- Android launch/share logic
- observation writing

That is how prototypes become haunted and unreadable.

### Special warning

If a file starts talking about both:
- SPY gamma walls
- and empty produce sections

it is cursed and wrong.
Split it.

---

## Failure modes future pilots should expect

### 1. Wrong layer does the job
Symptom:
- business truth hardcoded in launcher scripts

Fix:
- move truth back upstream into artifacts/contracts

### 2. Dashboard opens but no one knows what happened
Symptom:
- visible page, no result trail

Fix:
- add or repair observation writing

### 3. Trading and section-gap start sharing semantics
Symptom:
- one renderer or config file tries to speak both domains at once

Fix:
- keep one shared shell, separate the domain view builders

### 4. UI fallback becomes the default
Symptom:
- every flow depends on brittle taps

Fix:
- restore direct handoff / URL open / structured share preference

### 5. Companion starts acting like a permission engine
Symptom:
- downstream shell tries to overrule upstream blockers

Fix:
- remember: Phone Companion is presentation and routing, not authority

---

## Recovery checklist

If you return later and feel lost, do this in order:

1. read this file
2. read `docs/phone_companion_v1_architecture_map.md`
3. read `docs/phone_companion_v1_implementation_map.md`
4. inspect `phone_companion/contracts/`
5. inspect `phone_companion/launchers/`
6. inspect the domain lane you are touching:
   - `cockpit/` for trading
   - `ops/section_gap/` for section-gap
7. decide which one layer you are changing
8. refuse to “just cram it in” somewhere convenient

That last step matters more than people admit.

---

## Final rule

**Phone Companion v1 is a shared phone shell, not a domain brain. Keep the execution spine clean, keep the domains separate, keep launchers thin, and always record observations.**
