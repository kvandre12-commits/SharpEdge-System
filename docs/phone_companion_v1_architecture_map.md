# Phone Companion v1 Architecture Map

## One-sentence definition

**Phone Companion v1 is the phone-facing shell for SharpEdge-System: it reads approved domain artifacts, renders a focused mobile operator surface, invokes DroidPuppy for on-device actions, and captures observations back into state.**

It is **not** the domain brain.
It is **not** the contract authority.
It is **not** the Android executor itself.

It is the phone-native bridge between SharpEdge artifacts and real operator/device action.

---

## Control hierarchy

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

This is the canonical execution spine.

### Layer meanings

| Layer | Job | Must not do |
|---|---|---|
| Intent | express operator goal | choose Android taps or UI details |
| Contract | define structured request/result | decide domain truth |
| Code Puppy | plan, build artifacts, validate state | impersonate the device |
| DroidPuppy | choose concrete Android execution path | invent business intent |
| Phone Action | perform real side effect | reinterpret authority |
| Observation | record what actually happened | silently mutate upstream intent |

If any layer impersonates another, the system gets sloppy fast.

---

## V1 purpose

Phone Companion v1 exists to make SharpEdge usable from the phone without smashing together:

- domain logic
- presentation logic
- Android execution logic
- observation/audit logic

V1 should support two concrete lanes:

1. **Trading lane**
   - open/read existing cockpit and command deck artifacts
   - present operator-facing summaries on phone
   - route safe operator actions

2. **Section-gap ops lane**
   - open/read section-gap pilot artifacts
   - show reference vs observation state
   - route alert packaging to real channels
   - capture acknowledgement / follow-up observations

The key is reuse of the **phone shell pattern**, not merging the **domains**.

---

## V1 non-goals

Phone Companion v1 does **not**:

- replace SharpEdge domain logic
- replace DroidPuppy Android tooling
- grant new permissions
- auto-approve live trade execution
- become a whole-store omniscience engine
- mix trading and section-gap logic in one renderer or one state model

A fancy lie is still a lie.
V1 must stay boring, explicit, and inspectable.

---

## System boundary map

```text
SharpEdge-System/                domain artifacts + app-specific render/build logic
  cockpit/                       trading-only builders and pages
  ops/section_gap/               section-gap-only builders and state
  phone_companion/               phone-facing shell and shared mobile patterns

Code Puppy                       planning / orchestration / artifact generation runtime
DroidPuppy                       Android-native action layer
Android apps                     Brave / Teams / Outlook / camera / photos
Observation loop                 logs, events, acknowledgements, support artifacts
```

### Ownership boundary

| Area | Owns |
|---|---|
| `cockpit/` | trading artifact generation |
| `ops/section_gap/` | section-gap domain truth and pilot artifacts |
| `phone_companion/` | phone-facing view models, launch wrappers, mobile flow conventions |
| Code Puppy | planning, coordination, artifact assembly |
| DroidPuppy | Android intents, app launches, share flows, UI fallback |

---

## Repo hierarchy for Phone Companion v1

```text
SharpEdge-System/
  cockpit/                          # trading domain only
  ops/
    section_gap/                    # section-gap domain only
  phone_companion/
    README.md
    contracts/
      phone_companion_request_v1.json
      phone_companion_observation_v1.json
    views/
      trading/
      section_gap/
    launchers/
      run_phone_companion_trading.sh
      run_phone_companion_section_gap.sh
    observations/
    configs/
      phone_companion_routes.example.json
```

V1 does not need every file on day one, but this is the right fence line.

---

## Layer-by-layer architecture

## 1. Intent layer

This is the operator or upstream automation goal.

Examples:
- `open trading cockpit on phone`
- `show section-gap pilot dashboard`
- `route latest section-gap alert to Teams`
- `capture acknowledgement of alert`

Intent should be plain, explicit, and domain-aware.

### Intent examples

```text
open_trading_dashboard
open_section_gap_dashboard
send_section_gap_alert
record_alert_acknowledgement
```

Intent is the **WHAT**.
Not the app choice.
Not the button press.

---

## 2. Contract layer

This is the structured request/result shape between layers.

Phone Companion v1 should define at least two contract families:

### A. Request contract
Purpose: describe what the phone companion should do.

Suggested fields:
- `request_id`
- `intent_type`
- `domain`
- `artifact_inputs`
- `preferred_view`
- `preferred_channel`
- `requires_confirmation`
- `constraints`

### B. Observation contract
Purpose: describe what actually happened after phone action.

Suggested fields:
- `observation_id`
- `request_id`
- `status`
- `action_type`
- `target`
- `started_at`
- `ended_at`
- `result_summary`
- `artifacts_created`
- `operator_notes`

### Contract rule

Contracts are the seam.
They keep phone behavior swappable and testable.

---

## 3. Code Puppy layer

Code Puppy is responsible for:
- reading domain artifacts
- selecting the correct phone companion flow
- building view models
- validating safety / readiness
- producing launch-ready requests for DroidPuppy

Code Puppy is where planning and compression happen.
It should do the thinking once, clearly.

### Code Puppy responsibilities by lane

#### Trading lane
- read existing cockpit / command-deck / operator artifacts
- decide which mobile surface should open
- package operator-facing summaries without changing authority

#### Section-gap lane
- read section-gap pilot config and latest event artifacts
- decide whether to show normal / low / empty / needs_review
- package alert text and route metadata
- build phone-facing dashboard state

### Code Puppy must not
- hardcode Android tap paths into domain builders
- bypass contract blockers
- treat dashboard readiness as permission authority

---

## 4. DroidPuppy layer

DroidPuppy is the Android-native execution bridge.
It decides **HOW** to satisfy the request on the phone.

Examples:
- open Brave to local dashboard URL
- share alert text to Teams or Outlook
- launch camera/photos if review is needed
- use UI steering only when structured handoff is unavailable

### Preferred DroidPuppy action order

1. direct handoff / intent launch
2. URL open in Brave
3. structured share to target app
4. UI fallback only if needed

### DroidPuppy capabilities likely involved

- browser launch / Brave open
- Android intent send
- handoff text / URL
- UI dump / UI tap fallback
- support bundle and issue draft on failure

DroidPuppy owns execution strategy.
It does not decide business truth.

---

## 5. Phone Action layer

This is the real side effect.

Examples:
- Brave opens `cockpit.html`
- Brave opens a section-gap dashboard page
- Teams share sheet opens with alert text
- Outlook compose opens with packaged alert
- camera/photos review path opens

This layer is where “the phone actually did the thing.”

### V1 actions worth supporting

#### Trading
- open cockpit
- open command deck
- open operator review artifact

#### Section-gap
- open pilot dashboard
- send alert to Teams
- send alert to Outlook
- open evidence review path

Phone action is concrete, observable, and boring.
Good.

---

## 6. Observation layer

Observation closes the loop.

Without observation, Phone Companion becomes a launch toy instead of a real operator system.

### Observation examples
- dashboard opened successfully
- share target opened successfully
- alert draft packaged
- operator marked event acknowledged
- review remained unresolved
- fallback path used because direct handoff failed

### Observation outputs
V1 should persist or emit:
- action success/failure
- chosen target app
- timestamp
- any generated support artifact path
- optional operator note

Observation is the truth of what happened, not what we hoped happened.

---

## Domain separation rule

Phone Companion v1 is **shared shell**, not **shared domain soup**.

### Trading artifacts stay here
- `cockpit/`
- `scripts/agents/`
- `outputs/*trading/operator*`

### Section-gap artifacts stay here
- `ops/section_gap/`
- section configs
- section events
- section dashboard data

### Phone Companion reads both, but owns neither
That is the whole trick.

---

## Canonical v1 flows

## Flow A — trading dashboard open

```text
Intent:
  open trading dashboard

Contract:
  request { domain=trading, preferred_view=command_deck }

Code Puppy:
  read trading artifacts -> build phone view request

DroidPuppy:
  open Brave to local command deck URL

Phone Action:
  Brave shows trading page

Observation:
  dashboard_opened / failed_to_open
```

## Flow B — section-gap dashboard open

```text
Intent:
  open section-gap pilot dashboard

Contract:
  request { domain=section_gap, pilot_id=..., preferred_view=pilot_dashboard }

Code Puppy:
  read latest pilot artifacts -> build dashboard state

DroidPuppy:
  open Brave to local pilot dashboard URL

Phone Action:
  Brave shows pilot section state

Observation:
  dashboard_opened / failed_to_open
```

## Flow C — section-gap alert route

```text
Intent:
  send section-gap alert

Contract:
  request { domain=section_gap, preferred_channel=teams }

Code Puppy:
  package alert text + owner route + artifact refs

DroidPuppy:
  use structured text handoff to Teams, else fallback

Phone Action:
  Teams share or compose target opens

Observation:
  alert_packaged / alert_handoff_failed / needs_manual_completion
```

---

## Data and artifact map

### Trading inputs
- cockpit HTML and chart artifacts
- command deck HTML
- operator brief / dashboard artifacts
- approval / workflow state artifacts

### Section-gap inputs
- pilot config
- reference photos
- observation records
- latest section event artifact
- alert packaging artifact
- repeat-history artifact

### Phone Companion outputs
- request artifacts
- rendered phone view models
- launcher-specific state
- observation records

Important:
Phone Companion should prefer **reading built artifacts** over recomputing domain logic.

---

## Presentation architecture

Phone Companion v1 presentation should be driven by **view models**, not by direct access to random scripts.

### Recommended view families

#### `views/trading/`
- cockpit launcher view
- command deck launcher view
- operator review summary view

#### `views/section_gap/`
- pilot section dashboard view
- alert review view
- acknowledgement view

### Presentation rule

Renderers render.
Builders decide.
Launchers launch.
Do not let one file do all three unless you enjoy repo exorcisms.

---

## Launcher architecture

Launchers are thin wrappers.
They should:
- serve static files when needed
- ensure the page exists before opening Brave
- avoid stale/blank pages
- optionally skip browser auto-open for debugging
- emit simple observation status

### Launcher examples
- `run_phone_companion_trading.sh`
- `run_phone_companion_section_gap.sh`

These should imitate the good parts of `run_cockpit.sh` and `run_command_deck.sh` while dropping domain entanglement.

---

## Observation and audit model

V1 should create a small observation trail for every phone-companion action.

### Minimum fields
- request id
- domain
- chosen launcher
- chosen app target
- success/failure
- timestamps
- created artifact paths
- optional operator note

### Why this matters
This gives:
- supportability
- retry clarity
- operator trust
- historical pattern review

Without it, failures become ghost stories.

---

## Safety and authority rules

### Rule 1
The contract authority stays upstream.
Phone Companion displays or routes; it does not mint permissions.

### Rule 2
If approval or blocker artifacts say stop, Phone Companion stops escalation.

### Rule 3
For section-gap ops, classification confidence is not the same thing as routing permission.

### Rule 4
UI fallback is last resort, not first design choice.

### Rule 5
Observation must always be possible, even when action fails.

---

## Runtime topology

```text
SharpEdge artifact builders
        |
        v
Phone Companion view/build layer
        |
        v
Phone Companion launcher
        |
        v
DroidPuppy Android action layer
        |
        v
Brave / Teams / Outlook / camera / photos
        |
        v
Observation record back into SharpEdge state
```

---

## Initial file ownership plan

### Existing files to reuse as patterns
- `cockpit/make_cockpit.py`
- `cockpit/make_command_deck.py`
- `cockpit/run_cockpit.sh`
- `cockpit/run_command_deck.sh`

### Existing files that remain authoritative upstream
- `scripts/agents/operator_brief.py`
- `scripts/agents/morning_open_dashboard.py`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `outputs/execution_plan.json`
- `outputs/journal.json`

### New Phone Companion v1 home
- `phone_companion/`

### Section-gap domain home
- `ops/section_gap/`

---

## Recommended v1 implementation order

1. create `phone_companion/` home and README
2. define request + observation contracts
3. implement trading launcher wrapper using existing cockpit/deck outputs
4. implement section-gap launcher wrapper using pilot artifacts
5. add observation logging for both lanes
6. add alert routing packaging for section-gap
7. only then consider smarter review/classification UX

---

## Final architecture rule

**Phone Companion v1 is a phone-facing shell that sits between SharpEdge artifacts and DroidPuppy execution. It uses the hierarchy `Intent ↓ Contract ↓ Code Puppy ↓ DroidPuppy ↓ Phone Action ↓ Observation`, keeps domain truth upstream, and keeps Android execution downstream.**
