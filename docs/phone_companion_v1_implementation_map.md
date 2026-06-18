# Phone Companion v1 Implementation Map

This document turns the architecture into concrete repo structure, contracts, required artifacts, and cross-layer data flow.

## Canonical hierarchy

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

---

## 1. Folder structure

```text
SharpEdge-System/
  cockpit/                                  # trading domain only
  ops/
    section_gap/                            # section-gap domain only
  phone_companion/
    README.md
    contracts/
      phone_companion_request_v1.json
      phone_companion_view_model_v1.json
      phone_companion_observation_v1.json
    configs/
      README.md
      phone_companion_routes.example.json
    views/
      trading/
        README.md
      section_gap/
        README.md
    launchers/
      README.md
      run_phone_companion_trading.sh
      run_phone_companion_section_gap.sh
    observations/
      README.md
```

### Folder responsibilities

| Folder | Responsibility |
|---|---|
| `cockpit/` | trading artifact generation and pages |
| `ops/section_gap/` | section-gap domain truth and pilot artifacts |
| `phone_companion/contracts/` | structured seams between layers |
| `phone_companion/configs/` | delivery routing and target preferences |
| `phone_companion/views/` | phone-facing, domain-specific view packaging |
| `phone_companion/launchers/` | thin launch and delivery wrappers |
| `phone_companion/observations/` | result records from phone actions |

---

## 2. Required contracts

## A. `phone_companion_request_v1`
Purpose:
- the structured request entering Phone Companion from planning/build logic

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

Used between:
- **Intent → Contract → Code Puppy**
- and as the envelope for **Code Puppy → DroidPuppy** flow selection

## B. `phone_companion_view_model_v1`
Purpose:
- the mobile-facing view state built before launch/delivery

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

Used between:
- **Code Puppy → phone_companion/views/**
- and indirectly into launcher selection

## C. `phone_companion_observation_v1`
Purpose:
- the structured record of what happened after an attempted phone action

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

Used between:
- **Phone Action → Observation**
- and for support/audit/retry analysis downstream

---

## 3. Required artifacts

## Trading lane inputs
Phone Companion does not invent these. It consumes them.

Required upstream artifacts:
- `cockpit/cockpit.html`
- `cockpit/command_deck.html` when present
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `outputs/execution_plan.json`
- `outputs/journal.json`

## Section-gap lane inputs
Required upstream artifacts:
- `ops/section_gap/pilot_config*.json`
- `ops/section_gap/refs/*`
- `ops/section_gap/observations/*.json`
- `ops/section_gap/events/*.json`
- future section-gap dashboard html/json artifacts

## Phone Companion internal artifacts
Required internal artifacts:
- request artifact conforming to `phone_companion_request_v1`
- view model artifact conforming to `phone_companion_view_model_v1`
- observation artifact conforming to `phone_companion_observation_v1`
- route config via `configs/phone_companion_routes.example.json`

## DroidPuppy delivery artifacts
Optional but expected outputs depending on action:
- opened browser target
- launched share target
- support bundle path on failure
- issue-draft artifact on recovery path

---

## 4. Data flow between layers

## Layer 1 — Intent
Input:
- operator goal or upstream automation goal

Output:
- plain-language requested outcome

Examples:
- `open_trading_dashboard`
- `open_section_gap_dashboard`
- `send_section_gap_alert`

## Layer 2 — Contract
Input:
- intent statement

Output:
- `phone_companion_request_v1`

Flow:
```text
Intent
  -> normalize goal
  -> package request contract
```

## Layer 3 — Code Puppy
Input:
- request contract
- authoritative upstream artifacts

Output:
- selected flow
- view model
- launcher target
- action packaging

Flow:
```text
request contract
  + upstream artifacts
  -> validate blockers/readiness
  -> build phone_companion_view_model_v1
  -> choose DroidPuppy delivery path
```

## Layer 4 — DroidPuppy
Input:
- chosen launcher target
- chosen channel/browser/app
- packaged data from Code Puppy

Output:
- Android-native execution plan

Flow:
```text
view model + launch intent
  -> URL open / text share / Android intent / UI fallback
```

## Layer 5 — Phone Action
Input:
- DroidPuppy execution path

Output:
- real side effect

Examples:
- Brave opened a dashboard
- Teams compose/share opened
- Outlook compose opened
- fallback UI steering path invoked

## Layer 6 — Observation
Input:
- action result

Output:
- `phone_companion_observation_v1`

Flow:
```text
phone action result
  -> summarize outcome
  -> record timestamps, target, status, fallback use
  -> persist observation artifact
```

---

## 5. End-to-end examples

## Example A — trading dashboard

```text
Intent:
  open_trading_dashboard

Contract:
  phone_companion_request_v1

Code Puppy:
  reads operator/trading artifacts
  builds trading view model
  selects Brave launcher

DroidPuppy:
  opens local dashboard URL in Brave

Phone Action:
  dashboard visible on phone

Observation:
  phone_companion_observation_v1 recorded
```

## Example B — section-gap alert

```text
Intent:
  send_section_gap_alert

Contract:
  phone_companion_request_v1

Code Puppy:
  reads pilot config + latest event
  builds alert view model and channel package

DroidPuppy:
  hands off text to Teams or Outlook

Phone Action:
  share/compose target opens

Observation:
  success/failure/fallback recorded
```

---

## 6. Rules that keep this sane

1. Phone Companion reads domain truth; it does not mint domain truth.
2. Contracts are mandatory seams, not decorative JSON.
3. Launchers stay thin.
4. UI fallback is last resort.
5. Every meaningful action emits an observation.
6. Trading and section-gap remain separate domains sharing one phone shell.

## One-line summary

**The Phone Companion v1 repo skeleton is a shared phone shell with three required contracts, authoritative upstream artifacts per domain, and a one-way execution/data flow of `Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation`.**
