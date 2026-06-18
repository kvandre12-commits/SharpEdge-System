# Phone Companion v1 Golden Loop

## Purpose

The **Golden Loop** is the smallest complete end-to-end workflow that proves
Phone Companion v1 actually works.

It must complete the mission:

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

This loop is intentionally narrow:

- **trading only**
- **one action only**
- **one launcher only**
- **one observation path only**

If this loop does not work, Phone Companion v1 is not operational.

---

## Golden Loop definition

### The one thing it proves

A tired human can request:

> **open the trading dashboard on the phone**

and the system will:

1. turn that intent into a request contract
2. package a trading view-model artifact
3. invoke DroidPuppy to open Brave
4. cause a visible phone action
5. write an observation artifact
6. preserve evidence for later review

That is enough.
Anything larger is not the Golden Loop.

---

# 1. Step-by-step sequence

## Step 1 — Start with a trading intent

Human intent:

```text
open_trading_dashboard
```

Meaning:
- open the existing trading dashboard on the phone
- do not invent new trade logic
- do not place orders
- do not alter authority

---

## Step 2 — Write a request contract

Phone Companion receives a request artifact conforming to:

- `phone_companion_request_v1`

Minimum meaning:
- this is a trading request
- the target artifact is the existing trading dashboard
- Brave is the preferred delivery path

This is the **Contract** layer.

---

## Step 3 — Build one trading view-model artifact

Code Puppy reads the request and upstream trading artifacts, then produces a
single mobile-facing view-model artifact conforming to:

- `phone_companion_view_model_v1`

This is the **Code Puppy** layer.

The view-model does not invent market truth.
It points to existing trading artifacts and packages enough state for launch.

---

## Step 4 — Invoke DroidPuppy through the trading launcher

The trading launcher reads the request and/or view-model artifact and invokes
DroidPuppy using the preferred path:

- open Brave to the selected trading dashboard URL

Preferred target for the Golden Loop:

- `http://127.0.0.1:8777/cockpit.html`

This is the **DroidPuppy** layer.

---

## Step 5 — Cause a visible phone action

Expected phone action:

- Brave opens or comes to foreground
- Brave attempts to open the trading dashboard URL

This is the **Phone Action** layer.

Important truth:
- the Golden Loop proves the phone shell can launch the dashboard path
- it does **not** prove fresh market data or trade readiness
- it does **not** grant permission authority

---

## Step 6 — Write an observation artifact

After the launch attempt, Phone Companion writes an observation artifact
conforming to:

- `phone_companion_observation_v1`

Minimum observation meaning:
- what request was attempted
- what target was used
- when it started
- when it ended
- whether it succeeded or failed
- whether fallback was used

This is the **Observation** layer.

---

## Step 7 — Store evidence for later review

Evidence must survive the moment.

Minimum evidence set:
- request artifact
- view-model artifact
- observation artifact
- launcher stdout/stderr or result summary
- target URL used

If a future pilot cannot reconstruct what happened, the loop is not golden.

---

# 2. Required files

## Existing authoritative upstream files

These already exist and remain upstream truth:

- `cockpit/cockpit.html`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`

## Existing Phone Companion contract files

- `phone_companion/contracts/phone_companion_request_v1.json`
- `phone_companion/contracts/phone_companion_view_model_v1.json`
- `phone_companion/contracts/phone_companion_observation_v1.json`
- `phone_companion/configs/phone_companion_routes.example.json`
- `phone_companion/launchers/run_phone_companion_trading.sh`

## Minimum runtime artifacts the Golden Loop must produce

These are the minimum new runtime evidence artifacts the loop should write:

- `phone_companion/observations/golden_loop_latest.json`
- `phone_companion/observations/history/<timestamp>__golden_loop_observation.json`

## Minimum request/view-model evidence files

The Golden Loop should also persist the exact request and view-model used:

- `phone_companion/observations/history/<timestamp>__golden_loop_request.json`
- `phone_companion/observations/history/<timestamp>__golden_loop_view_model.json`

This keeps the loop reviewable later.

---

# 3. Required folders

## Must exist

- `phone_companion/`
- `phone_companion/contracts/`
- `phone_companion/configs/`
- `phone_companion/launchers/`
- `phone_companion/observations/`

## Must be added or treated as runtime evidence folders

- `phone_companion/observations/history/`

## Upstream required folders

- `cockpit/`
- `outputs/`

The Golden Loop should not depend on `backend/` or `frontend/`.
Those are not trusted v1 surfaces.

---

# 4. Inputs

## Human input

One operator intent:

```text
open_trading_dashboard
```

## Contract input

One request object with these minimum fields:

- `request_id`
- `intent_type = open_trading_dashboard`
- `domain = trading`
- `artifact_inputs`

## Upstream artifact inputs

Minimum upstream inputs:

- `cockpit/cockpit.html`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`

Useful but optional for richer display context:

- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`

## Delivery input

- Brave available on device
- DroidPuppy/browser-open path available
- local cockpit URL known

Recommended Golden Loop URL:

- `http://127.0.0.1:8777/cockpit.html`

---

# 5. Outputs

## Contract/output artifacts

The loop must emit:

1. request artifact
2. view-model artifact
3. observation artifact

## Runtime side effect

The loop must produce one real phone action:

- Brave opens the trading dashboard URL

## Evidence outputs

At minimum:

- latest observation file
- timestamped historical observation file
- timestamped request file
- timestamped view-model file
- result summary containing target URL and status

---

# 6. Success criteria

The Golden Loop is successful only if **all** of the following are true:

1. a trading intent was captured
2. a valid request contract was written
3. a valid trading view-model artifact was written
4. the trading launcher invoked DroidPuppy using the Brave-open path
5. Brave opened or attempted to open the dashboard URL on the phone
6. a valid observation artifact was written
7. the request, view-model, and observation can be reviewed later by request ID
8. the whole sequence can be executed by a tired human in under 10 minutes

## Practical proof of success

A future pilot should be able to answer:

- what did we ask for?
- what URL did we target?
- what launcher path did we use?
- did the phone action happen?
- what evidence files were created?

If those answers are obvious, the loop works.

---

# 7. Failure criteria

The Golden Loop fails if **any** of the following happens:

1. no request contract is created
2. no view-model artifact is created
3. launcher behavior depends on hidden manual interpretation
4. DroidPuppy is not invoked
5. no visible phone action occurs
6. no observation artifact is written
7. observation cannot be linked back to the request
8. evidence is lost after the run
9. the loop requires backend/frontend rescue work
10. a tired human cannot complete the process in under 10 minutes

## Special failure warnings

These count as architectural failure even if a page appears:

- launcher silently bypasses contract checks
- Phone Companion overrides `approval_decision`
- UI fallback becomes the normal path
- trading truth is recomputed inside the launcher

That is fake success. Do not accept it.

---

# 8. Example run

This example assumes the cockpit page already exists and the local static server
is available at:

- `http://127.0.0.1:8777/cockpit.html`

## Example request object

```json
{
  "request_id": "req-trading-golden-001",
  "intent_type": "open_trading_dashboard",
  "domain": "trading",
  "artifact_inputs": [
    "cockpit/cockpit.html",
    "outputs/approval_decision.json",
    "outputs/workflow_state.json"
  ],
  "preferred_view": "cockpit",
  "preferred_channel": "brave",
  "requires_confirmation": false,
  "constraints": [
    "do not override approval authority",
    "stop if required upstream artifacts are missing"
  ],
  "notes": "Golden Loop trading proof-of-life request."
}
```

## Example view-model object

```json
{
  "view_id": "view-trading-golden-001",
  "request_id": "req-trading-golden-001",
  "domain": "trading",
  "view_type": "cockpit",
  "headline": "Open trading cockpit",
  "status": "ready",
  "data": {
    "url": "http://127.0.0.1:8777/cockpit.html",
    "authority_source": "outputs/approval_decision.json",
    "workflow_source": "outputs/workflow_state.json"
  },
  "actions": [
    "open_dashboard"
  ],
  "artifacts": [
    "cockpit/cockpit.html",
    "outputs/approval_decision.json",
    "outputs/workflow_state.json"
  ]
}
```

## Example observation object

```json
{
  "observation_id": "obs-trading-golden-001",
  "request_id": "req-trading-golden-001",
  "status": "success",
  "action_type": "open_dashboard",
  "target": "brave:http://127.0.0.1:8777/cockpit.html",
  "started_at": "2026-06-17T22:10:00Z",
  "ended_at": "2026-06-17T22:10:03Z",
  "result_summary": "Brave opened the trading cockpit URL.",
  "artifacts_created": [
    "phone_companion/observations/golden_loop_latest.json"
  ],
  "fallback_used": false
}
```

## Example tired-human run

### Preflight

Confirm these exist:

- `cockpit/cockpit.html`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`
- `phone_companion/launchers/run_phone_companion_trading.sh`

### Run

Conceptual command:

```bash
bash phone_companion/launchers/run_phone_companion_trading.sh
```

### Expected visible result

- Brave opens or comes forward
- Brave attempts to show `http://127.0.0.1:8777/cockpit.html`

### Expected evidence result

- observation file written
- request and view-model evidence saved
- latest observation points to the request ID used for the run

### Time budget

If preflight artifacts already exist, this loop should take:

- **2-5 minutes** for a clean run
- **under 10 minutes** even for a tired human following a checklist

---

## Final rule

The Golden Loop is not “a mobile platform.”
It is one honest proof that Phone Companion v1 can complete:

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

for a trading dashboard open, with durable evidence left behind.
