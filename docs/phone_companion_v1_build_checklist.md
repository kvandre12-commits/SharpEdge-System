# Phone Companion v1 Build Checklist

## Purpose

This checklist defines the **smallest operational footprint** required to execute
the **Golden Loop** successfully.

Golden Loop mission:

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

This checklist contains **only** the files and folders required to make that
loop work for the trading path.

Scope rules:
- trading only
- one launcher only
- one dashboard-open action only
- one observation path only
- no backend dependency
- no frontend dependency
- no section-gap runtime dependency

---

## Minimum system boundary

The Golden Loop needs exactly four kinds of things:

1. **upstream trading truth**
2. **Phone Companion contract definitions**
3. **one trading launcher**
4. **runtime evidence storage**

Anything outside that boundary is not required for the first successful loop.

---

# Phase 1: Files and folders that must exist before any code is written

These items define the shell boundary and the contract seam.
Without them, future pilots will improvise structure and ruin the repo in new and exciting ways.

## Folders

| Path | Purpose | Owner component | Required or optional | Example contents |
|---|---|---|---|---|
| `phone_companion/` | Root home for the Phone Companion shell | Phone Companion | Required | `contracts/`, `configs/`, `launchers/`, `observations/` |
| `phone_companion/contracts/` | Holds the request/view-model/observation contract definitions | Contract layer | Required | `phone_companion_request_v1.json`, `phone_companion_view_model_v1.json`, `phone_companion_observation_v1.json` |
| `phone_companion/configs/` | Holds delivery routing defaults for the shell | Phone Companion | Required | `phone_companion_routes.example.json` |
| `phone_companion/requests/` | Holds the concrete request artifact used by the Golden Loop | Contract layer | Required | `golden_loop_request.json` |
| `phone_companion/views/` | Holds mobile-facing view artifacts | Code Puppy layer | Required | `trading/` |
| `phone_companion/views/trading/` | Holds the concrete trading view-model artifact used by the Golden Loop | Code Puppy layer | Required | `golden_loop_view_model.json` |
| `phone_companion/launchers/` | Holds the one trading launcher entrypoint | DroidPuppy handoff layer | Required | `run_phone_companion_trading.sh` |
| `phone_companion/observations/` | Holds latest observation and runtime evidence | Observation layer | Required | later `golden_loop_latest.json` |
| `cockpit/` | Upstream trading artifact home | Trading domain | Required | `cockpit.html` |
| `outputs/` | Upstream decision/authority artifact home | Upstream operator stack | Required | `approval_decision.json`, `workflow_state.json` |

## Files

| Path | Purpose | Owner component | Required or optional | Example contents |
|---|---|---|---|---|
| `phone_companion/contracts/phone_companion_request_v1.json` | Defines the request contract shape for the Golden Loop | Contract layer | Required | `{"required_fields": ["request_id", "intent_type", "domain", "artifact_inputs"]}` |
| `phone_companion/contracts/phone_companion_view_model_v1.json` | Defines the trading view-model contract shape | Contract layer | Required | `{"required_fields": ["view_id", "request_id", "domain", "view_type", "headline", "status", "data"]}` |
| `phone_companion/contracts/phone_companion_observation_v1.json` | Defines the observation contract shape | Observation layer | Required | `{"required_fields": ["observation_id", "request_id", "status", "action_type", "target", "started_at", "ended_at"]}` |
| `phone_companion/configs/phone_companion_routes.example.json` | Declares the default Golden Loop routing target | Phone Companion | Required | `{"domains": {"trading": {"default_view": "cockpit", "default_browser": "brave"}}}` |
| `phone_companion/launchers/run_phone_companion_trading.sh` | The one launcher entrypoint for the Golden Loop | DroidPuppy handoff layer | Required | shell script stub or implementation that verifies artifacts and opens Brave |
| `cockpit/cockpit.html` | The dashboard artifact the Golden Loop opens | Trading domain | Required | rendered trading cockpit HTML |
| `outputs/approval_decision.json` | Upstream authority artifact; must never be overridden | Upstream operator stack | Required | `{"decision": "hold", "trade_allowed": false, "broker_order_allowed": false}` |
| `outputs/workflow_state.json` | Upstream current-state artifact for display/preflight context | Upstream operator stack | Required | `{"state": {"readiness": "blocked"}, "market_context": {...}}` |
| `outputs/operator_brief.json` | Optional richer trading summary input for future display context | Upstream operator stack | Optional | operator-facing summary JSON |
| `outputs/morning_open_dashboard.json` | Optional richer open-check context | Upstream operator stack | Optional | morning dashboard JSON |

---

# Phase 2: Files and folders that must exist before the first successful run

These items convert the architecture from static structure into an executable loop.

## Folders

| Path | Purpose | Owner component | Required or optional | Example contents |
|---|---|---|---|---|
| `phone_companion/observations/history/` | Stores timestamped evidence from successful or failed runs | Observation layer | Required | `2026-06-17T22-10-03Z__golden_loop_observation.json` |

## Files

| Path | Purpose | Owner component | Required or optional | Example contents |
|---|---|---|---|---|
| `phone_companion/launchers/run_phone_companion_trading.sh` | Must be implemented enough to verify artifacts, invoke DroidPuppy, and write observations | DroidPuppy handoff layer | Required | `am start -a android.intent.action.VIEW -d http://127.0.0.1:8777/cockpit.html -p com.brave.browser` |
| `phone_companion/requests/golden_loop_request.json` | Concrete request artifact used for the first run | Contract layer / Code Puppy input | Required | `{"request_id": "req-trading-golden-001", "intent_type": "open_trading_dashboard", "domain": "trading", "artifact_inputs": ["cockpit/cockpit.html", "outputs/approval_decision.json", "outputs/workflow_state.json"]}` |
| `phone_companion/views/trading/golden_loop_view_model.json` | Concrete mobile-facing trading view-model used for the first run | Code Puppy layer | Required | `{"view_id": "view-trading-golden-001", "request_id": "req-trading-golden-001", "domain": "trading", "view_type": "cockpit", "headline": "Open trading cockpit", "status": "ready", "data": {"url": "http://127.0.0.1:8777/cockpit.html"}}` |
| `cockpit/cockpit.html` | Must already exist as the open target for the phone action | Trading domain | Required | HTML page with cockpit content |
| `outputs/approval_decision.json` | Must exist so the shell can read authority without inventing it | Upstream operator stack | Required | `{"decision": "hold", "trade_allowed": false}` |
| `outputs/workflow_state.json` | Must exist so the shell can read current state without recomputing it | Upstream operator stack | Required | `{"state": {"readiness": "blocked"}}` |
| `phone_companion/configs/phone_companion_routes.example.json` | Must map trading to Brave and the trading launcher path | Phone Companion | Required | `{"domains": {"trading": {"default_browser": "brave", "default_launcher": "launchers/run_phone_companion_trading.sh"}}}` |

## Notes on minimalism

The following currently exist in the repo but are **not required** for the first successful Golden Loop run:

- `phone_companion/README.md`
- `phone_companion/views/trading/README.md`
- `phone_companion/observations/README.md`
- `phone_companion/launchers/README.md`
- `phone_companion/configs/README.md`
- `command_deck.html`
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`

Useful? Sure.
Required for the smallest proof? Nope.

---

# Phase 3: Files created by a successful run

These are the proof-of-life artifacts.
If these are not created, the Golden Loop did not finish.

## Files

| Path | Purpose | Owner component | Required or optional | Example contents |
|---|---|---|---|---|
| `phone_companion/observations/golden_loop_latest.json` | Latest observation summary for the most recent run | Observation layer | Required | `{"observation_id": "obs-trading-golden-001", "request_id": "req-trading-golden-001", "status": "success", "target": "brave:http://127.0.0.1:8777/cockpit.html"}` |
| `phone_companion/observations/history/<timestamp>__golden_loop_observation.json` | Durable timestamped observation record for later review | Observation layer | Required | `{"started_at": "2026-06-17T22:10:00Z", "ended_at": "2026-06-17T22:10:03Z", "result_summary": "Brave opened the trading cockpit URL."}` |
| `phone_companion/observations/history/<timestamp>__golden_loop_request.json` | Durable copy of the exact request used for the run | Contract layer / evidence | Required | `{"request_id": "req-trading-golden-001", "intent_type": "open_trading_dashboard", "domain": "trading"}` |
| `phone_companion/observations/history/<timestamp>__golden_loop_view_model.json` | Durable copy of the exact view-model used for the run | Code Puppy layer / evidence | Required | `{"view_id": "view-trading-golden-001", "request_id": "req-trading-golden-001", "data": {"url": "http://127.0.0.1:8777/cockpit.html"}}` |
| `phone_companion/observations/history/<timestamp>__golden_loop_result.txt` | Plain-text human-readable run summary | Launcher / evidence | Optional | `SUCCESS request=req-trading-golden-001 target=brave:http://127.0.0.1:8777/cockpit.html` |

---

# Smallest operational footprint summary

If you strip everything down to bare survival minimum, the Golden Loop only needs:

## Required folders

- `phone_companion/`
- `phone_companion/contracts/`
- `phone_companion/configs/`
- `phone_companion/requests/`
- `phone_companion/views/`
- `phone_companion/views/trading/`
- `phone_companion/launchers/`
- `phone_companion/observations/`
- `phone_companion/observations/history/`
- `cockpit/`
- `outputs/`

## Required pre-existing files

- `phone_companion/contracts/phone_companion_request_v1.json`
- `phone_companion/contracts/phone_companion_view_model_v1.json`
- `phone_companion/contracts/phone_companion_observation_v1.json`
- `phone_companion/configs/phone_companion_routes.example.json`
- `phone_companion/launchers/run_phone_companion_trading.sh`
- `phone_companion/requests/golden_loop_request.json`
- `phone_companion/views/trading/golden_loop_view_model.json`
- `cockpit/cockpit.html`
- `outputs/approval_decision.json`
- `outputs/workflow_state.json`

## Required files created by success

- `phone_companion/observations/golden_loop_latest.json`
- `phone_companion/observations/history/<timestamp>__golden_loop_observation.json`
- `phone_companion/observations/history/<timestamp>__golden_loop_request.json`
- `phone_companion/observations/history/<timestamp>__golden_loop_view_model.json`

That is the smallest honest system.
Anything less is missing part of the loop.
Anything much more is probably architecture vanity.
