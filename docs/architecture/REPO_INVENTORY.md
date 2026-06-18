# SharpEdge Repo Inventory

## Projects

| Repo | Status | Purpose | Owns | Does not own |
|---|---:|---|---|---|
| `SharpEdge-System` | active | Signal generation and trade gate analytics | `outputs/signal.json`, cockpit pipeline, trade permission scoring, operator artifacts | Android UI, browser rendering, broker execution |
| `phone_companion/` inside `SharpEdge-System` | active | Contract packaging and Golden Loop evidence | request/view/prelaunch/observation artifacts, signal summaries | trading logic, Android UI, broker actions |
| `SharpEdge-Android` | active | Native Android viewer | Kotlin/Compose UI, Trade Gate display, signal contract rendering | signal generation, approval authority, broker actions |
| `DroidPuppy` | experimental | Android execution/observation utilities | Android intents, screenshots, UI dumps, browser handoff helpers | trading logic, broker actions, core viewer rendering |
| `SharpEdge-Robinhood-Bridge` | experimental | Broker capability routing | approval-gated broker plans/delegations | signal generation, Android rendering, autonomous live orders |
| `code_puppy` | active platform | Agent engine and plugin framework | agent/tool/plugin runtime | SharpEdge domain truth |

## Source vs artifacts

Source code and docs are stable inputs.

Generated artifacts are runtime evidence and may change every run:

```text
outputs/signal.json
outputs/workflow_state.json
outputs/approval_decision.json
phone_companion/launchers/prelaunch_trace.json
phone_companion/launchers/launch_attempt.json
phone_companion/launchers/launch_result.json
phone_companion/observations/golden_loop_latest.json
phone_companion/views/trading/golden_loop_view_model.json
```

Agents should inspect artifacts when debugging runtime state, but should avoid
formatting or refactoring them as if they were hand-authored source.

## Where to work

- Trade gate math: `SharpEdge-System/cockpit/trade_permission.py`
- Signal publication: `SharpEdge-System/cockpit/make_cockpit.py`
- Contract packaging: `SharpEdge-System/phone_companion/`
- Native viewer UI: `SharpEdge-Android/app/src/main/java/com/sharpedge/cockpit/`
- Android render contract docs: `SharpEdge-Android/docs/VIEWER_CONTRACT.md`
