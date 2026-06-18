# SharpEdge Ownership Map

This is the shortest boundary map. If an agent is confused, read this first.

## Ownership table

| Layer | Repo/path | Owns | Does not own |
|---|---|---|---|
| Signal generation | `SharpEdge-System` | market/cockpit analytics, `outputs/signal.json`, Trade Gate scoring | Android UI, broker execution |
| Contract packaging | `SharpEdge-System/phone_companion` | request/view/prelaunch/observation artifacts, signal summaries, Golden Loop evidence | trading math, Android rendering, broker actions |
| Native viewer | `SharpEdge-Android` | Kotlin/Compose UI, Trade Gate display, `sharpedge.signal.v1` rendering | signal generation, approval authority, broker actions |
| Android utilities | `code_puppy_backup_20260617/DroidPuppy` | Android intents, screenshots, UI dump/input, CDP/ADB helpers, optional execution tooling | trading decisions, native viewer truth, broker logic |
| Agent runtime | `code_puppy` | Code Puppy core runtime, tools, plugins, callbacks | SharpEdge domain logic |
| Broker bridge | `SharpEdge-Robinhood-Bridge` | Robinhood command classification, support tiers, approval-gated plans | signal generation, Android UI, autonomous live orders |

## Canonical contract movement

```text
SharpEdge-System outputs/signal.json
  → Phone Companion signal_summary / view-model
  → SharpEdge-Android native viewer
  → future viewer-render observation
```

## Canonical field names

Do not invent synonyms.

```json
{
  "trade_permission": {
    "trade_gate": "CAUTION",
    "trade_permission_score": 67,
    "bias": "NEUTRAL"
  }
}
```

Compact summaries may use:

```json
{
  "signal_summary": {
    "trade_permission": {
      "gate": "CAUTION",
      "score": 67,
      "bias": "NEUTRAL"
    }
  }
}
```

But the source-of-truth fields remain in `signal.json["trade_permission"]`.

## Agent rule

If the requested change is about:

- score/math/read quality → work in `SharpEdge-System`
- moving a contract between layers → work in `phone_companion`
- Android UI appearance → work in `SharpEdge-Android`
- tapping/opening/capturing Android screens → work in DroidPuppy tooling
- Robinhood command support → work in `SharpEdge-Robinhood-Bridge`
- model/tool/plugin runtime → work in `code_puppy`

Do not cross layers unless the task explicitly asks for an integration change.
