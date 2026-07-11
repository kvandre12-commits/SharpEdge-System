# SharpEdge Viewing Layer

SharpEdge needs a formal answer to a simple question:

```text
What should the operator see, in what order, with what proof,
without changing what the system means?
```

That answer is the **viewing layer**.

## Position in doctrine

The viewing layer is downstream of interpretation and downstream of authority.
It is not part of market reasoning.

```text
Compression
  -> Regime
  -> Liquidity event
  -> Acceptance
  -> Pressure
  -> Setup
  -> Score
  -> Bucket
  -> Phase
  -> Authority
  -> Viewing
```

Interpretation decides what the market means.
Authority decides whether action is allowed.
Viewing decides how the completed interpretation is shown to a human.

## Core doctrine

```text
View may derive presentation.
View may not derive meaning.
```

The viewing layer is allowed to compress information for cognition.
It is not allowed to invent, revise, or smuggle new market logic.

## What the viewing layer owns

The viewing layer owns three jobs:

1. **Projection**
   - package completed protocols into viewer-facing packets
   - preserve provenance and stable field names
2. **Rendering**
   - explain upstream conclusions in HTML, JSON view-models, or native UI
   - optimize order, grouping, and readability for humans
3. **Proof**
   - record what was shown, where, and whether rendering succeeded

## What the viewing layer does not own

The viewing layer does **not** own:

- market interpretation
- market-state inference
- setup detection
- readiness logic
- permission logic
- authority changes
- bucket selection
- phase selection
- score recomputation

If a change alters what the market means, it is not a viewing-layer change.
Move it upstream.

## Allowed derivations

Allowed:

- format
- summarize
- truncate
- group
- reorder for readability
- color-code
- choose chart emphasis
- attach provenance/evidence labels
- emit render observations
- create compact snapshots that clearly point back to a source-of-truth contract

Not allowed:

- infer market state
- reinterpret evidence
- change authority
- change readiness
- change permission
- change score values
- revise bucket identity
- revise phase identity
- replace warnings with friendlier language that changes risk meaning
- mutate protocols because the view "looks better"

## Boundary rule

```text
The view may explain a decision.
The view may not become the decision-maker.
```

That means:

- a compact mobile summary is okay
- a pretty execution-flow section is okay
- a top-3 reasons panel is okay
- a native render proof is good
- a viewer-specific fallback score is not okay
- a viewer changing `WAIT` into `CONTEXT_MATCH` is not okay
- a viewer inventing a better bucket label is not okay

## Viewing sublayers

### 1. View packet assembly

Purpose: convert completed protocols into render-ready packets without changing
meaning.

Typical examples:

- `cockpit/execution_card_builder.py`
- `phone_companion/build_golden_loop_view_model.py`
- `phone_companion/export_signal_to_android_viewer.py`

Rule:

```text
Packet assembly may derive.
Packet assembly may not infer.
```

### 2. Render explanation

Purpose: present completed meaning to the operator.

Typical examples:

- `cockpit/execution_flow_view.py`
- `cockpit/live_read_view.py`
- `cockpit/make_cockpit.py`
- `cockpit/make_operator_surface.py`
- SharpEdge-Android native viewer

Rule:

```text
Rendering may compress for cognition.
Rendering may not rewrite doctrine.
```

### 3. Render proof

Purpose: prove what was shown and whether it rendered correctly.

Typical examples:

- `phone_companion/contracts/sharpedge_viewer_observation_v1.json`
- `phone_companion/emit_golden_loop_prelaunch_trace.py`
- Android/native viewer observations

Rule:

```text
Proof may observe rendering.
Proof may not reinterpret the rendered contract.
```

## Agile dashboard doctrine

Detailed card ordering lives in:

```text
docs/architecture/AGILE_DASHBOARD_STACK.md
```


An **agile dashboard** is allowed inside the viewing layer.

Here, agile means:

```text
The surface may adapt its composition to operator context.
The surface may not adapt market meaning.
```

Good agility:

- show different card priority at morning open vs live session vs review
- expand blocker / freshness / authority cards when readiness is weak
- collapse secondary context when a live liquidity event is active
- switch between compact and deep evidence layouts
- choose the operator's next most useful question first

Bad agility:

- invent a stronger bucket because the layout wants a headline
- hide authority warnings to keep the dashboard clean
- recalculate permission because the mobile summary needs conviction
- reinterpret pressure/setup/acceptance to make the view feel smarter

### Agile dashboard modes

A clean starting set is:

1. **Open mode**
   - freshness
   - regime / battlefield
   - key levels
   - approval blockers
   - opening checklist
2. **Live mode**
   - liquidity event
   - acceptance
   - pressure state
   - active setup
   - bucket / phase / authority
3. **Review mode**
   - what changed
   - what fired
   - what was blocked
   - receipt / proof / journal context
4. **Handoff mode**
   - compact source-of-truth snapshot
   - transport/view-model integrity
   - render proof / observation status

These are viewing compositions, not new reasoning systems.

### Agile dashboard assembly rule

```text
Dashboard chooses which completed cards to foreground.
Dashboard does not create new doctrine to fill empty space.
```

That means an agile dashboard should be assembled from completed upstream
protocols such as:

- signal / trade permission
- setup state
- market day bucket
- bucket-conditioned spine
- phase annotations
- workflow state
- approval decision
- render proof

If a mode needs a new concept, that concept belongs upstream first.

## Canonical information flow

```text
outputs/signal.json
  -> trade_permission / market-day / setup / authority protocols
  -> view packet assembly
  -> cockpit HTML / operator HTML / phone companion view-model / native viewer
  -> render observation / proof artifact
```

This flow must stay monotonic.
Meaning flows one way.
The view does not feed interpretation back into the engine.

## Source-of-truth rule

Compact viewer fields are allowed, but they must remain explicitly downstream of
canonical contracts.

Example:

- canonical source: `signal.json["trade_permission"]["trade_gate"]`
- compact snapshot: `signal_summary.trade_permission.gate`

The snapshot is a transport/view convenience.
It is not a competing authority source.

## Ownership boundaries

| System | Owns |
|---|---|
| `SharpEdge-System` | signal generation, packet assembly, web/operator rendering |
| `phone_companion` | request/view-model transport, prelaunch trace, handoff proof |
| `SharpEdge-Android` | native rendering of exported/view-model contracts |
| `SharpEdge-Robinhood-Bridge` | execution routing and approval-gated broker planning |

## File map in this repo

| File | Viewing-layer role |
|---|---|
| `cockpit/execution_flow_view.py` | presentation-only explanation blocks for bucket/spine output |
| `cockpit/live_read_view.py` | cockpit/operator-facing cognitive compression and HTML composition |
| `cockpit/execution_card_builder.py` | packet assembly from completed protocols |
| `phone_companion/build_golden_loop_view_model.py` | cross-system trading view-model projection |
| `phone_companion/export_signal_to_android_viewer.py` | Android viewer bundle/export bridge |
| `phone_companion/emit_golden_loop_prelaunch_trace.py` | pre-render proof of intended view payload |

## Review checklist

Before changing viewer code, ask:

1. Am I changing how something is shown, or what it means?
2. Could this make two viewers disagree about gate, bias, bucket, phase, or authority?
3. Am I introducing new reasoning because the view feels incomplete?
4. Is there still a clear source-of-truth contract behind every displayed claim?
5. Does the proof layer observe rendering, rather than participate in reasoning?

If the answer to #1 or #3 is "what it means," the change belongs upstream.

## Viewing-layer question

If the market stack ends with:

```text
Authority -> Given everything above, do we act?
```

Then the viewing layer asks:

```text
Viewing -> What does the operator need to see, in what shape,
and with what proof, without mutating authority?
```
