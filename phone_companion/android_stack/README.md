# SharpEdge Android Stack

This folder defines the no-WiFi, no-CDP viewer path for Phone Companion.

## Decision

SharpEdge owns the viewer.

The primary mobile rendering target is the native Android app in:

```text
/data/data/com.termux/files/home/SharpEdge-Android
```

Brave/CDP remains useful for debugging, but it is not the core observation or
viewer dependency.

## Canonical flow

```text
Intent
→ phone_companion request contract
→ trading view-model
→ signal_summary / trade_permission snapshot
→ SharpEdge Android viewer
→ viewer observation
```

## Responsibilities

### SharpEdge-System

Owns:

- `outputs/signal.json`
- `signal.json["trade_permission"]`
- `phone_companion/requests/golden_loop_request.json`
- `phone_companion/views/trading/golden_loop_view_model.json`
- `phone_companion/launchers/prelaunch_trace.json`
- observation artifacts

Does not own:

- Android UI rendering implementation
- native activity lifecycle
- Play Store packaging

### SharpEdge-Android

Owns:

- native Kotlin/Compose viewer
- parsing `sharpedge.signal.v1`
- rendering the Trade Gate
- rendering support/warning reasons
- future viewer-side observation events

Does not own:

- trade authority
- broker actions
- signal generation
- approval decisions

## Current proof

Golden Loop already proves:

```text
request_id preserved from request → view-model → prelaunch → launch result
```

The view-model and prelaunch trace now also preserve:

```text
signal_summary.trade_permission.gate
signal_summary.trade_permission.score
signal_summary.trade_permission.bias
supporting_reasons
warning_reasons
```

## Next viewer observation contract

The native app should eventually emit or expose:

```json
{
  "observation_type": "sharpedge_viewer_rendered",
  "request_id": "req-trading-golden-001",
  "view_id": "view-trading-golden-001",
  "signal_ts": "...",
  "gate": "CAUTION",
  "score": 58,
  "bias": "NEUTRAL",
  "render_status": "rendered"
}
```

That observation is the replacement for browser DevTools page inspection.
