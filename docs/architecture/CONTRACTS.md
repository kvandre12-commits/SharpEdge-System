# SharpEdge Contracts

Stable schemas are the glue between repos. Prefer one canonical field shape over
near-duplicates.

## `sharpedge.signal.v1`

Producer:

```text
SharpEdge-System/cockpit/make_cockpit.py
```

Primary artifact:

```text
SharpEdge-System/outputs/signal.json
```

Consumers:

```text
phone_companion/build_golden_loop_view_model.py
SharpEdge-Android native viewer
```

Important fields:

```json
{
  "schema": "sharpedge.signal.v1",
  "symbol": "SPY",
  "spot": 746.05,
  "trade_permission": {
    "schema": "sharpedge.trade_permission.v1",
    "trade_gate": "CAUTION",
    "trade_permission_score": 58,
    "bias": "NEUTRAL",
    "bias_strength": 0.211,
    "supporting_reasons": [],
    "warning_reasons": [],
    "scores": {}
  }
}
```

## `sharpedge.trade_permission.v1`

Producer:

```text
SharpEdge-System/cockpit/trade_permission.py
```

Canonical field names:

- `trade_gate`
- `trade_permission_score`
- `bias`
- `bias_strength`
- `supporting_reasons`
- `warning_reasons`
- `scores`

Do not invent duplicate names like `signal_gate`, `permission_gate`, or
`score_gate`. Use the canonical names above.

## Phone Companion request

Contract file:

```text
phone_companion/contracts/phone_companion_request_v1.json
```

Golden Loop request:

```text
phone_companion/requests/golden_loop_request.json
```

Required fields:

- `request_id`
- `intent_type`
- `domain`
- `artifact_inputs`

## Phone Companion view-model

Contract file:

```text
phone_companion/contracts/phone_companion_view_model_v1.json
```

Golden Loop view-model:

```text
phone_companion/views/trading/golden_loop_view_model.json
```

The trading view-model includes:

```json
{
  "data": {
    "url": "http://127.0.0.1:8777/cockpit.html",
    "artifact_inputs": [],
    "preferred_channel": "brave",
    "signal_summary": {
      "trade_permission": {
        "gate": "CAUTION",
        "score": 58,
        "bias": "NEUTRAL"
      }
    }
  }
}
```

Note: `signal_summary.trade_permission.gate` is a compact mobile snapshot. The
source-of-truth field remains `signal.json["trade_permission"]["trade_gate"]`.

## Native viewer observation

Contract file:

```text
phone_companion/contracts/sharpedge_viewer_observation_v1.json
```

Purpose: replace browser DevTools page inspection with owned native render proof.

Expected shape:

```json
{
  "observation_type": "sharpedge_viewer_rendered",
  "request_id": "req-trading-golden-001",
  "view_id": "view-trading-golden-001",
  "gate": "CAUTION",
  "score": 58,
  "bias": "NEUTRAL",
  "render_status": "rendered"
}
```
