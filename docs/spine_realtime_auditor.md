# Spine Real-Time Auditor

Purpose: run a separate diagnostic loop that watches cockpit `signal.json`
snapshots, scores forward price outcomes, and writes a shadow weight-adjustment
overlay for the execution spine.

This is **not** broker authority and does not edit Python logic.

## Run once

```bash
cd ~/SharpEdge-System
python3 cockpit/spine_realtime_auditor.py
```

## Run as a loop

```bash
cd ~/SharpEdge-System
bash cockpit/run_spine_realtime_auditor.sh
```

Useful knobs:

```bash
SPINE_AUDIT_HORIZON_SECONDS=300 \
SPINE_AUDIT_INTERVAL_SECONDS=30 \
SPINE_AUDIT_MIN_MOVE_BPS=3.0 \
SPINE_AUDIT_MIN_SAMPLES=8 \
bash cockpit/run_spine_realtime_auditor.sh
```

## Outputs

```text
outputs/spine_realtime_audit/snapshots.jsonl
outputs/spine_realtime_audit/latest.json
outputs/spine_realtime_audit/latest.txt
outputs/spine_realtime_adjustments.json
```

## Optional live consumption

By default, the cockpit does **not** consume the overlay. To allow the next
cockpit refresh to apply capped diagnostic shadow deltas:

```bash
SHARPEDGE_SPINE_REALTIME_ADJUST=1 bash cockpit/run_local_dashboard.sh
```

The overlay is capped to `±0.03` weight delta per vector and is labeled:

```text
authority = diagnostic_shadow_overlay
```

Final permission remains:

```text
approval_decision_plus_operator
```

## Why this design

The auditor can adapt weights from live evidence, but the cockpit remains safe:

- audit loop is separate
- adjustments are JSON data, not source-code mutation
- live consumption is explicit opt-in
- score spine remains diagnostic/advisory
