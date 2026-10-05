#!/data/data/com.termux/files/usr/bin/bash
# Run the SharpEdge real-time spine auditor as a separate diagnostic loop.
#
# Usage:
#   bash cockpit/run_spine_realtime_auditor.sh
#   SPINE_AUDIT_HORIZON_SECONDS=300 SPINE_AUDIT_INTERVAL_SECONDS=30 bash cockpit/run_spine_realtime_auditor.sh
#
# Optional cockpit consumption of its shadow weight overlay:
#   SHARPEDGE_SPINE_REALTIME_ADJUST=1 bash cockpit/run_local_dashboard.sh
#
# The overlay is diagnostic/advisory only. It does not grant broker authority.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HORIZON_SECONDS="${SPINE_AUDIT_HORIZON_SECONDS:-300}"
INTERVAL_SECONDS="${SPINE_AUDIT_INTERVAL_SECONDS:-30}"
MIN_MOVE_BPS="${SPINE_AUDIT_MIN_MOVE_BPS:-3.0}"
MIN_SAMPLES="${SPINE_AUDIT_MIN_SAMPLES:-8}"

python3 cockpit/spine_realtime_auditor.py \
  --loop \
  --horizon-seconds "$HORIZON_SECONDS" \
  --interval-seconds "$INTERVAL_SECONDS" \
  --min-move-bps "$MIN_MOVE_BPS" \
  --min-samples "$MIN_SAMPLES"
