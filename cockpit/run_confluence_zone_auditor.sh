#!/data/data/com.termux/files/usr/bin/bash
# Run the SharpEdge confluence-zone weight auditor as a background loop.
#
# It watches confluence-zone snapshots, grades RESPECTED vs BROKEN from the
# forward spot path, and writes a per-factor-kind weight-multiplier overlay to
# outputs/confluence_zone_adjustments.json.
#
# Usage:
#   bash cockpit/run_confluence_zone_auditor.sh
#   CONFLUENCE_AUDIT_HORIZON_SECONDS=900 CONFLUENCE_AUDIT_INTERVAL_SECONDS=120 \
#     bash cockpit/run_confluence_zone_auditor.sh
#
# Optional engine consumption of the overlay (opt-in, default OFF):
#   SHARPEDGE_CONFLUENCE_REALTIME_ADJUST=1 bash cockpit/run_local_dashboard.sh
#
# The overlay is diagnostic/advisory only. It never trades and never edits the
# Python weight tables.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HORIZON_SECONDS="${CONFLUENCE_AUDIT_HORIZON_SECONDS:-900}"
INTERVAL_SECONDS="${CONFLUENCE_AUDIT_INTERVAL_SECONDS:-120}"
MIN_SAMPLES="${CONFLUENCE_AUDIT_MIN_SAMPLES:-12}"

python3 cockpit/confluence_zone_auditor.py \
  --loop \
  --horizon-seconds "$HORIZON_SECONDS" \
  --interval-seconds "$INTERVAL_SECONDS" \
  --min-samples "$MIN_SAMPLES"
