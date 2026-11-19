#!/data/data/com.termux/files/usr/bin/bash
# Install the reboot-persistent Android JobScheduler watchdog for SharpEdge.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WATCHDOG="$ROOT_DIR/ops/run_cockpit_watchdog_once.sh"
JOB_ID="${SHARPEDGE_WATCHDOG_JOB_ID:-8777}"
PERIOD_MS="${SHARPEDGE_WATCHDOG_PERIOD_MS:-900000}"
NETWORK="${SHARPEDGE_WATCHDOG_NETWORK:-any}"

if ! command -v termux-job-scheduler >/dev/null 2>&1; then
  echo "termux-job-scheduler is unavailable; install/enable Termux:API." >&2
  exit 1
fi
if [ ! -x "$WATCHDOG" ]; then
  echo "watchdog is missing or not executable: $WATCHDOG" >&2
  exit 1
fi

termux-job-scheduler \
  --script "$WATCHDOG" \
  --job-id "$JOB_ID" \
  --period-ms "$PERIOD_MS" \
  --network "$NETWORK" \
  --battery-not-low false \
  --storage-not-low false \
  --charging false \
  --persisted true

echo "installed persisted SharpEdge watchdog job id=$JOB_ID period_ms=$PERIOD_MS network=$NETWORK"
echo "watchdog=$WATCHDOG"
termux-job-scheduler --pending
