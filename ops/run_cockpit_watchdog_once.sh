#!/data/data/com.termux/files/usr/bin/bash
# Idempotently keep the local SharpEdge station alive across process loss/reboots.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
PORT="${COCKPIT_PORT:-8777}"
LAUNCHER="${SHARPEDGE_WATCHDOG_LAUNCHER:-$ROOT_DIR/cockpit/run_local_dashboard.sh}"
LOG_FILE="${SHARPEDGE_WATCHDOG_LOG:-$OUTPUT_DIR/cockpit_watchdog.log}"
DASHBOARD_LOG="${SHARPEDGE_WATCHDOG_DASHBOARD_LOG:-$OUTPUT_DIR/cockpit_watchdog_dashboard.log}"
PID_FILE="${SHARPEDGE_WATCHDOG_PID_FILE:-$OUTPUT_DIR/cockpit_watchdog_dashboard.pid}"
LOCK_DIR="${SHARPEDGE_WATCHDOG_LOCK_DIR:-$OUTPUT_DIR/.cockpit_watchdog.lock}"
STARTUP_WAIT_SECONDS="${SHARPEDGE_WATCHDOG_STARTUP_WAIT_SECONDS:-12}"

mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE" "$DASHBOARD_LOG"

log_line() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_FILE"
}

port_is_live() {
  python3 - "$PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

if port_is_live; then
  log_line "healthy port=$PORT action=none"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log_line "restart_skipped reason=watchdog_lock_held port=$PORT"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# Re-check after acquiring the lock; another watchdog may have won the race.
if port_is_live; then
  log_line "healthy_after_lock port=$PORT action=none"
  exit 0
fi

if [ ! -x "$LAUNCHER" ]; then
  log_line "restart_failed reason=launcher_missing launcher=$LAUNCHER"
  exit 1
fi

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock >/dev/null 2>&1 || true
fi

nohup env \
  COCKPIT_PORT="$PORT" \
  COCKPIT_OPEN_BROWSER= \
  COCKPIT_OPEN_OPERATOR_SURFACE= \
  bash "$LAUNCHER" >> "$DASHBOARD_LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
log_line "restart_started pid=$pid port=$PORT launcher=$LAUNCHER"

sleep "$STARTUP_WAIT_SECONDS"
if port_is_live; then
  log_line "restart_verified pid=$pid port=$PORT"
  exit 0
fi

log_line "restart_failed pid=$pid port=$PORT reason=port_not_live_after_wait"
exit 1
