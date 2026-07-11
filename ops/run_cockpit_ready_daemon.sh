#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCHER="${SHARPEDGE_READY_LAUNCHER:-$ROOT_DIR/cockpit/run_local_dashboard.sh}"
OPEN_SCRIPT="${SHARPEDGE_READY_OPEN_SCRIPT:-$ROOT_DIR/cockpit/open_cockpit.sh}"
STATE_FILE="${SHARPEDGE_READY_STATE_FILE:-$OUTPUT_DIR/cockpit_ready_state.env}"
LOG_FILE="${SHARPEDGE_READY_LOG_FILE:-$OUTPUT_DIR/cockpit_ready_daemon.log}"
DASHBOARD_LOG="${SHARPEDGE_READY_DASHBOARD_LOG:-$OUTPUT_DIR/cockpit_ready_dashboard.log}"
CHECK_INTERVAL="${SHARPEDGE_READY_CHECK_INTERVAL_SECONDS:-20}"
TRIGGER_HHMM="${SHARPEDGE_READY_HHMM:-0900}"
TRIGGER_TZ="${SHARPEDGE_READY_TZ:-America/New_York}"
PORT="${COCKPIT_PORT:-8777}"
OPEN_OPERATOR_SURFACE="${SHARPEDGE_READY_OPEN_OPERATOR_SURFACE:-}"

mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE" "$DASHBOARD_LOG"

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock >/dev/null 2>&1 || true
fi

wall_date() {
  TZ="$TRIGGER_TZ" date +"%F"
}

wall_weekday() {
  TZ="$TRIGGER_TZ" date +"%u"
}

wall_hhmm() {
  TZ="$TRIGGER_TZ" date +"%H%M"
}

port_is_live() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

load_state() {
  LAST_READY_DATE=""
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  printf 'LAST_READY_DATE=%s\n' "$1" > "$STATE_FILE"
}

log_line() {
  echo "$(date -Is) $*" >> "$LOG_FILE"
}

open_existing_cockpit() {
  if [ ! -x "$OPEN_SCRIPT" ]; then
    log_line "open script missing or not executable: $OPEN_SCRIPT"
    return
  fi
  COCKPIT_PORT="$PORT" bash "$OPEN_SCRIPT" >> "$DASHBOARD_LOG" 2>&1 || true
  log_line "opened existing cockpit in Brave on :$PORT"
}

launch_dashboard() {
  nohup env \
    COCKPIT_PORT="$PORT" \
    COCKPIT_OPEN_BROWSER=1 \
    COCKPIT_OPEN_OPERATOR_SURFACE="$OPEN_OPERATOR_SURFACE" \
    bash "$LAUNCHER" >> "$DASHBOARD_LOG" 2>&1 &
  echo "$!" > "$OUTPUT_DIR/cockpit_ready_dashboard.pid"
  log_line "launched dashboard pid=$! port=$PORT trigger=$TRIGGER_HHMM tz=$TRIGGER_TZ"
}

ready_cockpit() {
  if port_is_live; then
    log_line "dashboard already live on :$PORT; reopening cockpit"
    open_existing_cockpit
    return
  fi
  launch_dashboard
}

log_line "cockpit ready daemon started (trigger=$TRIGGER_HHMM tz=$TRIGGER_TZ weekdays_only=yes)"

while true; do
  load_state
  today="$(wall_date)"
  weekday="$(wall_weekday)"
  hhmm="$(wall_hhmm)"

  if [ "$weekday" -le 5 ] && [ "$hhmm" = "$TRIGGER_HHMM" ] && [ "${LAST_READY_DATE:-}" != "$today" ]; then
    ready_cockpit
    save_state "$today"
  fi

  sleep "$CHECK_INTERVAL"
done
