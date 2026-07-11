#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCHER="$ROOT_DIR/cockpit/run_local_ace_dashboard.sh"
STATE_FILE="$OUTPUT_DIR/ace_authority_market_open_state.env"
LOG_FILE="$OUTPUT_DIR/ace_authority_market_open.log"
DASHBOARD_LOG="$OUTPUT_DIR/run_local_dashboard_autostart.log"
CHECK_INTERVAL="${ACE_OPEN_CHECK_INTERVAL_SECONDS:-20}"
TRIGGER_HHMM="${ACE_MARKET_OPEN_HHMM:-0930}"
PORT="${COCKPIT_PORT:-8777}"

mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE" "$DASHBOARD_LOG"

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock >/dev/null 2>&1 || true
fi

ny_date() {
  TZ=America/New_York date +"%F"
}

ny_weekday() {
  TZ=America/New_York date +"%u"
}

ny_hhmm() {
  TZ=America/New_York date +"%H%M"
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
  LAST_LAUNCH_DATE=""
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  printf 'LAST_LAUNCH_DATE=%s\n' "$1" > "$STATE_FILE"
}

launch_dashboard() {
  if port_is_live; then
    echo "$(date -Is) ace authority dashboard already serving on :$PORT" >> "$LOG_FILE"
    return
  fi
  nohup bash "$LAUNCHER" >> "$DASHBOARD_LOG" 2>&1 &
  echo "$!" > "$OUTPUT_DIR/ace_authority_dashboard.pid"
  echo "$(date -Is) launched ace authority dashboard pid=$! port=$PORT" >> "$LOG_FILE"
}

echo "$(date -Is) ace authority market-open daemon started (trigger=$TRIGGER_HHMM NY)" >> "$LOG_FILE"

while true; do
  load_state
  today="$(ny_date)"
  weekday="$(ny_weekday)"
  hhmm="$(ny_hhmm)"

  if [ "$weekday" -le 5 ] && [ "$hhmm" = "$TRIGGER_HHMM" ] && [ "${LAST_LAUNCH_DATE:-}" != "$today" ]; then
    launch_dashboard
    save_state "$today"
  fi

  sleep "$CHECK_INTERVAL"
done
