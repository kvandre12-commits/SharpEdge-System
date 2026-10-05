#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge dedicated live runner-handoff feature page.
#
# Purpose:
#   Serve the repo root on its own localhost port so the handoff page lives at:
#     http://127.0.0.1:8765/cockpit/runner_handoff_live.html
#
# Usage:
#   bash cockpit/run_runner_handoff_feature.sh
#   HANDOFF_PORT=8766 HANDOFF_REFRESH_SECONDS=30 bash cockpit/run_runner_handoff_feature.sh
#   HANDOFF_NO_BROWSER=1 bash cockpit/run_runner_handoff_feature.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${HANDOFF_PORT:-8765}"
REFRESH_SECONDS="${HANDOFF_REFRESH_SECONDS:-45}"
URL="http://127.0.0.1:${PORT}/cockpit/runner_handoff_live.html"
LOG_DIR="$REPO_ROOT/outputs"
SERVER_LOG="$LOG_DIR/cockpit_root_server_${PORT}.log"
SERVER_PID="$LOG_DIR/cockpit_root_server_${PORT}.pid"
LOOP_LOG="$LOG_DIR/sharpedge_make_cockpit_handoff.out"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

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

ensure_server() {
  if port_is_live; then
    echo "root server already live on :$PORT"
    return
  fi
  nohup python3 -m http.server "$PORT" --bind 127.0.0.1 \
    >"$SERVER_LOG" 2>&1 &
  echo "$!" >"$SERVER_PID"
  echo "root server started on :$PORT (pid $(cat "$SERVER_PID"), log $SERVER_LOG)"
  sleep 1
}

open_browser() {
  if [ -n "${HANDOFF_NO_BROWSER:-}" ]; then
    return
  fi
  if command -v am >/dev/null 2>&1; then
    am start -a android.intent.action.VIEW -p com.brave.browser -d "$URL" \
      >/dev/null 2>&1 && echo "opened Brave -> $URL" \
      || echo "(could not auto-open Brave; browse to $URL yourself)"
  elif command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$URL" && echo "opened browser -> $URL"
  else
    echo "open this in your browser: $URL"
  fi
}

build_once() {
  python3 cockpit/make_cockpit.py >"$LOOP_LOG" 2>&1
}

ensure_server
echo "building runner handoff feature once..."
if ! build_once; then
  echo "first build failed; keeping root server up and retrying in loop"
fi
open_browser

cat <<EOF

SharpEdge runner handoff feature is live.

  feature page: $URL
  server log:    $SERVER_LOG
  build log:     $LOOP_LOG

Regenerating every ${REFRESH_SECONDS}s. Press Ctrl+C to stop the refresh loop.
EOF

while true; do
  sleep "$REFRESH_SECONDS"
  build_once || echo "handoff refresh failed; retrying next loop"
done
