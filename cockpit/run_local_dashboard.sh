#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge local dashboard runner.
#
# Purpose:
#   Run the SPY cockpit as a plain localhost dashboard with no ADB, no wireless
#   debugging, no CDP, and no Android browser automation.
#
# Usage:
#   bash cockpit/run_local_dashboard.sh
#   COCKPIT_PORT=8777 COCKPIT_INTERVAL=45 bash cockpit/run_local_dashboard.sh
#   COCKPIT_OPEN_BROWSER=1 bash cockpit/run_local_dashboard.sh
#   COCKPIT_OPEN_BROWSER=1 COCKPIT_OPEN_OPERATOR_SURFACE=1 bash cockpit/run_local_dashboard.sh
#
# Then open these URLs manually in any browser on the phone, or let the script
# open them via Android intents when COCKPIT_OPEN_BROWSER=1:
#   http://127.0.0.1:8777/cockpit.html
#   http://127.0.0.1:8777/operator_surface.html
#   http://127.0.0.1:8777/runner_handoff_live.html

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COCKPIT_DIR="$ROOT_DIR/cockpit"
PORT="${COCKPIT_PORT:-8777}"
INTERVAL="${COCKPIT_INTERVAL:-45}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
OPERATOR_SURFACE_URL="http://127.0.0.1:${PORT}/operator_surface.html"
RUNNER_HANDOFF_URL="http://127.0.0.1:${PORT}/runner_handoff_live.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
SERVER_LOG="$LOGDIR/sharpedge_cockpit_server_${PORT}.log"
AUTHORITY_ENGINE="${SHARPEDGE_AUTHORITY_ENGINE:-legacy}"
OPEN_BROWSER="${COCKPIT_OPEN_BROWSER:-}"
OPEN_OPERATOR_SURFACE="${COCKPIT_OPEN_OPERATOR_SURFACE:-1}"

mkdir -p "$LOGDIR"
cd "$COCKPIT_DIR"

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

start_server() {
  if port_is_live; then
    echo "server already live on :$PORT"
    return
  fi

  python3 -m http.server "$PORT" --bind 127.0.0.1 >"$SERVER_LOG" 2>&1 &
  echo "server started on :$PORT (pid $!, log $SERVER_LOG)"
}

build_once() {
  local failed=0
  python3 make_cockpit.py || failed=1
  bash ./refresh_operator_surface_inputs.sh || failed=1
  python3 make_operator_surface.py || failed=1
  return "$failed"
}

open_url() {
  local target_url="$1"
  local label="$2"
  if command -v am >/dev/null 2>&1; then
    am start -a android.intent.action.VIEW -p com.brave.browser -d "$target_url" \
      >/dev/null 2>&1 && echo "opened Brave -> $label: $target_url" \
      || echo "(could not auto-open Brave for $label; browse to $target_url yourself)"
  elif command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$target_url" && echo "opened browser -> $label: $target_url"
  else
    echo "open $label manually: $target_url"
  fi
}

maybe_open_surfaces() {
  if [ -z "$OPEN_BROWSER" ]; then
    return
  fi
  open_url "$URL" "cockpit"
  if [ -n "$OPEN_OPERATOR_SURFACE" ]; then
    open_url "$OPERATOR_SURFACE_URL" "operator surface"
  fi
}

start_server

echo "building cockpit once..."
if ! build_once; then
  echo "first build failed; keeping server up and retrying in loop"
fi

maybe_open_surfaces

cat <<EOF

SharpEdge local dashboard is running without ADB/wireless/CDP.
Authority engine: ${AUTHORITY_ENGINE}
First-class live surfaces:

  cockpit:             $URL
  operator surface:    $OPERATOR_SURFACE_URL
  runner handoff live: $RUNNER_HANDOFF_URL

Regenerating every ${INTERVAL}s. Press Ctrl+C to stop the refresh loop.
EOF

while true; do
  sleep "$INTERVAL"
  build_once || echo "refresh failed; retrying next loop"
done
