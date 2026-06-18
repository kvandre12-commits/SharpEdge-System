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
#
# Then open this URL manually in any browser on the phone:
#   http://127.0.0.1:8777/cockpit.html

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COCKPIT_DIR="$ROOT_DIR/cockpit"
PORT="${COCKPIT_PORT:-8777}"
INTERVAL="${COCKPIT_INTERVAL:-45}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
SERVER_LOG="$LOGDIR/sharpedge_cockpit_server_${PORT}.log"

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
  python3 make_cockpit.py
}

start_server

echo "building cockpit once..."
if ! build_once; then
  echo "first build failed; keeping server up and retrying in loop"
fi

cat <<EOF

SharpEdge local dashboard is running without ADB/wireless/CDP.
Open manually in any browser on this phone:

  $URL

Regenerating every ${INTERVAL}s. Press Ctrl+C to stop the refresh loop.
EOF

while true; do
  sleep "$INTERVAL"
  build_once || echo "refresh failed; retrying next loop"
done
