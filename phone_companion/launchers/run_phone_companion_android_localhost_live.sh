#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${SHARPEDGE_ANDROID_LOCAL_PORT:-8765}"
INTERVAL_SECONDS="${SHARPEDGE_ANDROID_REFRESH_SECONDS:-45}"

cd "$REPO_ROOT"
mkdir -p outputs

server_is_up() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.25)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

ensure_server() {
  if server_is_up; then
    return
  fi
  nohup python3 -m http.server "$PORT" --bind 127.0.0.1 \
    > outputs/android_localhost_server.log 2>&1 &
  echo "$!" > outputs/android_localhost_server.pid
  sleep 1
}

refresh_once() {
  date -Is
  python3 cockpit/make_cockpit.py
  bash cockpit/refresh_operator_surface_inputs.sh
  python3 cockpit/make_operator_surface.py
  python3 phone_companion/export_operator_packet_to_android.py \
    > outputs/android_localhost_export_latest.json
}

ensure_server
while true; do
  refresh_once
  sleep "$INTERVAL_SECONDS"
done
