#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge live cockpit launcher.
# Starts a local web server and regenerates the read every 45s.
# Open http://127.0.0.1:8777/cockpit.html in Brave to watch it live.

set -e
cd "$(dirname "$0")"
PORT="${COCKPIT_PORT:-8777}"

# start static server if not already running
if ! curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
  python3 -m http.server "$PORT" >/tmp/cockpit_server.log 2>&1 &
  echo "server started on :${PORT} (pid $!)"
fi

echo "regenerating cockpit every 45s -- Ctrl+C to stop"
while true; do
  python3 make_cockpit.py 2>/dev/null \
    | grep -E "spot|BULLS|BEARS|BALANCED" || echo "(refresh failed, retrying)"
  sleep 45
done
