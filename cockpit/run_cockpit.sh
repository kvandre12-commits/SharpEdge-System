#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge live cockpit launcher.
# Starts a local web server, opens the dashboard in Brave, and
# regenerates the read every 45s.
#
#   bash run_cockpit.sh            # serve + auto-open Brave + live loop
#   COCKPIT_NO_BROWSER=1 bash ...  # serve + loop only, skip Brave
#   COCKPIT_PORT=9000 bash ...     # use a different port

set -e
cd "$(dirname "$0")"
PORT="${COCKPIT_PORT:-8777}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
mkdir -p "$LOGDIR"

# start static server if not already running
if ! curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
  python3 -m http.server "$PORT" >"$LOGDIR/cockpit_server.log" 2>&1 &
  echo "server started on :${PORT} (pid $!)"
fi

# generate the page once BEFORE opening the browser so Brave never
# lands on a 404 / blank tab
python3 make_cockpit.py >/dev/null 2>&1 || echo "(first build hiccuped, loop will retry)"

# auto-open Brave (Android/Termux). Skippable via COCKPIT_NO_BROWSER=1.
if [ -z "${COCKPIT_NO_BROWSER:-}" ]; then
  if command -v am >/dev/null 2>&1; then
    am start -a android.intent.action.VIEW -p com.brave.browser -d "$URL" \
      >/dev/null 2>&1 && echo "opened Brave -> $URL" \
      || echo "(could not auto-open Brave; browse to $URL yourself)"
  elif command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$URL" && echo "opened browser -> $URL"
  else
    echo "open this in your browser: $URL"
  fi
fi

echo "regenerating cockpit every 45s -- Ctrl+C to stop"
while true; do
  python3 make_cockpit.py 2>/dev/null \
    | grep -E "spot|BULLS|BEARS|BALANCED" || echo "(refresh failed, retrying)"
  sleep 45
done
