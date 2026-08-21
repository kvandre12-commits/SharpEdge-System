#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge live cockpit launcher.
# Starts a local web server, opens the dashboard in Brave, and
# regenerates the read every 5s by default.
#
#   bash run_cockpit.sh                       # serve + auto-open Brave + live loop
#   COCKPIT_BUILD_ONCE=1 bash run_cockpit.sh # build/open once, then exit for recording
#   COCKPIT_NO_BROWSER=1 bash ...             # serve + build, skip browser open
#   COCKPIT_REMIND=1 bash ...                 # also send a tap-to-open reminder notification
#   COCKPIT_PORT=9000 bash ...                # use a different port
#   COCKPIT_REFRESH_SECONDS=5 bash ...        # adjust live refresh loop interval
#   COCKPIT_ARTIFACT_CACHE_EVERY=60 bash ...  # dump/prune artifacts periodically

set -e
cd "$(dirname "$0")"
PORT="${COCKPIT_PORT:-8777}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
REFRESH_SECONDS="${COCKPIT_REFRESH_SECONDS:-5}"
BUILD_ONCE="${COCKPIT_BUILD_ONCE:-}"
REMIND="${COCKPIT_REMIND:-}"
ARTIFACT_CACHE_EVERY="${COCKPIT_ARTIFACT_CACHE_EVERY:-60}"
LAST_ARTIFACT_CACHE_DUMP=0
mkdir -p "$LOGDIR"

# start static server if not already running
if ! curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
  python3 -m http.server "$PORT" >"$LOGDIR/cockpit_server.log" 2>&1 &
  echo "server started on :${PORT} (pid $!)"
fi

# generate the page once BEFORE opening the browser so Brave never
# lands on a 404 / blank tab
python3 make_cockpit.py >/dev/null 2>&1 || echo "(first build hiccuped, loop will retry)"

maybe_dump_artifact_cache() {
  if [ "${COCKPIT_ARTIFACT_CACHE:-1}" = "0" ]; then
    return
  fi
  local now
  now="$(date +%s)"
  if [ $((now - LAST_ARTIFACT_CACHE_DUMP)) -lt "$ARTIFACT_CACHE_EVERY" ]; then
    return
  fi
  LAST_ARTIFACT_CACHE_DUMP="$now"
  bash ./cache_cockpit_artifacts.sh || echo "artifact cache dump failed; continuing live loop"
}

maybe_dump_artifact_cache

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

if [ -n "$REMIND" ] && [ -x "./remind_open_cockpit.sh" ]; then
  COCKPIT_PORT="$PORT" bash ./remind_open_cockpit.sh >/dev/null 2>&1 \
    && echo "sent cockpit reminder notification"
fi

if [ -n "$BUILD_ONCE" ]; then
  cat <<EOF
ready for recording
- URL: $URL
- HTML: $(pwd)/cockpit.html
- signal: $(cd .. && pwd)/outputs/signal.json
- mode: one-shot build/open
EOF
  exit 0
fi

echo "regenerating cockpit every ${REFRESH_SECONDS}s; artifact cache every ${ARTIFACT_CACHE_EVERY}s -- Ctrl+C to stop"
while true; do
  if python3 make_cockpit.py 2>/dev/null \
    | grep -E "spot|BULLS|BEARS|BALANCED"; then
    maybe_dump_artifact_cache
  else
    echo "(refresh failed, retrying)"
  fi
  sleep "$REFRESH_SECONDS"
done
