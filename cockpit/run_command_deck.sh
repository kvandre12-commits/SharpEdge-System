#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge Command Deck launcher.
# Serves the deck, opens it in Brave, and refreshes signal+deck every 45s.
# The deck shows the whole chain: SIGNAL -> GATE -> DECISION + scoreboard.
#
#   bash run_command_deck.sh             # serve + auto-open Brave + live loop
#   DECK_NO_BROWSER=1 bash ...           # serve + loop only, skip Brave
#   DECK_PORT=9000 bash ...              # use a different port

set -e
cd "$(dirname "$0")"
PORT="${DECK_PORT:-8778}"
URL="http://127.0.0.1:${PORT}/command_deck.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
mkdir -p "$LOGDIR"

if ! curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
  python3 -m http.server "$PORT" >"$LOGDIR/command_deck_server.log" 2>&1 &
  echo "server started on :${PORT} (pid $!)"
fi

# build signal first (writes signal.json) then the deck that consumes it,
# so Brave never lands on a stale/blank deck.
python3 make_cockpit.py >/dev/null 2>&1 || echo "(signal build hiccuped, loop will retry)"
python3 make_command_deck.py >/dev/null 2>&1 || echo "(deck build hiccuped, loop will retry)"

if [ -z "${DECK_NO_BROWSER:-}" ]; then
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

echo "refreshing signal + deck every 45s -- Ctrl+C to stop"
while true; do
  python3 make_cockpit.py >/dev/null 2>&1 || echo "(signal refresh failed)"
  python3 make_command_deck.py 2>/dev/null \
    | grep -E "wrote" >/dev/null || echo "(deck refresh failed, retrying)"
  echo "deck refreshed $(date '+%H:%M:%S')"
  sleep 45
done
