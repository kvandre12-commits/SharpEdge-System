#!/data/data/com.termux/files/usr/bin/bash
# Re-open the SharpEdge cockpit in the browser without restarting the live loop.
#
# Usage:
#   bash cockpit/open_cockpit.sh
#   COCKPIT_PORT=9000 bash cockpit/open_cockpit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${COCKPIT_PORT:-8777}"
URL="http://127.0.0.1:${PORT}/cockpit.html"

open_url() {
  if command -v am >/dev/null 2>&1; then
    am start -a android.intent.action.VIEW -p com.brave.browser -d "$URL" \
      >/dev/null 2>&1 && echo "opened Brave -> $URL" \
      || echo "(could not auto-open Brave; browse to $URL yourself)"
    return
  fi
  if command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$URL" && echo "opened browser -> $URL"
    return
  fi
  echo "open this in your browser: $URL"
}

if ! curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then
  echo "(warning: localhost server on :${PORT} does not look live yet)"
fi

open_url
