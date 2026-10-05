#!/data/data/com.termux/files/usr/bin/bash
# Send a small Android reminder that the SharpEdge cockpit is ready to open.
#
# Usage:
#   bash cockpit/remind_open_cockpit.sh
#   COCKPIT_REMINDER_OPEN_NOW=1 bash cockpit/remind_open_cockpit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${COCKPIT_PORT:-8777}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
OPEN_SCRIPT="$SCRIPT_DIR/open_cockpit.sh"
TITLE="SharpEdge cockpit ready"
CONTENT="Tap to open the live cockpit: $URL"

if [ "${COCKPIT_REMINDER_OPEN_NOW:-}" = "1" ]; then
  bash "$OPEN_SCRIPT"
fi

if command -v termux-notification >/dev/null 2>&1; then
  termux-notification \
    --id sharpedge-cockpit-open \
    --title "$TITLE" \
    --content "$CONTENT" \
    --icon open_in_browser \
    --priority high \
    --button1 "Open" \
    --button1-action "$OPEN_SCRIPT" \
    --action "$OPEN_SCRIPT"
  echo "sent cockpit reminder notification"
  exit 0
fi

if command -v termux-toast >/dev/null 2>&1; then
  termux-toast "$CONTENT"
  echo "sent cockpit reminder toast"
  exit 0
fi

echo "$TITLE"
echo "$CONTENT"
