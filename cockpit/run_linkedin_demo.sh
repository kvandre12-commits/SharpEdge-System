#!/data/data/com.termux/files/usr/bin/bash
# One-shot cockpit prep for recording a LinkedIn demo.
# Builds current artifacts, serves cockpit.html, opens Brave, then exits.
#
# Usage:
#   bash cockpit/run_linkedin_demo.sh
#   COCKPIT_NO_BROWSER=1 bash cockpit/run_linkedin_demo.sh
#
# Cute, tiny, and not pretending to be a platform.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
COCKPIT_BUILD_ONCE=1 bash ./run_cockpit.sh
