#!/data/data/com.termux/files/usr/bin/bash
# Opinionated operator startup wrapper for the SharpEdge local station.
#
# This just sets the sane defaults so the operator gets:
# - cockpit loop
# - operator surface refresh
# - richer NERV liquidity coverage
# - browser auto-open on Android/Termux

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${COCKPIT_PORT:-8777}"

export COCKPIT_OPEN_BROWSER="${COCKPIT_OPEN_BROWSER:-1}"
export COCKPIT_OPEN_OPERATOR_SURFACE="${COCKPIT_OPEN_OPERATOR_SURFACE:-1}"
export COCKPIT_INTERVAL="${COCKPIT_INTERVAL:-5}"
export COCKPIT_REFRESH_SECONDS="${COCKPIT_REFRESH_SECONDS:-$COCKPIT_INTERVAL}"
export COCKPIT_PAGE_REFRESH_SECONDS="${COCKPIT_PAGE_REFRESH_SECONDS:-$COCKPIT_REFRESH_SECONDS}"
export COCKPIT_OPERATOR_SURFACE_EVERY="${COCKPIT_OPERATOR_SURFACE_EVERY:-60}"
export COCKPIT_ARTIFACT_CACHE_EVERY="${COCKPIT_ARTIFACT_CACHE_EVERY:-60}"
export COCKPIT_NERV_EVERY="${COCKPIT_NERV_EVERY:-300}"
export COCKPIT_NERV_MAX_EXPIRATIONS="${COCKPIT_NERV_MAX_EXPIRATIONS:-6}"
export COCKPIT_NERV_BOARD_LIMIT="${COCKPIT_NERV_BOARD_LIMIT:-120}"
export COCKPIT_NERV_SYMBOLS="${COCKPIT_NERV_SYMBOLS:-SPY,WMT,AAPL,AMZN,GOOGL,META,MSFT,PLTR}"
export POSITION_LAB_NERV_MAX_EXPIRATIONS="${POSITION_LAB_NERV_MAX_EXPIRATIONS:-20}"
export AGENT_LANGUAGE_AGENT_ID="${AGENT_LANGUAGE_AGENT_ID:-code-puppy-fd03ba}"

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock >/dev/null 2>&1 || true
fi

cat <<EOF
Starting SharpEdge operator station.

Live URLs once the server is up:
  cockpit:          http://127.0.0.1:${PORT}/cockpit.html
  operator surface: http://127.0.0.1:${PORT}/operator_surface.html
  hey guy:          http://127.0.0.1:${PORT}/hey_guy.html
  nerv tabs:        http://127.0.0.1:${PORT}/regime_nerv_tabs.html

Defaults:
  NERV symbols:     ${COCKPIT_NERV_SYMBOLS}
  expirations:      ${COCKPIT_NERV_MAX_EXPIRATIONS}
  position lab exp: ${POSITION_LAB_NERV_MAX_EXPIRATIONS}
  cockpit interval: ${COCKPIT_INTERVAL}s
  NERV refresh:     ${COCKPIT_NERV_EVERY}s

Yeah, this is the less-annoying boot path.
EOF

cd "$ROOT_DIR"
exec bash "$ROOT_DIR/cockpit/run_local_dashboard.sh"
