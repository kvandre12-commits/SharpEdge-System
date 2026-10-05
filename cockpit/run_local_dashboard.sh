#!/data/data/com.termux/files/usr/bin/bash
# SharpEdge local dashboard runner.
#
# Purpose:
#   Run the SPY cockpit as a plain localhost dashboard with no ADB, no wireless
#   debugging, no CDP, and no Android browser automation.
#
# Usage:
#   bash cockpit/run_local_dashboard.sh
#   COCKPIT_PORT=8777 COCKPIT_INTERVAL=5 bash cockpit/run_local_dashboard.sh
#   COCKPIT_ARTIFACT_CACHE_EVERY=60 COCKPIT_ARTIFACT_CACHE_MAX=8 bash cockpit/run_local_dashboard.sh
#   COCKPIT_OPEN_BROWSER=1 bash cockpit/run_local_dashboard.sh
#   COCKPIT_OPEN_BROWSER=1 COCKPIT_OPEN_OPERATOR_SURFACE=1 bash cockpit/run_local_dashboard.sh
#   SHARPEDGE_SPINE_REALTIME_ADJUST=1 bash cockpit/run_local_dashboard.sh
#   COCKPIT_NERV_SYMBOLS=SPY,WMT COCKPIT_NERV_EVERY=300 bash cockpit/run_local_dashboard.sh
#
# Separate adaptive audit loop:
#   bash cockpit/run_spine_realtime_auditor.sh
#
# Then open these URLs manually in any browser on the phone, or let the script
# open them via Android intents when COCKPIT_OPEN_BROWSER=1:
#   http://127.0.0.1:8777/cockpit.html
#   http://127.0.0.1:8777/operator_surface.html
#   http://127.0.0.1:8777/runner_handoff_live.html
#   http://127.0.0.1:8777/regime_nerv_split.html

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COCKPIT_DIR="$ROOT_DIR/cockpit"
PORT="${COCKPIT_PORT:-8777}"
INTERVAL="${COCKPIT_INTERVAL:-5}"
URL="http://127.0.0.1:${PORT}/cockpit.html"
OPERATOR_SURFACE_URL="http://127.0.0.1:${PORT}/operator_surface.html"
RUNNER_HANDOFF_URL="http://127.0.0.1:${PORT}/runner_handoff_live.html"
REGIME_NERV_SPLIT_URL="http://127.0.0.1:${PORT}/regime_nerv_split.html"
REGIME_NERV_TABS_URL="http://127.0.0.1:${PORT}/regime_nerv_tabs.html"
HEY_GUY_URL="http://127.0.0.1:${PORT}/hey_guy.html"
LOGDIR="${TMPDIR:-$HOME/.cache}"
SERVER_LOG="$LOGDIR/sharpedge_cockpit_server_${PORT}.log"
AUTHORITY_ENGINE="${SHARPEDGE_AUTHORITY_ENGINE:-legacy}"
OPEN_BROWSER="${COCKPIT_OPEN_BROWSER:-}"
OPEN_OPERATOR_SURFACE="${COCKPIT_OPEN_OPERATOR_SURFACE:-1}"
ARTIFACT_CACHE_EVERY="${COCKPIT_ARTIFACT_CACHE_EVERY:-60}"
OPERATOR_SURFACE_EVERY="${COCKPIT_OPERATOR_SURFACE_EVERY:-60}"
STANDARD_NERV_SYMBOLS="${COCKPIT_NERV_SYMBOLS:-SPY,WMT}"
STANDARD_NERV_EVERY="${COCKPIT_NERV_EVERY:-300}"
STANDARD_NERV_MAX_EXPIRATIONS="${COCKPIT_NERV_MAX_EXPIRATIONS:-6}"
STANDARD_NERV_BOARD_LIMIT="${COCKPIT_NERV_BOARD_LIMIT:-80}"
STANDARD_NERV_OUTPUT_DIR="outputs/nerv_cockpit_standard"
PYTHON_BIN="${SHARPEDGE_PYTHON:-python3}"
LAST_ARTIFACT_CACHE_DUMP=0
LAST_OPERATOR_SURFACE_REFRESH=0
LAST_STANDARD_NERV_REFRESH=0

mkdir -p "$LOGDIR"
cd "$COCKPIT_DIR"

configure_python_runtime() {
  if "$PYTHON_BIN" -c 'import numpy, requests' >/dev/null 2>&1; then
    return 0
  fi

  # Code Puppy and uv may prepend a managed Python to PATH. On Termux that
  # interpreter cannot see native packages installed by pkg, notably NumPy.
  # Reuse the matching Termux system site-packages instead of rebuilding NumPy.
  if [ -n "${PREFIX:-}" ] && [ -x "$PREFIX/bin/python3" ]; then
    local termux_site_packages
    termux_site_packages="$($PREFIX/bin/python3 -c 'import site; print(site.getsitepackages()[0])')"
    if [ -d "$termux_site_packages" ]; then
      export PYTHONPATH="$termux_site_packages${PYTHONPATH:+:$PYTHONPATH}"
    fi
  fi

  if ! "$PYTHON_BIN" -c 'import numpy, requests' >/dev/null 2>&1; then
    echo "SharpEdge runtime is missing required Python modules: numpy and/or requests." >&2
    echo "See docs/android_termux_dependencies.md or set SHARPEDGE_PYTHON." >&2
    return 1
  fi
}

port_is_live() {
  "$PYTHON_BIN" - "$PORT" <<'PY'
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

  "$PYTHON_BIN" -m http.server "$PORT" --bind 127.0.0.1 >"$SERVER_LOG" 2>&1 &
  echo "server started on :$PORT (pid $!, log $SERVER_LOG)"
}

build_cockpit_once() {
  "$PYTHON_BIN" make_cockpit.py
}

build_regime_nerv_panel_once() {
  "$PYTHON_BIN" regime_nerv_panel.py "$INTERVAL"
}

refresh_standard_nerv_once() {
  if [ "${COCKPIT_NERV_AUTO_REFRESH:-1}" = "0" ]; then
    return 0
  fi
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" scripts/nerv_free_data_adapter.py \
      --symbols "$STANDARD_NERV_SYMBOLS" \
      --max-expirations "$STANDARD_NERV_MAX_EXPIRATIONS" \
      --board-limit "$STANDARD_NERV_BOARD_LIMIT" \
      --output-dir "$STANDARD_NERV_OUTPUT_DIR" \
      --retention-hours 24
  )
}

maybe_refresh_standard_nerv() {
  local force="${1:-0}"
  if [ "${COCKPIT_NERV_AUTO_REFRESH:-1}" = "0" ]; then
    return 0
  fi
  local now
  now="$(date +%s)"
  if [ "$force" != "1" ] && [ $((now - LAST_STANDARD_NERV_REFRESH)) -lt "$STANDARD_NERV_EVERY" ]; then
    return 0
  fi
  LAST_STANDARD_NERV_REFRESH="$now"
  refresh_standard_nerv_once || echo "standard NERV refresh failed; continuing cockpit loop"
}

build_operator_surface_once() {
  local failed=0
  bash ./refresh_operator_surface_inputs.sh || failed=1
  "$PYTHON_BIN" make_operator_surface.py || failed=1
  return "$failed"
}

maybe_build_operator_surface() {
  local force="${1:-0}"
  local now
  now="$(date +%s)"
  if [ "$force" != "1" ] && [ $((now - LAST_OPERATOR_SURFACE_REFRESH)) -lt "$OPERATOR_SURFACE_EVERY" ]; then
    return 0
  fi
  LAST_OPERATOR_SURFACE_REFRESH="$now"
  build_operator_surface_once
}

build_once() {
  local failed=0
  maybe_refresh_standard_nerv || failed=1
  build_cockpit_once || failed=1
  build_regime_nerv_panel_once || failed=1
  maybe_build_operator_surface || failed=1
  return "$failed"
}

open_url() {
  local target_url="$1"
  local label="$2"
  if command -v am >/dev/null 2>&1; then
    am start -a android.intent.action.VIEW -p com.brave.browser -d "$target_url" \
      >/dev/null 2>&1 && echo "opened Brave -> $label: $target_url" \
      || echo "(could not auto-open Brave for $label; browse to $target_url yourself)"
  elif command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$target_url" && echo "opened browser -> $label: $target_url"
  else
    echo "open $label manually: $target_url"
  fi
}

maybe_open_surfaces() {
  if [ -z "$OPEN_BROWSER" ]; then
    return
  fi
  open_url "$URL" "cockpit"
  open_url "$REGIME_NERV_SPLIT_URL" "cockpit + regime/NERV split"
  if [ -n "$OPEN_OPERATOR_SURFACE" ]; then
    open_url "$OPERATOR_SURFACE_URL" "operator surface"
  fi
}

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

configure_python_runtime
start_server

echo "using Python runtime: $PYTHON_BIN"
echo "building cockpit once..."
if maybe_refresh_standard_nerv 1 && build_cockpit_once && build_regime_nerv_panel_once && maybe_build_operator_surface 1; then
  maybe_dump_artifact_cache
else
  echo "first build failed; keeping server up and retrying in loop"
fi

maybe_open_surfaces

cat <<EOF

SharpEdge local dashboard is running without ADB/wireless/CDP.
Authority engine: ${AUTHORITY_ENGINE}
First-class live surfaces:

  cockpit:             $URL
  operator surface:    $OPERATOR_SURFACE_URL
  runner handoff live: $RUNNER_HANDOFF_URL
  cockpit + regime:    $REGIME_NERV_SPLIT_URL
  regime tabs:         $REGIME_NERV_TABS_URL
  hey guy:             $HEY_GUY_URL

Regenerating cockpit/regime panel every ${INTERVAL}s. Standard NERV (${STANDARD_NERV_SYMBOLS}) every ${STANDARD_NERV_EVERY}s. Operator surface every ${OPERATOR_SURFACE_EVERY}s; artifact cache dump every ${ARTIFACT_CACHE_EVERY}s. Press Ctrl+C to stop the refresh loop.
EOF

while true; do
  sleep "$INTERVAL"
  if build_once; then
    maybe_dump_artifact_cache
  else
    echo "refresh failed; retrying next loop"
  fi
done
