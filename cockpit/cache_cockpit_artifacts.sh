#!/data/data/com.termux/files/usr/bin/bash
# Snapshot generated cockpit artifacts and prune old snapshots.
#
# This is intentionally boring: copy the live dashboard artifacts into a small
# rotating cache so a fast cockpit loop can preserve audit breadcrumbs without
# turning outputs/ into a landfill with a ticker symbol.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${COCKPIT_ARTIFACT_CACHE_DIR:-$ROOT_DIR/outputs/cockpit_artifact_cache}"
MAX_SNAPSHOTS="${COCKPIT_ARTIFACT_CACHE_MAX:-8}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_DIR="$CACHE_DIR/$STAMP"
LATEST_DIR="$CACHE_DIR/latest"

copy_if_exists() {
  local src="$1"
  local dest_root="$2"
  local rel="${src#$ROOT_DIR/}"
  if [ -f "$src" ]; then
    mkdir -p "$dest_root/$(dirname "$rel")"
    cp "$src" "$dest_root/$rel"
  fi
}

copy_snapshot() {
  local dest="$1"
  mkdir -p "$dest"

  copy_if_exists "$ROOT_DIR/outputs/signal.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/ace_snapshot.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/setup_markers_spy.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/spy_scalp_dashboard.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/surface_execution_card.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/core_execution_spine_view.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/spine_realtime_adjustments.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/spine_realtime_audit/latest.json" "$dest"
  copy_if_exists "$ROOT_DIR/outputs/spine_realtime_audit/latest.txt" "$dest"

  copy_if_exists "$ROOT_DIR/cockpit/cockpit.html" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/gem_dashboard.html" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/runner_handoff_live.html" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/operator_surface.html" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/cockpit_chart.svg" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/gem_chart.svg" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/cockpit_weekly_context.svg" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/cockpit_monthly_context.svg" "$dest"
  copy_if_exists "$ROOT_DIR/cockpit/spy_scalp_chart.svg" "$dest"

  cat >"$dest/manifest.json" <<JSON
{
  "schema": "sharpedge.cockpit_artifact_cache.v1",
  "snapshot_utc": "$STAMP",
  "source_root": "$ROOT_DIR",
  "max_snapshots": $MAX_SNAPSHOTS
}
JSON
}

prune_old_snapshots() {
  python3 - "$CACHE_DIR" "$MAX_SNAPSHOTS" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

cache = Path(sys.argv[1])
keep = max(1, int(sys.argv[2]))
snapshots = sorted(
    path
    for path in cache.iterdir()
    if path.is_dir() and path.name != "latest"
)
for stale in snapshots[:-keep]:
    shutil.rmtree(stale, ignore_errors=True)
PY
}

mkdir -p "$CACHE_DIR"
rm -rf "$LATEST_DIR.tmp"
copy_snapshot "$LATEST_DIR.tmp"
rm -rf "$LATEST_DIR"
mv "$LATEST_DIR.tmp" "$LATEST_DIR"
copy_snapshot "$SNAPSHOT_DIR"
prune_old_snapshots

echo "cockpit artifact cache dumped -> $SNAPSHOT_DIR"
