#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export SHARPEDGE_AUTHORITY_ENGINE=ace
exec bash cockpit/run_local_dashboard.sh
