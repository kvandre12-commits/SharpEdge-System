#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python3 cockpit/make_cockpit.py
bash cockpit/refresh_operator_surface_inputs.sh
python3 cockpit/make_operator_surface.py
bash "$SCRIPT_DIR/run_phone_companion_android_operator_import.sh" "$@"
