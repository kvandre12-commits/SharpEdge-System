#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "[1/4] build trading view-model"
python3 phone_companion/build_golden_loop_view_model.py

echo "[2/4] emit prelaunch trace"
python3 phone_companion/emit_golden_loop_prelaunch_trace.py

launch_exit=0

echo "[3/4] launch trading dashboard"
set +e
bash phone_companion/launchers/run_phone_companion_trading.sh
launch_exit=$?
set -e

echo "[4/4] emit observation"
python3 phone_companion/emit_golden_loop_observation.py

cat <<'EOF'

Golden Loop artifacts:
- phone_companion/views/trading/golden_loop_view_model.json
- phone_companion/launchers/prelaunch_trace.json
- phone_companion/launchers/launch_result.json
- phone_companion/observations/golden_loop_latest.json

Optional audit-only trace:
- phone_companion/requests/golden_loop_request_trace.json
EOF

exit "$launch_exit"
