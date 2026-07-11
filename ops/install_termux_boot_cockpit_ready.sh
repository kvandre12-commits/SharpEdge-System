#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_DIR="$HOME/.termux/boot"
BOOT_SCRIPT="$BOOT_DIR/start_sharpedge_cockpit_ready_scheduler.sh"
OUTPUT_DIR="$ROOT_DIR/outputs"
TRIGGER_HHMM="${SHARPEDGE_READY_HHMM:-0900}"
TRIGGER_TZ="${SHARPEDGE_READY_TZ:-America/New_York}"

mkdir -p "$BOOT_DIR" "$OUTPUT_DIR"

cat > "$BOOT_SCRIPT" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$ROOT_DIR"
nohup env SHARPEDGE_READY_HHMM="$TRIGGER_HHMM" SHARPEDGE_READY_TZ="$TRIGGER_TZ" bash ops/run_cockpit_ready_daemon.sh >> outputs/cockpit_ready_termux_boot.log 2>&1 &
EOF

chmod +x "$BOOT_SCRIPT"

echo "installed Termux:Boot launcher -> $BOOT_SCRIPT"
echo "daily trigger: $TRIGGER_HHMM ($TRIGGER_TZ), weekdays only"
echo "next: enable the Termux:Boot app/plugin, then reboot or run the daemon manually"
