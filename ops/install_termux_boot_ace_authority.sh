#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_DIR="$HOME/.termux/boot"
BOOT_SCRIPT="$BOOT_DIR/start_sharpedge_ace_authority_scheduler.sh"
OUTPUT_DIR="$ROOT_DIR/outputs"

mkdir -p "$BOOT_DIR" "$OUTPUT_DIR"

cat > "$BOOT_SCRIPT" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$ROOT_DIR"
nohup bash ops/run_ace_authority_market_open_daemon.sh >> outputs/ace_authority_termux_boot.log 2>&1 &
EOF

chmod +x "$BOOT_SCRIPT"

echo "installed Termux:Boot launcher -> $BOOT_SCRIPT"
echo "next: enable the Termux:Boot app/plugin, then reboot or run the daemon manually"
