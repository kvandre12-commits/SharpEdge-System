"""Export the latest SharpEdge signal contract into the native Android viewer.

This is a development bridge, not live sync. It copies the current
``outputs/signal.json`` into the SharpEdge-Android sample asset so the native
viewer can render the same Trade Gate after rebuild/reinstall.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

DEFAULT_SIGNAL_PATH = Path("outputs/signal.json")
DEFAULT_ANDROID_ROOT = Path.home() / "SharpEdge-Android"
DEFAULT_PROOF_PATH = Path("phone_companion/views/trading/android_viewer_export.json")
TARGET_RELATIVE_PATHS = [
    Path("app/src/main/assets/sample_signal.json"),
    Path("app_contracts/sharpedge.signal.v1.sample.json"),
]
REQUIRED_TRADE_PERMISSION_FIELDS = [
    "trade_gate",
    "trade_permission_score",
    "bias",
    "supporting_reasons",
    "warning_reasons",
    "scores",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_signal(signal: dict) -> None:
    if signal.get("schema") != "sharpedge.signal.v1":
        raise ValueError("signal schema must be sharpedge.signal.v1")
    trade_permission = signal.get("trade_permission")
    if not isinstance(trade_permission, dict):
        raise ValueError("signal.trade_permission must be a JSON object")
    missing = [
        field
        for field in REQUIRED_TRADE_PERMISSION_FIELDS
        if field not in trade_permission
    ]
    if missing:
        raise ValueError(f"trade_permission missing fields: {', '.join(missing)}")


def export_signal(
    signal_path: Path = DEFAULT_SIGNAL_PATH,
    android_root: Path = DEFAULT_ANDROID_ROOT,
    proof_path: Path = DEFAULT_PROOF_PATH,
) -> dict:
    signal = _load_json(signal_path)
    _validate_signal(signal)

    written_paths = []
    for relative_path in TARGET_RELATIVE_PATHS:
        target = android_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(signal, indent=2) + "\n", encoding="utf-8")
        written_paths.append(str(target))

    trade_permission = signal["trade_permission"]
    proof = {
        "artifact_type": "sharpedge_android_viewer_export",
        "status": "exported",
        "source_signal_path": str(signal_path),
        "android_root": str(android_root),
        "written_paths": written_paths,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "signal_ts": signal.get("ts"),
        "symbol": signal.get("symbol"),
        "spot": signal.get("spot"),
        "trade_permission": {
            "trade_gate": trade_permission.get("trade_gate"),
            "trade_permission_score": trade_permission.get("trade_permission_score"),
            "bias": trade_permission.get("bias"),
        },
        "note": "Rebuild/reinstall SharpEdge-Android to view this exact packaged signal.",
    }
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main(argv: list[str]) -> int:
    signal_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_SIGNAL_PATH
    android_root = Path(argv[2]) if len(argv) > 2 else DEFAULT_ANDROID_ROOT
    proof_path = Path(argv[3]) if len(argv) > 3 else DEFAULT_PROOF_PATH
    proof = export_signal(signal_path, android_root, proof_path)
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
