"""Capture one operator-visible Robinhood Social feed into private evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .evidence_store import (
        CAPTURE_SCHEMA,
        SOURCE_TIER,
        parse_visible_nodes,
        store_capture,
    )
except ImportError:  # Direct script execution from the repository checkout.
    from evidence_store import (  # type: ignore[no-redef]
        CAPTURE_SCHEMA,
        SOURCE_TIER,
        parse_visible_nodes,
        store_capture,
    )

ROBINHOOD_PACKAGE = "com.robinhood.android"
DEFAULT_CORPUS_ROOT = Path(
    os.getenv("SHARPEDGE_ROBINHOOD_SOCIAL_ROOT", "~/.sharpedge/robinhood_social")
).expanduser()
REMOTE_HIERARCHY_PATH = "/sdcard/sharpedge_robinhood_social_capture.xml"
SENSITIVE_MARKERS = ("Buying power", "Total portfolio value", "Account number")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime) -> str:
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


class AdbClient:
    """Small read-only ADB seam used by the manual capture command."""

    def __init__(self, executable: str = "adb") -> None:
        self.executable = executable

    def _run(self, *args: str, text: bool = False) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            [self.executable, *args],
            check=True,
            capture_output=True,
            text=text,
            timeout=30,
        )

    def connected_device(self) -> tuple[str, str]:
        output = self._run("devices", "-l", text=True).stdout
        devices = []
        for line in output.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2 and fields[1] == "device":
                devices.append((fields[0], line))
        if len(devices) != 1:
            raise RuntimeError(
                f"expected exactly one authorized ADB device; found {len(devices)}"
            )
        return devices[0]

    def shell_text(self, *args: str) -> str:
        return self._run("shell", *args, text=True).stdout

    def screenshot(self) -> bytes:
        return self._run("exec-out", "screencap", "-p").stdout

    def hierarchy(self) -> bytes:
        try:
            self.shell_text("uiautomator", "dump", REMOTE_HIERARCHY_PATH)
            return self._run("exec-out", "cat", REMOTE_HIERARCHY_PATH).stdout
        finally:
            subprocess.run(
                [self.executable, "shell", "rm", "-f", REMOTE_HIERARCHY_PATH],
                check=False,
                capture_output=True,
                timeout=10,
            )


def _extract_activity(activity_dump: str) -> str:
    patterns = (
        rf"(?:mResumedActivity|topResumedActivity|mFocusedApp).*?({ROBINHOOD_PACKAGE}/[^\s}}]+)",
        rf"mFocusedWindow=.*?({ROBINHOOD_PACKAGE}/[^\s}}]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, activity_dump)
        if match:
            return match.group(1)
    raise RuntimeError("Robinhood is not the foreground Android activity")


def _extract_package_value(package_dump: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}=([^\s]+)", package_dump)
    return match.group(1) if match else None


def _validate_social_feed(nodes: list[dict[str, Any]]) -> None:
    visible = {
        value
        for node in nodes
        for value in (node["text"], node["content_description"])
        if value
    }
    if "Social" not in visible or not {"For you", "Following"}.intersection(visible):
        raise RuntimeError("foreground Robinhood screen is not the Social feed")
    leaked = [marker for marker in SENSITIVE_MARKERS if marker in visible]
    if leaked:
        raise RuntimeError(
            f"refusing capture with account-sensitive marker: {leaked[0]}"
        )


def record_observation(
    *,
    corpus_root: Path,
    db_path: Path,
    screenshot: bytes,
    hierarchy_xml: bytes,
    provenance: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    """Write immutable artifacts and index one validated manual observation."""
    nodes = parse_visible_nodes(hierarchy_xml)
    _validate_social_feed(nodes)

    observed_at_utc = _iso_utc(observed_at)
    screenshot_hash = _sha256(screenshot)
    hierarchy_hash = _sha256(hierarchy_xml)
    identity = f"{observed_at_utc}:{screenshot_hash}:{hierarchy_hash}".encode()
    capture_id = _sha256(identity)[:24]
    date_part = observed_at.astimezone(UTC).date().isoformat()
    relative_root = Path("raw") / date_part / capture_id
    capture_dir = corpus_root / relative_root
    if capture_dir.exists():
        raise FileExistsError(f"immutable capture already exists: {capture_dir}")

    staging = corpus_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        screenshot_path = staging / "screenshot.png"
        hierarchy_path = staging / "hierarchy.xml"
        screenshot_path.write_bytes(screenshot)
        hierarchy_path.write_bytes(hierarchy_xml)

        artifacts = [
            {
                "kind": "screenshot",
                "relative_path": str(relative_root / screenshot_path.name),
                "sha256": screenshot_hash,
                "byte_count": len(screenshot),
            },
            {
                "kind": "ui_hierarchy",
                "relative_path": str(relative_root / hierarchy_path.name),
                "sha256": hierarchy_hash,
                "byte_count": len(hierarchy_xml),
            },
        ]
        manifest = {
            "schema": CAPTURE_SCHEMA,
            "capture_id": capture_id,
            "observed_at_utc": observed_at_utc,
            "source_tier": SOURCE_TIER,
            "capture_mode": "manual_curated",
            "artifact_root": str(relative_root),
            "artifacts": artifacts,
            "visible_node_count": len(nodes),
            "provenance": provenance,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        (staging / "manifest.json").write_bytes(manifest_bytes)

        capture_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(capture_dir)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
        raise

    manifest_relative_path = str(relative_root / "manifest.json")
    store_capture(
        db_path,
        manifest,
        nodes,
        manifest_path=manifest_relative_path,
        manifest_sha256=_sha256(manifest_bytes),
    )
    return {
        "status": "captured",
        "schema": CAPTURE_SCHEMA,
        "capture_id": capture_id,
        "observed_at_utc": observed_at_utc,
        "source_tier": SOURCE_TIER,
        "capture_mode": "manual_curated",
        "corpus_root": str(corpus_root),
        "db_path": str(db_path),
        "artifact_root": str(capture_dir),
        "visible_node_count": len(nodes),
        "artifact_hashes": {
            artifact["kind"]: artifact["sha256"] for artifact in artifacts
        },
    }


def capture_from_adb(
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    db_path: Path | None = None,
    *,
    adb: AdbClient | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    client = adb or AdbClient()
    serial, device_line = client.connected_device()
    activity_dump = client.shell_text("dumpsys", "activity", "activities")
    activity_name = _extract_activity(activity_dump)
    package_dump = client.shell_text("dumpsys", "package", ROBINHOOD_PACKAGE)
    device_model = client.shell_text("getprop", "ro.product.model").strip()

    provenance = {
        "collector": "sharpedge_robinhood_social_manual_capture",
        "collector_version": 1,
        "package_name": ROBINHOOD_PACKAGE,
        "activity_name": activity_name,
        "app_version_name": _extract_package_value(package_dump, "versionName"),
        "app_version_code": _extract_package_value(package_dump, "versionCode"),
        "device_model": device_model,
        "device_transport_hash": _sha256(serial.encode()),
        "adb_device_metadata_hash": _sha256(device_line.encode()),
        "social_surface": "feed",
    }
    root = corpus_root.expanduser()
    database = (db_path or root / "robinhood_social_research.db").expanduser()
    return record_observation(
        corpus_root=root,
        db_path=database,
        screenshot=client.screenshot(),
        hierarchy_xml=client.hierarchy(),
        provenance=provenance,
        observed_at=observed_at or _utc_now(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the foreground Robinhood Social feed as private evidence."
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--adb", default="adb", help="ADB executable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = capture_from_adb(
        corpus_root=args.corpus_root,
        db_path=args.db_path,
        adb=AdbClient(args.adb),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
