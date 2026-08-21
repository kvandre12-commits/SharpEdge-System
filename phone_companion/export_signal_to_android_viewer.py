"""Export the latest SharpEdge signal contract into the native Android viewer.

This is a development bridge. It copies the current ``outputs/signal.json`` into
SharpEdge-Android sample assets for packaged builds, and it also writes a
shareable live-import JSON artifact that the Android app can open/share-import
without rebuild.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

DEFAULT_SIGNAL_PATH = Path("outputs/signal.json")
DEFAULT_ANDROID_ROOT = Path.home() / "SharpEdge-Android"
DEFAULT_PROOF_PATH = Path("phone_companion/views/trading/android_viewer_export.json")
DEFAULT_LIVE_IMPORT_PATH = Path(
    "phone_companion/views/trading/sharpedge_android_live_import.json"
)
TARGET_RELATIVE_PATHS = [
    Path("app/src/main/assets/sample_signal.json"),
    Path("app_contracts/sharpedge.signal.v1.sample.json"),
]
ANDROID_VIEWER_BUNDLE_KEY = "android_viewer_bundle"
ANDROID_VIEWER_BUNDLE_SCHEMA = "sharpedge.android_viewer_bundle.v1"
REQUIRED_COCKPIT_ARTIFACTS = {
    "cockpit_html": "cockpit.html",
    "cockpit_chart_svg": "cockpit_chart.svg",
}
OPTIONAL_COCKPIT_ARTIFACTS = {
    "cockpit_weekly_context_svg": "cockpit_weekly_context.svg",
    "cockpit_monthly_context_svg": "cockpit_monthly_context.svg",
}
COCKPIT_ASSET_TARGETS = {
    "cockpit_html": Path("app/src/main/assets/sample_cockpit.html"),
    "cockpit_chart_svg": Path("app/src/main/assets/sample_cockpit_chart.svg"),
    "cockpit_weekly_context_svg": Path(
        "app/src/main/assets/sample_cockpit_weekly_context.svg"
    ),
    "cockpit_monthly_context_svg": Path(
        "app/src/main/assets/sample_cockpit_monthly_context.svg"
    ),
}
REQUIRED_TRADE_PERMISSION_FIELDS = [
    "trade_gate",
    "trade_permission_score",
    "bias",
    "supporting_reasons",
    "warning_reasons",
    "scores",
]
WEB_VIEWER_REFRESH_SCHEMA = "sharpedge.web_viewer_refresh.v1"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _repo_root_for_signal(signal_path: Path) -> Path:
    return signal_path.parent.parent


def _cockpit_artifact_paths(
    signal_path: Path,
    artifacts: dict[str, str],
) -> dict[str, Path]:
    cockpit_root = _repo_root_for_signal(signal_path) / "cockpit"
    return {key: cockpit_root / file_name for key, file_name in artifacts.items()}


def _looks_like_html(text: str) -> bool:
    lower = text.lower()
    return "<html" in lower or "<!doctype html" in lower


def _looks_like_svg(text: str) -> bool:
    return "<svg" in text.lower()


def _sanitize_html_for_android_viewer(html: str) -> str:
    return re.sub(
        r"<meta\s+http-equiv=(['\"])refresh\1\s+content=(['\"]).*?\2\s*/?>",
        "",
        html,
        flags=re.IGNORECASE,
    )


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _meta_refresh_seconds(html: str) -> int | None:
    for match in re.finditer(r"<meta\s+[^>]*>", html, flags=re.IGNORECASE):
        tag = match.group(0)
        if not re.search(r"http-equiv\s*=\s*(['\"])refresh\1", tag, re.IGNORECASE):
            continue
        content = re.search(r"content\s*=\s*(['\"])([^'\"]+)\1", tag, re.IGNORECASE)
        if not content:
            return None
        first_part = content.group(2).split(";", maxsplit=1)[0].strip()
        return int(first_part) if first_part.isdigit() else None
    return None


def validate_web_viewer_refresh(signal_path: Path) -> dict:
    """Prove the source web cockpit refreshes before Android copies anything.

    Android imports intentionally strip meta-refresh from bundled HTML. The source
    localhost web viewer must still carry the refresh contract so the live browser
    keeps pulling regenerated cockpit data.
    """

    signal = _load_json(signal_path)
    cockpit_paths = _cockpit_artifact_paths(signal_path, REQUIRED_COCKPIT_ARTIFACTS)
    cockpit_html_path = cockpit_paths["cockpit_html"]
    cockpit_chart_path = cockpit_paths["cockpit_chart_svg"]
    if not cockpit_html_path.is_file():
        raise FileNotFoundError(f"missing source web cockpit: {cockpit_html_path}")
    if not cockpit_chart_path.is_file():
        raise FileNotFoundError(f"missing source cockpit chart: {cockpit_chart_path}")

    cockpit_html = _load_text(cockpit_html_path)
    if not _looks_like_html(cockpit_html):
        raise ValueError(f"{cockpit_html_path} does not look like HTML")
    cockpit_refresh_seconds = _meta_refresh_seconds(cockpit_html)
    if not cockpit_refresh_seconds:
        raise ValueError(
            f"{cockpit_html_path} must include a meta refresh for live web viewing"
        )

    operator_surface_path = (
        _repo_root_for_signal(signal_path) / "cockpit/operator_surface.html"
    )
    operator_refresh_seconds = None
    if operator_surface_path.is_file():
        operator_refresh_seconds = _meta_refresh_seconds(
            _load_text(operator_surface_path)
        )
        if not operator_refresh_seconds:
            raise ValueError(
                f"{operator_surface_path} must include a meta refresh when present"
            )

    signal_mtime = signal_path.stat().st_mtime
    cockpit_mtime = cockpit_html_path.stat().st_mtime
    return {
        "schema": WEB_VIEWER_REFRESH_SCHEMA,
        "status": "refresh_ready",
        "source_signal_path": str(signal_path),
        "signal_ts": signal.get("ts"),
        "signal_mtime_utc": _iso_mtime(signal_path),
        "cockpit_html_path": str(cockpit_html_path),
        "cockpit_html_mtime_utc": _iso_mtime(cockpit_html_path),
        "cockpit_refresh_seconds": cockpit_refresh_seconds,
        "cockpit_chart_path": str(cockpit_chart_path),
        "cockpit_chart_mtime_utc": _iso_mtime(cockpit_chart_path),
        "operator_surface_path": str(operator_surface_path),
        "operator_surface_present": operator_surface_path.is_file(),
        "operator_refresh_seconds": operator_refresh_seconds,
        "html_is_newer_than_signal": cockpit_mtime >= signal_mtime,
    }


def build_android_viewer_bundle(signal_path: Path) -> dict:
    required_paths = _cockpit_artifact_paths(signal_path, REQUIRED_COCKPIT_ARTIFACTS)
    if not all(path.is_file() for path in required_paths.values()):
        return {}

    included_paths = dict(required_paths)
    optional_paths = _cockpit_artifact_paths(signal_path, OPTIONAL_COCKPIT_ARTIFACTS)
    for key, path in optional_paths.items():
        if path.is_file():
            included_paths[key] = path

    bundle = {
        "schema": ANDROID_VIEWER_BUNDLE_SCHEMA,
        "source_artifacts": {key: str(path) for key, path in included_paths.items()},
    }
    for key, path in included_paths.items():
        bundle[key] = _load_text(path)
    bundle["cockpit_html"] = _sanitize_html_for_android_viewer(bundle["cockpit_html"])

    if not _looks_like_html(bundle["cockpit_html"]):
        raise ValueError(f"{required_paths['cockpit_html']} does not look like HTML")

    svg_keys = [key for key in included_paths if key.endswith("_svg")]
    for key in svg_keys:
        if not _looks_like_svg(bundle[key]):
            raise ValueError(f"{included_paths[key]} does not look like SVG")
    return bundle


def _write_signal_targets(
    signal: dict,
    android_root: Path,
) -> list[str]:
    written_paths = []
    for relative_path in TARGET_RELATIVE_PATHS:
        target = android_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(signal, indent=2) + "\n", encoding="utf-8")
        written_paths.append(str(target))
    return written_paths


def _write_cockpit_assets(
    viewer_bundle: dict,
    android_root: Path,
) -> list[str]:
    if not viewer_bundle:
        return []

    written_paths = []
    for bundle_key, relative_path in COCKPIT_ASSET_TARGETS.items():
        payload = viewer_bundle.get(bundle_key)
        if not payload:
            continue
        target = android_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        written_paths.append(str(target))
    return written_paths


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
    live_import_path: Path = DEFAULT_LIVE_IMPORT_PATH,
) -> dict:
    signal = _load_json(signal_path)
    _validate_signal(signal)
    viewer_bundle = build_android_viewer_bundle(signal_path)

    written_paths = _write_signal_targets(signal, android_root)
    cockpit_asset_paths = _write_cockpit_assets(viewer_bundle, android_root)

    live_payload = dict(signal)
    if viewer_bundle:
        live_payload[ANDROID_VIEWER_BUNDLE_KEY] = viewer_bundle

    live_import_path.parent.mkdir(parents=True, exist_ok=True)
    live_import_path.write_text(
        json.dumps(live_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    trade_permission = signal["trade_permission"]
    source_freshness = signal.get("source_freshness") or {}
    permission_score_trend = signal.get("permission_score_trend") or {}
    decision_receipt = signal.get("decision_receipt") or {}
    proof = {
        "artifact_type": "sharpedge_android_viewer_export",
        "status": "exported",
        "source_signal_path": str(signal_path),
        "android_root": str(android_root),
        "written_paths": written_paths,
        "cockpit_asset_paths": cockpit_asset_paths,
        "live_import_path": str(live_import_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "signal_ts": signal.get("ts"),
        "symbol": signal.get("symbol"),
        "spot": signal.get("spot"),
        "trade_permission": {
            "trade_gate": trade_permission.get("trade_gate"),
            "trade_permission_score": trade_permission.get("trade_permission_score"),
            "bias": trade_permission.get("bias"),
        },
        "source_freshness": {
            "signal_generated_at": source_freshness.get("signal_generated_at"),
            "price_last_bar_utc": (source_freshness.get("price") or {}).get(
                "last_bar_utc"
            ),
            "options_latest_trade": (source_freshness.get("options") or {}).get(
                "latest_option_trade_time_raw"
            ),
        },
        "permission_score_trend": {
            "current": permission_score_trend.get("current"),
            "delta": permission_score_trend.get("delta"),
            "direction": permission_score_trend.get("direction"),
        },
        "decision_receipt": {
            "gate": decision_receipt.get("gate"),
            "setup": decision_receipt.get("setup"),
            "reachable_today": decision_receipt.get("reachable_today"),
        },
        "android_viewer_bundle": {
            "included": bool(viewer_bundle),
            "schema": viewer_bundle.get("schema", ""),
            "source_artifacts": viewer_bundle.get("source_artifacts", {}),
        },
        "note": "Packaged sample assets were updated. Live import carries the advanced cockpit bundle when cockpit.html and cockpit_chart.svg exist, plus context SVG companions when available.",
    }
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main(argv: list[str]) -> int:
    signal_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_SIGNAL_PATH
    android_root = Path(argv[2]) if len(argv) > 2 else DEFAULT_ANDROID_ROOT
    proof_path = Path(argv[3]) if len(argv) > 3 else DEFAULT_PROOF_PATH
    live_import_path = Path(argv[4]) if len(argv) > 4 else DEFAULT_LIVE_IMPORT_PATH
    proof = export_signal(signal_path, android_root, proof_path, live_import_path)
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
