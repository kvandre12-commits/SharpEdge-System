from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phone_companion"))

from export_signal_to_android_viewer import (  # noqa: E402
    ANDROID_VIEWER_BUNDLE_KEY,
    export_signal,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_export_signal_to_android_viewer_copies_trade_gate_contract(tmp_path):
    signal_path = tmp_path / "outputs/signal.json"
    android_root = tmp_path / "SharpEdge-Android"
    proof_path = tmp_path / "phone_companion/views/trading/android_viewer_export.json"
    live_import_path = (
        tmp_path / "phone_companion/views/trading/sharpedge_android_live_import.json"
    )
    cockpit_html = "<!DOCTYPE html><html><body><h1>SharpEdge Cockpit</h1></body></html>"
    cockpit_chart = (
        "<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>"
    )
    cockpit_weekly = (
        "<svg xmlns='http://www.w3.org/2000/svg'><text>weekly</text></svg>"
    )
    cockpit_monthly = (
        "<svg xmlns='http://www.w3.org/2000/svg'><text>monthly</text></svg>"
    )
    signal = {
        "schema": "sharpedge.signal.v1",
        "ts": "2026-06-18T15:40:58",
        "symbol": "SPY",
        "spot": 746.94,
        "source_freshness": {
            "signal_generated_at": "2026-06-18T15:40:58",
            "price": {"last_bar_utc": "2026-06-18T15:40:00+00:00"},
            "options": {"latest_option_trade_time_raw": "2026-06-18T15:39:10"},
        },
        "permission_score_trend": {
            "current": 67,
            "delta": 2,
            "direction": "strengthening",
        },
        "decision_receipt": {
            "gate": "CAUTION",
            "setup": "STICKY DAY",
            "reachable_today": {"label": "VWAP", "price": 746.5},
        },
        "trade_permission": {
            "trade_gate": "CAUTION",
            "trade_permission_score": 67,
            "bias": "NEUTRAL",
            "supporting_reasons": ["location: at VWAP"],
            "warning_reasons": ["volume: thin"],
            "scores": {},
        },
    }
    _write_json(signal_path, signal)
    _write_text(tmp_path / "cockpit/cockpit.html", cockpit_html)
    _write_text(tmp_path / "cockpit/cockpit_chart.svg", cockpit_chart)
    _write_text(tmp_path / "cockpit/cockpit_weekly_context.svg", cockpit_weekly)
    _write_text(tmp_path / "cockpit/cockpit_monthly_context.svg", cockpit_monthly)

    proof = export_signal(signal_path, android_root, proof_path, live_import_path)

    signal_asset = android_root / "app/src/main/assets/sample_signal.json"
    contract = android_root / "app_contracts/sharpedge.signal.v1.sample.json"
    html_asset = android_root / "app/src/main/assets/sample_cockpit.html"
    chart_asset = android_root / "app/src/main/assets/sample_cockpit_chart.svg"
    weekly_asset = android_root / "app/src/main/assets/sample_cockpit_weekly_context.svg"
    monthly_asset = android_root / "app/src/main/assets/sample_cockpit_monthly_context.svg"
    live_import = json.loads(live_import_path.read_text())
    assert proof["status"] == "exported"
    assert proof["trade_permission"] == {
        "trade_gate": "CAUTION",
        "trade_permission_score": 67,
        "bias": "NEUTRAL",
    }
    assert json.loads(signal_asset.read_text()) == signal
    assert json.loads(contract.read_text()) == signal
    assert html_asset.read_text() == cockpit_html
    assert chart_asset.read_text() == cockpit_chart
    assert weekly_asset.read_text() == cockpit_weekly
    assert monthly_asset.read_text() == cockpit_monthly
    assert live_import["schema"] == "sharpedge.signal.v1"
    assert live_import[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_html"] == cockpit_html
    assert live_import[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_chart_svg"] == cockpit_chart
    assert live_import[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_weekly_context_svg"] == cockpit_weekly
    assert live_import[ANDROID_VIEWER_BUNDLE_KEY]["cockpit_monthly_context_svg"] == cockpit_monthly
    proof_json = json.loads(proof_path.read_text())
    assert proof_json["status"] == "exported"
    assert proof_json["live_import_path"] == str(live_import_path)
    assert proof_json["android_viewer_bundle"]["included"] is True
    assert (
        proof_json["source_freshness"]["price_last_bar_utc"]
        == "2026-06-18T15:40:00+00:00"
    )
    assert proof_json["permission_score_trend"]["direction"] == "strengthening"
    assert proof_json["decision_receipt"]["setup"] == "STICKY DAY"
