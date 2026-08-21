from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from scripts.alpha_swarm import surface_comparison_agent as agent


def _manifest(session_date="2026-08-11"):
    slots = []
    for symbol in agent.EXPECTED_SYMBOLS:
        slots.append(
            {
                "slot_id": f"{session_date}-1045-{symbol}",
                "symbol": symbol,
                "session_date": session_date,
                "prediction_ts": f"{session_date}T10:45:00-04:00",
                "exit_ts": f"{session_date}T15:45:00-04:00",
                "label_available_ts": f"{session_date}T16:15:00-04:00",
            }
        )
    return {"slots": slots}


def _publication(symbol, session_date="2026-08-11", published_at=None):
    published_at = published_at or f"{session_date}T14:45:00+00:00"
    return {
        "schema": "sharpedge.alpha_swarm.hypothesis_publication.v1",
        "slot_id": f"{session_date}-1045-{symbol}",
        "session_date": session_date,
        "symbol": symbol,
        "published_at": published_at,
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "candidate": {
            "slot_id": f"{session_date}-1045-{symbol}",
            "symbol": symbol,
            "decision": "stand_down",
            "prediction_ts": f"{session_date}T10:45:00-04:00",
            "published_at": f"{session_date}T10:45:00-04:00",
            "risk_cap_dollars": 0.0,
            "feature_values": {
                "vs_vwap_pct": -0.005,
                "momentum_15m_pct": -0.026,
                "volume_ratio": 0.705,
            },
        },
    }


def _publications(tmp_path):
    selected = {}
    for symbol in agent.EXPECTED_SYMBOLS:
        path = tmp_path / f"{symbol}.json"
        payload = _publication(symbol)
        path.write_text(json.dumps(payload), encoding="utf-8")
        selected[symbol] = {
            "payload": payload,
            "path": str(path),
            "sha256": agent._file_sha256(path),
            "_published": datetime(2026, 8, 11, 14, 45, tzinfo=UTC),
        }
    return selected


def _signal():
    return {
        "schema": "sharpedge.signal.v1",
        "ts": "2026-08-11T14:41:00",
        "symbol": "SPY",
        "spot": 769.87,
        "vwap": 772.21,
        "vs_vwap": -0.303,
        "mom15": -0.092,
        "vol_mult": 1.15,
        "gamma_regime": "negative",
        "pin": 770,
        "max_pain": 771,
        "price_authority": {"price_feed_stale": False},
        "trade_permission": {
            "trade_gate": "CAUTION",
            "trade_permission_score": 58,
            "execution_permission_score": 58,
            "setup_conviction": {
                "setup_gate": "WATCH",
                "setup_tag": "DOWNSIDE EXHAUSTION",
                "bias": "watch for reversal UP (calls)",
            },
        },
    }


def test_report_compares_spy_without_rewriting_paper_horizon(tmp_path):
    report = agent.build_report(
        signal=_signal(),
        manifest=_manifest(),
        publications=_publications(tmp_path),
        session_date="2026-08-11",
        worker={
            "pid": 999999,
            "heartbeat_at": "2026-08-11T18:40:00+00:00",
            "paper_only": True,
            "execution_permitted": False,
            "events": {
                "x": {"status": "completed"},
                "y": {"status": "missed"},
            },
        },
        pilot_root=tmp_path,
        generated_at=datetime(2026, 8, 11, 18, 42, tzinfo=UTC),
    )

    assert report["schema"] == agent.SCHEMA
    assert len(report["paper_surface"]["symbols"]) == 6
    assert report["paper_surface"]["available_symbol_count"] == 6
    assert report["spy_comparison"]["action_alignment"] == "both_no_action"
    assert report["spy_comparison"]["direction_change"] == "mixed_to_bearish"
    assert report["live_sharpedge_surface"]["location_lean"] == "bearish"
    assert report["paper_surface"]["current_session_event_counts"] == {"pending": 48}
    assert report["temporal_alignment"]["state"] == "later_live_snapshot"
    assert report["temporal_alignment"]["same_decision_window"] is False
    assert report["safety"] == {
        "paper_only": True,
        "authoritative": False,
        "execution_permitted": False,
        "can_mutate_paper_artifacts": False,
        "can_override_approval_decision": False,
        "aggregate_score_computed": False,
        "hindsight_use": "comparison_only",
    }
    assert "aggregate_score" not in report


def test_report_marks_stale_live_signal(tmp_path):
    report = agent.build_report(
        signal=_signal(),
        manifest=_manifest(),
        publications=_publications(tmp_path),
        session_date="2026-08-11",
        worker={},
        pilot_root=tmp_path,
        generated_at=datetime(2026, 8, 11, 20, 0, tzinfo=UTC),
    )
    assert report["temporal_alignment"]["state"] == "stale_live_signal"


def test_discovery_uses_latest_coherent_session_and_rejects_authority(tmp_path):
    warnings = []
    old = tmp_path / "2026-08-10" / "2026-08-10-1045-SPY"
    old.mkdir(parents=True)
    (old / "phase3_hypothesis.json").write_text(
        json.dumps(_publication("SPY", "2026-08-10")), encoding="utf-8"
    )
    latest = tmp_path / "2026-08-11" / "2026-08-11-1045-SPY"
    latest.mkdir(parents=True)
    (latest / "phase3_hypothesis.json").write_text(
        json.dumps(_publication("SPY")), encoding="utf-8"
    )
    bad = tmp_path / "2026-08-11" / "2026-08-11-1045-QQQ"
    bad.mkdir(parents=True)
    payload = _publication("QQQ")
    payload["authoritative"] = True
    (bad / "phase3_hypothesis.json").write_text(json.dumps(payload), encoding="utf-8")

    session_date, selected = agent.discover_latest_publications(tmp_path, warnings)

    assert session_date == "2026-08-11"
    assert set(selected) == {"SPY"}
    assert any("ignored invalid" in warning for warning in warnings)


def test_markdown_and_html_repeat_safety_boundary_and_escape(tmp_path):
    report = agent.build_report(
        signal=_signal(),
        manifest=_manifest(),
        publications=_publications(tmp_path),
        session_date="2026-08-11",
        worker={},
        pilot_root=tmp_path,
        generated_at=datetime(2026, 8, 11, 18, 42, tzinfo=UTC),
    )
    report["headline"] = "safe <script>alert(1)</script>"
    markdown = agent.render_markdown(report)
    html = agent.render_html(report, markdown)

    assert "cannot authorize execution" in markdown
    assert "No candidate mutation" in markdown
    assert "Current-session event counts" in markdown
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "READ-ONLY • PAPER-ONLY • NO EXECUTION" in html


def test_path_builder_writes_three_outputs_without_mutating_inputs(tmp_path):
    signal_path = tmp_path / "signal.json"
    manifest_path = tmp_path / "manifest.json"
    pilot_root = tmp_path / "pilot"
    signal_path.write_text(json.dumps(_signal()), encoding="utf-8")
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    for symbol in agent.EXPECTED_SYMBOLS:
        root = pilot_root / "2026-08-11" / f"2026-08-11-1045-{symbol}"
        root.mkdir(parents=True)
        (root / "phase3_hypothesis.json").write_text(
            json.dumps(_publication(symbol)), encoding="utf-8"
        )
    (pilot_root / "worker_state.json").write_text(
        json.dumps(
            {
                "pid": 0,
                "paper_only": True,
                "execution_permitted": False,
                "events": {},
            }
        ),
        encoding="utf-8",
    )
    source_before = signal_path.read_bytes()
    args = argparse.Namespace(
        signal=signal_path,
        manifest=manifest_path,
        pilot_root=pilot_root,
        output_json=tmp_path / "out" / "latest.json",
        output_markdown=tmp_path / "out" / "latest.md",
        output_html=tmp_path / "out" / "latest.html",
    )

    report = agent.build_from_paths(args)

    assert signal_path.read_bytes() == source_before
    assert args.output_json.exists()
    assert args.output_markdown.exists()
    assert args.output_html.exists()
    assert json.loads(args.output_json.read_text())["schema"] == agent.SCHEMA
    assert report["safety"]["aggregate_score_computed"] is False
