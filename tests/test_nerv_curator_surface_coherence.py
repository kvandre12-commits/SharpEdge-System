"""Coherence guards for the operator-facing NERV curator surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.agents.nerv_curator import build_packet


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_curator_cli_runs_directly_outside_repo_root(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "agents" / "nerv_curator.py"
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs" / "nerv_curator.json").exists()
    assert (tmp_path / "outputs" / "nerv_curator.txt").exists()


def test_curator_prefers_live_cockpit_spot_and_avoids_na_guidance(
    tmp_path: Path,
) -> None:
    board = _write_json(
        tmp_path / "board.json",
        {"schema": "sharpedge.nerv_liquidity_board.v1", "contracts": []},
    )
    iv_heat = _write_json(
        tmp_path / "iv.json",
        {
            "symbol": "SPY",
            "underlying_price": 778.93,
            "target_strike": 750.0,
            "overall_heat_label": "hot",
            "median_iv_rv13_ratio": 1.25,
            "expiry_reads": [],
        },
    )
    signal = _write_json(
        tmp_path / "signal.json",
        {
            "ts": "2026-08-20T14:28:00+00:00",
            "spot": 766.12,
            "entry_setup_bias": "CALLS",
        },
    )

    packet = build_packet(
        board_path=board,
        iv_heat_path=iv_heat,
        signal_path=signal,
    )

    summary = packet.hey_guy_summary
    guidance = [packet.headline, *summary["confirms"], *summary["invalidates"]]
    assert packet.underlying_price == 766.12
    assert summary["status"] == "degraded"
    assert summary["blockers"] == [
        "no_focus_contracts",
        "quote_quality_unavailable",
    ]
    assert all("n/a" not in line.lower() for line in guidance)
    assert any(
        "displayed spot uses the live cockpit" in line for line in packet.warnings
    )
