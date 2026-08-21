from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from regime_nerv_panel import (
    BoardSource,
    render_hey_guy_html,
    render_panel_html,
    render_tabs_html,
    write_surfaces,
)


def test_render_panel_includes_structure_taxonomy(tmp_path: Path) -> None:
    board = tmp_path / "ctc_nerv_trade_desk.json"
    board.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-26T22:30:00+00:00",
                "summary": {
                    "row_count": 1,
                    "manual_validate_count": 0,
                    "states": {"refresh_quote_required": 1},
                },
                "rows": [
                    {
                        "rank": 1,
                        "ticker": "WMT",
                        "company": "Walmart Inc.",
                        "desk_state": "refresh_quote_required",
                        "structure_family": "branch_defined_debit_spread",
                        "structure_complexity": "branch_pending",
                        "nerv_score": 49.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_panel_html([BoardSource("Cartridge: WMT", board)])

    assert "Cartridge: WMT" in html
    assert "branch_defined_debit_spread" in html
    assert "branch_pending" in html
    assert "Walmart Inc." in html


def test_render_panel_includes_raw_nerv_liquidity_contracts(tmp_path: Path) -> None:
    board = tmp_path / "nerv_liquidity_board.json"
    board.write_text(
        json.dumps(
            {
                "schema": "sharpedge.nerv_liquidity_board.v1",
                "generated_at": "2026-07-27T15:45:00+00:00",
                "contracts": [
                    {
                        "underlying": "SPY",
                        "expiration": "2026-07-27",
                        "option_type": "call",
                        "strike": 744.0,
                        "contract_symbol": "SPY260727C00744000",
                        "manual_validation_priority": "reject",
                        "rejection_flags": "zero_or_tiny_market;missing_midpoint;stale_quote",
                        "volume": 51012,
                        "open_interest": 6344,
                        "bid": 0.0,
                        "ask": 0.0,
                        "midpoint": None,
                        "nerv_score": 49.0,
                        "fetch_timestamp": "2026-07-27T15:45:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_panel_html([BoardSource("Standard NERV: SPY/WMT", board)])

    assert "Standard NERV: SPY/WMT" in html
    assert "Contracts: 1" in html
    assert "SPY260727C00744000" in html
    assert "priority reject: 1" in html
    assert "zero_or_tiny_market;missing_midpoint;stale_quote" in html


def test_render_panel_includes_nerv_curator_card(tmp_path: Path) -> None:
    curator = tmp_path / "nerv_curator.json"
    curator.write_text(
        json.dumps(
            {
                "schema": "sharpedge.nerv_curator.v1",
                "generated_at_utc": "2026-07-28T01:00:00+00:00",
                "headline": "Curate NERV around 750 calls.",
                "stance": "wait_for_acceptance_or_harvest",
                "target_strike": 750.0,
                "underlying_price": 739.0,
                "iv_heat": {"median_iv_rv13_ratio": 1.44},
                "cockpit_context": {"spot": 739.0},
                "watch_next": ["Wait for reclaim acceptance."],
                "warnings": ["approval_decision remains authority."],
                "hey_guy_summary": {
                    "title": "Hey guy — SharpEdge/NERV read",
                    "one_liner": "Curate NERV around 750 calls.",
                    "plain_english": "SharpEdge screams calls; NERV found the liquid 750C neighborhood.",
                    "liquidity_spot": "2026-07-29 CALL 750 mid 0.25.",
                    "near_money_tape": ["2026-07-28 CALL 740 mid 1.90."],
                    "confirms": ["Accept above 747.41."],
                    "invalidates": ["Fail below 747.41."],
                    "operator_note": "Research-only.",
                    "stance": "wait_for_acceptance_or_harvest",
                },
                "focus_contracts": [
                    {
                        "role": "target-call",
                        "expiration": "2026-07-29",
                        "contract": "SPY260729C00750000",
                        "option_type": "call",
                        "strike": 750.0,
                        "mid": 0.25,
                        "bid": 0.24,
                        "ask": 0.26,
                        "volume": 15000,
                        "open_interest": 2400,
                        "priority": "high",
                        "flags": "none",
                        "reason": "near thesis strike",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_panel_html([BoardSource("NERV curator", curator)])

    assert "NERV curator" in html
    assert "Curate NERV around 750 calls." in html
    assert "IV/RV13: 1.44x" in html
    assert "SPY260729C00750000" in html
    assert "Wait for reclaim acceptance." in html


def test_render_hey_guy_html_reads_curator(monkeypatch, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "nerv_curator.json").write_text(
        json.dumps(
            {
                "schema": "sharpedge.nerv_curator.v1",
                "generated_at_utc": "2026-07-28T01:00:00+00:00",
                "headline": "Fallback headline.",
                "stance": "wait_for_acceptance_or_harvest",
                "target_strike": 750.0,
                "underlying_price": 739.0,
                "hey_guy_summary": {
                    "title": "Hey guy — SharpEdge/NERV read",
                    "one_liner": "SharpEdge calls, NERV liquidity found.",
                    "plain_english": "Wait for acceptance, then check 750C liquidity.",
                    "liquidity_spot": "2026-07-29 CALL 750 mid 0.25.",
                    "flow_balance": "CALL-led tape.",
                    "bias_alignment": "aligned",
                    "quote_quality_context": "Both sides have usable enough quotes.",
                    "put_pressure_score": 1100.0,
                    "call_pressure_score": 1900.0,
                    "put_pressure_pct": 37,
                    "call_pressure_pct": 63,
                    "dominant_side": "call",
                    "put_flow": ["2026-07-28 PUT 735 mid 1.10."],
                    "call_flow": ["2026-07-28 CALL 740 mid 1.90."],
                    "near_money_tape": ["2026-07-28 CALL 740 mid 1.90."],
                    "confirms": ["Accept above 747.41."],
                    "invalidates": ["Fail below 747.41."],
                    "operator_note": "Research-only.",
                    "stance": "wait_for_acceptance_or_harvest",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("regime_nerv_panel.ROOT", tmp_path)

    html = render_hey_guy_html(refresh_seconds=10)

    assert "Hey guy" in html
    assert "<main>" in html
    assert "Data status: degraded" in html
    assert "SharpEdge calls, NERV liquidity found." in html
    assert "2026-07-29 CALL 750 mid 0.25." in html
    assert "Pressure split" in html
    assert "lead: call" in html
    assert "2026-07-28 PUT 735 mid 1.10." in html
    assert "2026-07-28 CALL 740 mid 1.90." in html
    assert "Accept above 747.41." in html


def test_render_tabs_exposes_accessible_tab_state_and_panels() -> None:
    html = render_tabs_html(refresh_seconds=10)

    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 4
    assert html.count('role="tabpanel"') == 4
    assert 'aria-controls="decision-panel"' in html
    assert 'src="operator_decision_card.html"' in html
    assert 'aria-controls="graph-panel"' in html
    assert 'aria-labelledby="graph-tab"' in html
    assert "aria-selected" in html
    assert "ArrowRight" in html
    assert "<main>" in html
    assert 'http-equiv="refresh"' not in html


def test_write_surfaces_creates_panel_split_tabs_and_hey_guy(tmp_path: Path) -> None:
    paths = write_surfaces(cockpit_dir=tmp_path, refresh_seconds=10, sources=[])

    panel = Path(paths["panel"])
    split = Path(paths["split"])
    tabs = Path(paths["tabs"])
    assert panel.exists()
    assert split.exists()
    hey_guy = Path(paths["hey_guy"])
    assert tabs.exists()
    assert hey_guy.exists()
    assert "No Regime/NERV board found" in panel.read_text(encoding="utf-8")
    assert "cockpit.html" in split.read_text(encoding="utf-8")
    tabs_text = tabs.read_text(encoding="utf-8")
    assert "Graph + Read" in tabs_text
    assert "Spine" in tabs_text
    assert "Options" in tabs_text
    assert "operator_surface.html" in tabs_text
    assert "regime_nerv_panel.html" in tabs_text
