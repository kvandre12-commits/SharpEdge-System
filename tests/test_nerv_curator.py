from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.agents import nerv_curator as curator
from scripts.agents.nerv_curator import build_packet, render_text


def test_nerv_curator_builds_operator_read(tmp_path: Path) -> None:
    board = tmp_path / "nerv_liquidity_board.json"
    board.write_text(
        json.dumps(
            {
                "schema": "sharpedge.nerv_liquidity_board.v1",
                "summary": {"data_mode": "test_delayed"},
                "contracts": [
                    {
                        "expiration": "2026-07-29",
                        "option_type": "call",
                        "strike": 750.0,
                        "contract_symbol": "SPY260729C00750000",
                        "midpoint": 0.25,
                        "bid": 0.24,
                        "ask": 0.26,
                        "volume": 15000,
                        "open_interest": 2400,
                        "width_pct": 0.04,
                        "manual_validation_priority": "high",
                        "rejection_flags": "none",
                    },
                    {
                        "expiration": "2026-07-28",
                        "option_type": "put",
                        "strike": 738.0,
                        "contract_symbol": "SPY260728P00738000",
                        "midpoint": 1.90,
                        "bid": 1.89,
                        "ask": 1.91,
                        "volume": 68000,
                        "open_interest": 3700,
                        "width_pct": 0.01,
                        "manual_validation_priority": "high",
                        "rejection_flags": "none",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    iv_heat = tmp_path / "spy_iv_heat_harvest.json"
    iv_heat.write_text(
        json.dumps(
            {
                "schema": "sharpedge.iv_heat_harvest.v1",
                "symbol": "SPY",
                "underlying_price": 739.0,
                "target_strike": 750.0,
                "overall_heat_label": "hot",
                "median_iv_rv13_ratio": 1.44,
                "realized_vol": {"rv13_pct": 10.5},
                "nearest_event": "FOMC",
                "days_to_nearest_event": 2,
                "expiry_reads": [
                    {
                        "expiration": "2026-07-29",
                        "dte_calendar": 2,
                        "atm_iv_pct": 14.5,
                        "iv_rv13_ratio": 1.38,
                        "heat_label": "hot",
                        "call_750_mid": 0.25,
                        "call_750_iv_pct": 11.2,
                        "call_750_open_interest": 2400,
                        "harvest_window": "event_crush_window",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "spot": 739.0,
                "gamma_regime": "negative",
                "historical_refill_context": {
                    "active_refill_stack": {
                        "stack_label": "double_dip_stack",
                        "active_count": 2,
                        "nearest_target": 747.41,
                        "highest_target": 757.09,
                    }
                },
                "event_radar": {"headline": "FOMC decision in 2d"},
            }
        ),
        encoding="utf-8",
    )

    packet = build_packet(board_path=board, iv_heat_path=iv_heat, signal_path=signal)
    text = render_text(packet)

    assert packet.schema == "sharpedge.nerv_curator.v1"
    assert packet.stance == "wait_for_acceptance_or_harvest"
    assert "IV/RV13 1.44x" in packet.headline
    assert "747.41" in packet.headline
    assert any(contract.role == "target-call" for contract in packet.focus_contracts)
    assert "liquid call neighborhood" in packet.hey_guy_summary["plain_english"]
    assert packet.hey_guy_summary["call_flow"]
    assert packet.hey_guy_summary["put_flow"]
    assert "Crosswired tape" in packet.hey_guy_summary["flow_balance"]
    assert (
        "quote-weighted focus-line pressure" in packet.hey_guy_summary["flow_balance"]
    )
    assert packet.hey_guy_summary["bias_alignment"] == "crosswired"
    assert packet.hey_guy_summary["put_pressure_pct"] > 0
    assert (
        packet.hey_guy_summary["put_pressure_pct"]
        > packet.hey_guy_summary["call_pressure_pct"]
    )
    assert packet.hey_guy_summary["dominant_side"] == "put"
    assert "2026-07-29 CALL 750" in packet.hey_guy_summary["liquidity_spot"]
    assert "approval_decision remains the only authority" in text


def test_curator_flips_to_downside_language_when_setup_bias_is_bearish(
    tmp_path: Path,
) -> None:
    board = tmp_path / "nerv_liquidity_board.json"
    board.write_text(
        json.dumps(
            {
                "contracts": [
                    {
                        "expiration": "2026-07-29",
                        "option_type": "put",
                        "strike": 738.0,
                        "contract_symbol": "SPY260729P00738000",
                        "midpoint": 1.25,
                        "bid": 1.2,
                        "ask": 1.3,
                        "volume": 12000,
                        "open_interest": 2200,
                        "width_pct": 0.08,
                        "manual_validation_priority": "high",
                        "rejection_flags": "none",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    iv_heat = tmp_path / "spy_iv_heat_harvest.json"
    iv_heat.write_text(
        json.dumps(
            {
                "symbol": "SPY",
                "underlying_price": 739.0,
                "target_strike": 750.0,
                "overall_heat_label": "cool",
                "median_iv_rv13_ratio": 0.8,
                "expiry_reads": [],
            }
        ),
        encoding="utf-8",
    )
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "spot": 739.0,
                "setup_tag": "watch for reversal DOWN (puts)",
                "entry_setup_bias": "watch for reversal DOWN (puts)",
                "historical_refill_context": {
                    "active_refill_stack": {
                        "nearest_target": 737.5,
                        "highest_target": 745.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    packet = build_packet(board_path=board, iv_heat_path=iv_heat, signal_path=signal)

    assert "downside hedge lines" in packet.headline
    assert "liquid put neighborhood" in packet.hey_guy_summary["plain_english"]
    assert packet.hey_guy_summary["put_flow"]
    assert packet.hey_guy_summary["call_flow"] == []
    assert packet.hey_guy_summary["bias_alignment"] == "aligned"
    assert (
        packet.hey_guy_summary["put_pressure_pct"]
        > packet.hey_guy_summary["call_pressure_pct"]
    )
    assert packet.hey_guy_summary["dominant_side"] == "put"
    assert "PUT-led tape" in packet.hey_guy_summary["flow_balance"]
    assert (
        "quote-weighted focus-line pressure" in packet.hey_guy_summary["flow_balance"]
    )
    assert "PUT 738" in packet.hey_guy_summary["liquidity_spot"]
    assert packet.watch_next[0].startswith("Track rejection/failure")


def test_dead_quotes_get_downweighted_in_flow_read(tmp_path: Path) -> None:
    board = tmp_path / "nerv_liquidity_board.json"
    board.write_text(
        json.dumps(
            {
                "contracts": [
                    {
                        "expiration": "2026-07-29",
                        "option_type": "put",
                        "strike": 738.0,
                        "contract_symbol": "SPY260729P00738000",
                        "midpoint": 1.25,
                        "bid": 1.2,
                        "ask": 1.3,
                        "volume": 4000,
                        "open_interest": 1000,
                        "width_pct": 0.05,
                        "manual_validation_priority": "high",
                        "rejection_flags": "none",
                    },
                    {
                        "expiration": "2026-07-29",
                        "option_type": "call",
                        "strike": 741.0,
                        "contract_symbol": "SPY260729C00741000",
                        "midpoint": None,
                        "bid": 0.0,
                        "ask": 0.0,
                        "volume": 50000,
                        "open_interest": 10000,
                        "width_pct": None,
                        "manual_validation_priority": "reject",
                        "rejection_flags": "zero_or_tiny_market;missing_midpoint",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    iv_heat = tmp_path / "spy_iv_heat_harvest.json"
    iv_heat.write_text(
        json.dumps(
            {
                "symbol": "SPY",
                "underlying_price": 739.0,
                "target_strike": 750.0,
                "overall_heat_label": "cool",
                "median_iv_rv13_ratio": 0.8,
                "expiry_reads": [],
            }
        ),
        encoding="utf-8",
    )
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "spot": 739.0,
                "setup_tag": "watch for reversal DOWN (puts)",
                "entry_setup_bias": "watch for reversal DOWN (puts)",
            }
        ),
        encoding="utf-8",
    )

    packet = build_packet(board_path=board, iv_heat_path=iv_heat, signal_path=signal)

    assert packet.hey_guy_summary["bias_alignment"] == "aligned"
    assert (
        packet.hey_guy_summary["put_pressure_pct"]
        > packet.hey_guy_summary["call_pressure_pct"]
    )
    assert "PUT-led tape" in packet.hey_guy_summary["flow_balance"]
    assert "bad quotes are muting it" in packet.hey_guy_summary["quote_quality_context"]


def test_resolve_default_board_prefers_fresh_cockpit_board(
    monkeypatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "nerv_spy_month" / "nerv_liquidity_board.json"
    preferred = tmp_path / "nerv_cockpit_standard" / "nerv_liquidity_board.json"
    fallback.parent.mkdir(parents=True)
    preferred.parent.mkdir(parents=True)
    fallback.write_text("{}", encoding="utf-8")
    preferred.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        curator,
        "DEFAULT_BOARD_CANDIDATES",
        (preferred, fallback),
    )

    assert curator.resolve_default_board_path() == preferred
