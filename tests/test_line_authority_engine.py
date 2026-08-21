from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from line_authority_engine import build_line_authority
from trade_permission import score_trade_permission


def _pa(spot: float = 100.24, **overrides):
    data = {
        "spot": spot,
        "day_open": 100.0,
        "hi": 101.0,
        "lo": 99.0,
        "vwap": 100.0,
        "vs_vwap": 0.24,
        "mom15": 0.12,
        "vol_mult": 1.4,
        "rng_pos": 72.0,
        "day_chg": 0.24,
        "balance_high": 100.40,
        "balance_low": 99.80,
        "balance_width_pct": 0.6,
        "position_in_balance": 0.8,
        "balance_state": "inside",
        "balance_label": "MIDDLE",
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "recent_balance",
        "dominant_balance_reason": "test",
        "balance_confluence": {},
        "balance_disagreement": {},
        "dominant_balance_flip": {},
    }
    data.update(overrides)
    return data


def test_line_authority_detects_vwap_reclaim():
    bars = [
        (0, 100.05, 100.08, 99.92, 99.96, 1000),
        (1, 99.96, 100.02, 99.82, 99.90, 1200),
        (2, 99.90, 100.20, 99.88, 100.12, 1400),
        (3, 100.12, 100.30, 100.05, 100.24, 1500),
    ]

    packet = build_line_authority(bars, _pa(), {"PDH": 101.0, "PDL": 99.0})
    vwap = next(line for line in packet["lines"] if line["name"] == "VWAP")

    assert packet["schema"] == "sharpedge.line_authority.v1"
    assert packet["weighted_in_permission"] is False
    assert vwap["event"] == "reclaimed"
    assert vwap["bias"] == "CALLS"
    assert vwap["score"] >= 70
    assert packet["summary"]["bias"] == "CALLS"


def test_line_authority_detects_pdh_rejection():
    bars = [
        (0, 99.80, 100.10, 99.75, 99.95, 1000),
        (1, 99.95, 100.32, 99.90, 100.18, 1300),
        (2, 100.18, 100.35, 99.88, 99.94, 1500),
        (3, 99.94, 100.02, 99.70, 99.80, 1600),
    ]

    packet = build_line_authority(bars, _pa(spot=99.80, vwap=99.95), {"PDH": 100.0})
    pdh = next(line for line in packet["lines"] if line["name"] == "PDH")

    assert pdh["event"] == "rejected"
    assert pdh["bias"] == "PUTS"
    assert packet["summary"]["bias"] == "PUTS"


def test_line_authority_adds_balance_and_midpoint_rails():
    bars = [
        (0, 100.00, 100.20, 99.90, 100.08, 1000),
        (1, 100.08, 100.18, 100.00, 100.10, 1000),
        (2, 100.10, 100.16, 100.02, 100.12, 1000),
    ]

    packet = build_line_authority(bars, _pa(spot=100.12), {})
    names = {line["name"] for line in packet["lines"]}

    assert {"VWAP", "BALANCE_HIGH", "BALANCE_LOW", "BALANCE_MID", "DAY_MID"} <= names
    midpoint = next(line for line in packet["lines"] if line["name"] == "BALANCE_MID")
    assert midpoint["role"] == "channel_midpoint"


def test_trade_permission_exposes_line_authority_as_advisory_surface_only():
    bars = [
        (0, 100.05, 100.08, 99.92, 99.96, 1000),
        (1, 99.96, 100.02, 99.82, 99.90, 1200),
        (2, 99.90, 100.20, 99.88, 100.12, 1400),
        (3, 100.12, 100.30, 100.05, 100.24, 1500),
        (4, 100.24, 100.35, 100.18, 100.30, 1600),
        (5, 100.30, 100.42, 100.24, 100.38, 1700),
    ]

    card = score_trade_permission(
        bars, _pa(spot=100.38), {"PDH": 101.0}, [], {}, {}, {}
    )
    advisory = card["execution_hierarchy"]["advisory_surfaces"]
    line_rows = [row for row in advisory if row["name"] == "line_authority_score"]

    assert card["line_authority"]["schema"] == "sharpedge.line_authority.v1"
    assert card["line_authority"]["weighted_in_permission"] is False
    assert line_rows
    assert line_rows[0]["weight"] == 0.0
