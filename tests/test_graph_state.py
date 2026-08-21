from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from trade_permission import score_trade_permission


def _bull_bars():
    bars = []
    price = 100.0
    for minute in range(45):
        open_ = price
        close = price + 0.08
        high = close + 0.04
        low = open_ - 0.02
        volume = 1000 + minute * 12
        bars.append((minute, open_, high, low, close, volume))
        price = close
    return bars


def _pa(bars, **overrides):
    closes = [bar[4] for bar in bars]
    pa = {
        "spot": closes[-1],
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "balance_high": closes[-2],
        "balance_low": closes[-8],
        "position_in_balance": 1.0,
        "balance_state": "above",
        "balance_label": "TOP",
        "balance_width_pct": 0.42,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
        "dominant_balance_name": "recent_balance",
        "dominant_balance_reason": "mid-session: active recent box matters most",
        "session_position_in_range": 0.92,
        "rng_pos": 92.0,
        "day_chg": 1.2,
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
    }
    pa.update(overrides)
    return pa


def test_graph_canon_is_attached_to_all_nine_core_spine_rows():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    features = card["execution_hierarchy"]["core_spine"]["features"]
    assert card["graph_state"]["schema"] == "sharpedge.graph_state.v1"
    assert card["graph_state"]["authority_role"] == "operator_visual_canon"
    assert card["bucket_conditioned_spine"]["graph_state"] == card["graph_state"]
    assert len(features) == 9
    for row in features:
        assert row["graph_agreement"]["schema"] == "sharpedge.graph_agreement.v1"
        assert row["graph_agreement"]["authority_role"] == "operator_visual_canon"


def test_graph_canon_marks_opposing_vertical_as_conflict():
    bars = _bull_bars()
    pa = _pa(bars, vs_vwap=-0.35, mom15=-0.4, rng_pos=25.0)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}

    card = score_trade_permission(bars, pa, levels, [], {"atm_iv": 0.18}, {}, {})

    assert card["graph_state"]["graph_bias"] == "PUTS"
    conflicts = [
        row
        for row in card["execution_hierarchy"]["core_spine"]["features"]
        if row["graph_agreement"]["relation"] == "conflict"
    ]
    assert conflicts
    assert all(
        row["graph_agreement"]["action"] == "demote_or_explain_before_trusting"
        for row in conflicts
    )
