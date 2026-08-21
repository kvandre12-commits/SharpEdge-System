from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from make_cockpit import synthesize


def _pa(rng_pos: float, vs_vwap: float = 0.2) -> dict:
    return {
        "spot": 100.0,
        "vwap": 99.8 if vs_vwap >= 0 else 100.2,
        "vs_vwap": vs_vwap,
        "rng_pos": rng_pos,
        "balance_state": "inside",
        "balance_reference": "recent_20_bar",
        "position_in_balance": 0.5,
        "balance_low": 99.5,
        "balance_high": 100.5,
        "mom15": 0.06 if vs_vwap >= 0 else -0.06,
        "vol_mult": 1.1,
        "volume_profile": {
            "confirmation": "mixed",
            "reason": "participation is mixed",
        },
    }


def _op() -> dict:
    return {
        "call_wall": 101.0,
        "put_wall": 99.0,
        "exp": "2026-07-11",
        "pcr": 1.02,
        "atm_iv": 0.18,
    }


def test_synthesize_uses_canonical_pressing_edge_for_high_label():
    lines = synthesize(_pa(78.0, vs_vwap=0.2), _op())

    assert lines[1][0] == "At day HIGHS"
    assert "78% of range" in lines[1][2]


def test_synthesize_uses_canonical_pressing_edge_for_low_label():
    lines = synthesize(_pa(22.0, vs_vwap=-0.2), _op())

    assert lines[1][0] == "At day LOWS"
    assert "22% of range" in lines[1][2]


def test_synthesize_keeps_non_edge_reads_mid_range():
    lines = synthesize(_pa(50.0, vs_vwap=0.02), _op())

    assert lines[1][0] == "Mid-range"
    assert "50% of day range" in lines[1][2]
