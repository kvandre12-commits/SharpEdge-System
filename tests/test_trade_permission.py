from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from trade_permission import score_trade_permission  # noqa: E402


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
        "rng_pos": 92.0,
        "day_chg": 1.2,
        "vwap": closes[-1] - 0.35,
        "vs_vwap": 0.25,
        "mom15": 0.6,
        "vol_mult": 1.7,
    }
    pa.update(overrides)
    return pa


def test_trade_permission_permits_clean_bullish_acceptance():
    bars = _bull_bars()
    pa = _pa(bars)
    levels = {"ORH": bars[-5][4] - 0.4, "ORL": 99.8, "PDC": 99.5}
    op = {"atm_iv": 0.18, "call_wall": 105.0, "put_wall": 99.0}

    card = score_trade_permission(
        bars, pa, levels, [], op, {}, {"premium_read": "cheap"}
    )

    assert card["trade_gate"] in {"PERMIT", "CAUTION"}
    assert card["trade_permission_score"] >= 58
    assert card["bias"] == "CALLS"
    assert card["scores"]["acceptance_score"]["score"] >= 60
    for key in (
        "exhaustion_score",
        "trap_score",
        "dealer_gamma_score",
        "regime_score",
    ):
        assert key in card["scores"]


def test_trade_permission_blocks_midday_thin_chop():
    flat_bars = [
        (150 + idx, 100.0, 100.08, 99.94, 100.01 if idx % 2 else 99.99, 500)
        for idx in range(40)
    ]
    pa = _pa(
        flat_bars,
        spot=100.0,
        rng_pos=50.0,
        vs_vwap=0.01,
        mom15=0.0,
        vol_mult=0.55,
    )
    levels = {"ORH": 101.0, "ORL": 99.0, "PDC": 100.0}

    card = score_trade_permission(flat_bars, pa, levels, [], {"atm_iv": 0.10}, {}, {})

    assert card["trade_gate"] == "BLOCK"
    assert card["trade_permission_score"] < 58
    assert card["scores"]["volume_score"]["score"] <= 42


def test_failed_breakdown_adds_bullish_pressure():
    bars = _bull_bars()
    pa = _pa(bars, rng_pos=35.0)
    setup = {"tag": "FAILED BREAKDOWN", "bias": "CALLS (bullish)", "kind": "ok"}
    levels = {"ORH": 104.0, "ORL": 100.4, "PDC": 99.8}

    card = score_trade_permission(bars, pa, levels, [setup], {"atm_iv": 0.20}, {}, {})

    assert card["bias"] == "CALLS"
    assert card["scores"]["pressure_score"]["score"] >= 80
    assert card["scores"]["rejection_score"]["score"] >= 80
    assert card["scores"]["trap_score"]["score"] >= 90


def test_positive_gamma_pin_dampens_trade_permission():
    bars = _bull_bars()
    pa = _pa(bars, spot=103.6, rng_pos=55.0, vs_vwap=0.02, mom15=0.01)
    levels = {"ORH": 104.0, "ORL": 100.0, "PDC": 100.0}
    op = {"atm_iv": 0.18, "call_wall": 104.0, "put_wall": 100.0}
    gp = {"regime": "positive", "pin": 103.6}

    card = score_trade_permission(bars, pa, levels, [], op, gp, {})

    assert card["scores"]["dealer_gamma_score"]["score"] <= 40
    assert "pinning" in card["scores"]["dealer_gamma_score"]["reason"]


def test_failed_breakout_trap_scores_bearish():
    bars = _bull_bars()
    level = bars[-2][4] - 0.03
    trap_bars = bars + [
        (45, level, level + 0.25, level - 0.02, level - 0.04, 2500),
    ]
    pa = _pa(trap_bars, spot=level - 0.04, rng_pos=82.0, mom15=-0.12)
    levels = {"ORH": level, "ORL": 100.0, "PDC": 100.0}

    card = score_trade_permission(trap_bars, pa, levels, [], {"atm_iv": 0.22}, {}, {})

    assert card["scores"]["trap_score"]["bias"] == "PUTS"
    assert card["scores"]["trap_score"]["score"] >= 78
