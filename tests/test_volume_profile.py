from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from trade_permission import score_trade_permission
from volume_profile import build_volume_profile


def _grind_bars(aligned_volume: bool = True) -> list[tuple]:
    bars = []
    price = 100.0
    for minute in range(40):
        open_ = price
        if minute >= 25:
            close = price + 0.08
            volume = 1400 if aligned_volume else 500
        else:
            close = price + (0.02 if minute % 2 else -0.01)
            volume = 900
        high = max(open_, close) + 0.03
        low = min(open_, close) - 0.02
        bars.append((minute, open_, high, low, close, volume))
        price = close
    return bars


def _pa(bars: list[tuple], profile: dict) -> dict:
    closes = [bar[4] for bar in bars]
    return {
        "spot": closes[-1],
        "day_open": closes[0],
        "hi": max(closes),
        "lo": min(closes),
        "rng_pos": 92.0,
        "day_chg": 1.0,
        "vwap": closes[-1] - 0.25,
        "vs_vwap": 0.24,
        "mom15": 0.55,
        "vol_mult": profile["composite_mult"],
        "volume_profile": profile,
        "balance_high": closes[-2],
        "balance_low": closes[-20],
        "position_in_balance": 1.0,
        "balance_state": "above",
        "balance_label": "ABOVE",
        "balance_width_pct": 0.4,
        "balance_window_bars": 20,
        "balance_reference": "recent_20_bar",
    }


def test_volume_profile_confirms_aligned_local_participation():
    profile = build_volume_profile(_grind_bars(aligned_volume=True))

    assert profile["move_direction"] == "up"
    assert profile["confirmation"] == "confirmed"
    assert profile["local_mult"] > 1.1
    assert profile["aligned_volume_share"] >= 0.58


def test_volume_profile_marks_unparticipated_move_missing():
    profile = build_volume_profile(_grind_bars(aligned_volume=False))

    assert profile["move_direction"] == "up"
    assert profile["confirmation"] in {"missing", "mixed"}
    assert profile["local_mult"] < 0.75


def test_trade_permission_volume_score_uses_move_aware_profile():
    strong_bars = _grind_bars(aligned_volume=True)
    weak_bars = _grind_bars(aligned_volume=False)
    strong_profile = build_volume_profile(strong_bars)
    weak_profile = build_volume_profile(weak_bars)
    levels = {"ORH": 100.2, "ORL": 99.7, "PDC": 99.8}

    strong = score_trade_permission(
        strong_bars,
        _pa(strong_bars, strong_profile),
        levels,
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )
    weak = score_trade_permission(
        weak_bars,
        _pa(weak_bars, weak_profile),
        levels,
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )

    assert (
        strong["scores"]["volume_score"]["score"]
        > weak["scores"]["volume_score"]["score"]
    )
    assert strong["volume_state"]["schema"] == "sharpedge.volume_profile.v1"
    assert strong["volume_state"]["confirmation"] == strong_profile["confirmation"]
    assert "participation" in strong["scores"]["volume_score"]["reason"]
    assert weak["trade_permission_score"] < strong["trade_permission_score"]


def test_trade_permission_rebuilds_volume_state_from_bars_not_pa_packet():
    strong_bars = _grind_bars(aligned_volume=True)
    strong_profile = build_volume_profile(strong_bars)
    fake_profile = {
        **strong_profile,
        "confirmation": "missing",
        "move_direction": "flat",
        "reason": "fake stale packet",
    }
    levels = {"ORH": 100.2, "ORL": 99.7, "PDC": 99.8}

    card = score_trade_permission(
        strong_bars,
        _pa(strong_bars, fake_profile),
        levels,
        [],
        {"atm_iv": 0.18},
        {},
        {},
    )

    assert card["volume_state"]["confirmation"] == strong_profile["confirmation"]
    assert card["volume_state"]["reason"] != "fake stale packet"
    assert card["scores"]["volume_score"]["score"] == 85
