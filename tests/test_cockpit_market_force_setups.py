from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from setups import (  # noqa: E402
    detect_negative_gamma_continuation,
    detect_sticky_noise,
)


def test_negative_gamma_continuation_detects_runner_species():
    pa = {"spot": 100.0, "vs_vwap": 0.18, "mom15": 0.22, "vol_mult": 1.6}
    op = {"call_wall": 101.5, "put_wall": 98.8}
    gp = {"regime": "negative", "pin": 99.6}

    card = detect_negative_gamma_continuation(pa, op, gp)

    assert card is not None
    assert card["tag"] == "NEGATIVE GAMMA CONTINUATION"
    assert card["bias"] == "CALLS (runner continuation)"
    assert "above VWAP" in card["detail"]
    assert "volume 1.6x confirms" in card["detail"]
    assert "expansion odds > mean reversion odds" in card["detail"]


def test_negative_gamma_continuation_promotes_recent_exhaustion_into_handoff():
    bars = [
        (0, 100.0, 100.05, 99.95, 100.0, 1000),
        (1, 100.0, 100.02, 98.8, 99.7, 5000),
        (2, 99.2, 99.5, 99.1, 99.45, 1800),
        (3, 99.45, 99.9, 99.4, 99.85, 1700),
        (4, 99.85, 100.2, 99.8, 100.1, 1900),
        (5, 100.1, 100.35, 100.0, 100.3, 2100),
    ]
    pa = {
        "spot": 100.3,
        "vwap": 99.7,
        "vs_vwap": 0.6,
        "mom15": 0.4,
        "vol_mult": 1.9,
        "rng_pos": 83.0,
    }
    op = {"call_wall": 101.5, "put_wall": 98.5}
    gp = {"regime": "negative", "pin": 99.8}

    card = detect_negative_gamma_continuation(pa, op, gp, bars=bars)

    assert card is not None
    assert card["tag"] == "EXHAUSTION -> RUNNER HANDOFF"
    assert card["bias"] == "CALLS (reversal promoted to runner)"
    assert "downside exhaustion" in card["detail"]
    assert "fade has graduated into continuation" in card["detail"]


def test_negative_gamma_continuation_rejects_wall_pinned_breakout():
    pa = {"spot": 100.0, "vs_vwap": 0.24, "mom15": 0.19, "vol_mult": 1.8}
    op = {"call_wall": 100.1, "put_wall": 98.5}
    gp = {"regime": "negative", "pin": 99.4}

    card = detect_negative_gamma_continuation(pa, op, gp)

    assert card is None


def test_sticky_noise_detects_positive_gamma_no_edge_species():
    pa = {"spot": 100.0, "vs_vwap": 0.01, "mom15": 0.01, "vol_mult": 0.8}
    op = {"call_wall": 100.1, "put_wall": 99.9}
    gp = {"regime": "positive", "pin": 100.02}

    card = detect_sticky_noise(pa, op, gp)

    assert card is not None
    assert card["tag"] == "STICKY NOISE"
    assert card["bias"] == "stand down / mean reversion only"
    assert "positive gamma/OI proxy chop context" in card["detail"]
    assert "vs VWAP" in card["detail"]
    assert "not confirming" in card["detail"]
