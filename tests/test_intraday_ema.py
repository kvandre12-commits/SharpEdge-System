from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

from make_cockpit import _ema


def test_ema_tracks_constant_and_rising_prices():
    assert _ema([], 9) is None
    assert _ema([100.0] * 20, 9) == 100.0
    assert _ema([100.0, 101.0, 102.0], 3) == 101.25
