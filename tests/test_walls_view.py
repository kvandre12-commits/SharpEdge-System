from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import walls_view as wv

SIGNAL = {
    "spot": 765.72,
    "call_wall": 772.0,
    "put_wall": 765.0,
    "pin": 765.0,
    "max_pain": 768.0,
    "pcr": 1.05,
    "atm_iv": 0.0607,
    "gamma_regime": "negative",
    "day_chg": 0.409,
    "exp": "2026-08-24",
}


def test_position_pct_between_walls():
    assert abs(wv._position_pct(766.0, 765.0, 772.0) - (1 / 7 * 100)) < 1e-9


def test_position_pct_clamps_and_defaults():
    assert wv._position_pct(800.0, 765.0, 772.0) == 100.0
    assert wv._position_pct(700.0, 765.0, 772.0) == 0.0
    assert wv._position_pct(None, None, None) == 50.0
    assert wv._position_pct(766.0, 770.0, 770.0) == 50.0  # equal walls -> midpoint


def test_delta_from_spot_signs():
    assert wv._delta_from_spot(772.0, 765.72).startswith("+6.28")
    assert wv._delta_from_spot(762.0, 765.72).startswith("-3.72")
    assert wv._delta_from_spot(None, 765.72) == "—"


def test_build_walls_html_renders_core_levels():
    doc = wv.build_walls_html(SIGNAL)
    assert "SPY $765.72" in doc
    assert "call wall" in doc and "$772.00" in doc
    assert "put wall" in doc and "$765.00" in doc
    assert "NEG \u03b3 \u00b7 RUNNER" in doc  # negative gamma pill
    assert "6.1%" in doc  # ATM IV rendered as percent
    assert "2026-08-24" in doc  # expiry


def test_positive_regime_pill():
    doc = wv.build_walls_html({**SIGNAL, "gamma_regime": "positive"})
    assert "POS \u03b3 \u00b7 STICKY" in doc


def test_missing_fields_do_not_crash():
    doc = wv.build_walls_html({"spot": 500.0})
    assert "SPY $500.00" in doc
    assert "—" in doc  # missing walls render as em dash
    assert "\u03b3 \u00b7 UNKNOWN" in doc


def test_stale_note_when_no_timestamp():
    doc = wv.build_walls_html(SIGNAL)
    assert "markets closed" in doc
    doc2 = wv.build_walls_html({**SIGNAL, "ts": "2026-08-22T15:59:00Z"})
    assert "as of 2026-08-22T15:59:00Z" in doc2
