from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cockpit"))

import confluence_zones as cz


def _signal(**over):
    base = {
        "spot": 765.72,
        "gamma_regime": "positive",
        "gamma_data_quality": "ok",
        "atm_straddle_mid": 3.4,
        "atm_iv": 0.06,
        "put_wall": 765.0,
        "call_wall": 772.0,
        "pin": 765.0,
        "max_pain": 768.0,
        "ema9": 766.0,
        "ema20": 766.11,
        "line_authority": {"lines": [
            {"name": "ORL", "price": 764.17, "score": 78, "event": "reclaimed"},
            {"name": "VWAP", "price": 766.06, "score": 50, "event": None},
        ]},
        "fair_value_gaps": {"open_gaps": [{"midpoint": 770.5, "direction": "bearish", "fill_state": "open"}]},
        "level_states": {},
    }
    base.update(over)
    return base


# --- factor adapter ---------------------------------------------------------

def test_collect_points_pulls_both_sources_and_skips_none():
    pts = cz._collect_factor_points(_signal(ema20=None))
    names = {p["name"] for p in pts}
    assert {"ORL", "VWAP", "PUT_WALL", "PIN", "MAX_PAIN", "CALL_WALL", "EMA9"} <= names
    assert "EMA20" not in names  # None skipped
    orl = next(p for p in pts if p["name"] == "ORL")
    assert orl["source"] == "line_authority" and orl["authority_multiplier"] == 0.78


def test_collect_points_dedups_static_price_colliding_with_scored_line():
    # A static field sharing an exact price with a scored line is suppressed.
    sig = _signal(line_authority={"lines": [{"name": "PDC", "price": 765.0, "score": 60, "event": None}]})
    pts = cz._collect_factor_points(sig)
    at_765 = [p for p in pts if p["price"] == 765.0]
    assert any(p["name"] == "PDC" for p in at_765)
    # put_wall and pin are both 765.0 -> suppressed in favor of the scored line
    assert not any(p["name"] in {"PUT_WALL", "PIN"} for p in at_765)


# --- expected move + tolerance ----------------------------------------------

def test_expected_move_straddle_first():
    assert cz._expected_move(_signal())["dollars"] == 3.4
    assert cz._expected_move(_signal(atm_straddle_mid=0))["source"] == "none"


def test_tolerance_uses_quarter_em_bounded():
    tol, src = cz._cluster_tolerance(3.4, 765.72)
    assert src == "0.25x_expected_move" and tol == round(0.25 * 3.4, 4)
    # cap: huge EM clamps to 0.4% of spot
    tol_cap, _ = cz._cluster_tolerance(1000.0, 765.72)
    assert tol_cap == round(0.004 * 765.72, 4)
    # no EM -> fallback
    assert cz._cluster_tolerance(None, 765.72)[1] == "fixed_0.15pct_fallback"


# --- clustering -------------------------------------------------------------

def test_cluster_merges_within_and_splits_beyond_tolerance():
    pts = [{"price": p, "name": str(p)} for p in (764.2, 765.0, 765.0, 768.0)]
    clusters = cz._cluster(pts, tol=0.85)
    assert [len(c) for c in clusters] == [3, 1]  # 764.2/765/765 merge; 768 splits
    assert cz._cluster([], 1.0) == []


# --- scoring ----------------------------------------------------------------

def test_score_monotonic_in_factor_weight():
    one = cz._score_zone([{"price": 765, "weight": 1.0, "authority_multiplier": 1.0}], 765.0, 3.4)
    two = cz._score_zone(
        [{"price": 765, "weight": 1.0, "authority_multiplier": 1.0},
         {"price": 765, "weight": 1.0, "authority_multiplier": 1.0}], 765.0, 3.4)
    assert two["raw"] > one["raw"]
    assert one["base"] == round(1.0 / cz._EXPECTED_MAX * 100, 2)


# --- side / stance ----------------------------------------------------------

def test_side_stance_geometry():
    assert cz._side_stance(764.0, 765.72) == ("support", "bounce")
    assert cz._side_stance(770.0, 765.72) == ("resistance", "rejection")


# --- regime gate ------------------------------------------------------------

def _cl(name="ORL", price=764.0):
    return [{"name": name, "price": price, "weight": 0.6, "authority_multiplier": 0.78}]


def test_regime_gate_positive_boosts():
    g = cz._apply_regime_gate("support", _cl(), "positive", {})
    assert g["applied"] == "boost" and g["multiplier"] == 1.15


def test_regime_gate_negative_support_is_trap_vetoed():
    g = cz._apply_regime_gate("support", _cl(), "negative", {})
    assert g["applied"] == "trap_veto" and g["multiplier"] == 0.3


def test_regime_gate_negative_support_unvetoed_by_confirmed_reclaim():
    ls = {"ORL": {"event_state": "failed_break_reclaimed", "entry_window_open": True}}
    g = cz._apply_regime_gate("support", _cl(), "negative", ls)
    assert g["applied"] == "penalty" and g["multiplier"] == 0.6


def test_regime_gate_negative_resistance_penalized():
    assert cz._apply_regime_gate("resistance", _cl(), "negative", {})["applied"] == "penalty"


def test_regime_gate_unknown_is_neutral():
    assert cz._apply_regime_gate("support", _cl(), "unknown", {})["applied"] == "neutral"


# --- assembly + fail-soft ---------------------------------------------------

def test_build_zones_full_signal_is_advisory_and_ranked():
    out = cz.build_confluence_zones(_signal())
    assert out["schema"] == cz.SCHEMA
    assert out["weighted_in_permission"] is False
    assert out["zones"], "expected at least one zone"
    convictions = [z["conviction"] for z in out["zones"]]
    assert convictions == sorted(convictions, reverse=True)  # ranked
    assert out["zones"][0]["rank"] == 1


def test_build_zones_support_cluster_below_spot_in_positive_gamma():
    out = cz.build_confluence_zones(_signal())
    # put_wall 765 + pin 765 + ORL 764.17 sit below spot -> a boosted support zone
    support = [z for z in out["zones"] if z["side"] == "support"]
    assert support
    top = support[0]
    assert top["stance"] == "bounce"
    assert top["regime_gate"]["applied"] == "boost"
    assert top["factor_count"] >= 2
    assert "reclaim required" not in top["trigger"].lower()  # positive gamma adds no reclaim gate


def test_build_zones_negative_gamma_traps_support_bounce():
    out = cz.build_confluence_zones(_signal(gamma_regime="negative"))
    support = [z for z in out["zones"] if z["side"] == "support"]
    assert support and support[0]["regime_gate"]["applied"] == "trap_veto"
    assert "reclaim required" in support[0]["trigger"].lower()


def test_build_zones_failsoft_on_missing_gamma_quality():
    out = cz.build_confluence_zones(_signal(gamma_data_quality="missing"))
    assert out["zones"] == []
    assert "no zones" in out["summary"]["reason"]


def test_build_zones_failsoft_without_spot():
    out = cz.build_confluence_zones(_signal(spot=None))
    assert out["zones"] == []
